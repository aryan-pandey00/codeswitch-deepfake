"""Fine-tune XLS-R with LoRA adapters, to separate the frozen frontend from the finding.

The linear probe leaves the frontend untouched, which is what makes its result clean and also
leaves an opening: a frozen representation cannot adapt to a new recording condition, so a
reader can ask whether the collapse is the cost of freezing. This lets the encoder move.
Adapters go into the query and value projections of every retained block, roughly 657,000
trainable parameters out of 315 million, on the same splits with the same speakers held out.

They are implemented here rather than pulled from a library: about twenty lines, no three-way
version dependency in a notebook that runs unattended, and the method section can state what
was adapted instead of naming a package.

Every clip is scored on a centred five-second span, which is what the service windows, where
the cached probe numbers pooled whole utterances. Utterances average 4.7 s so they mostly agree.

Requirements
    Kaggle notebook, GPU T4, roughly 1.5 hours.
    Attached inputs: codeswitch-deepfake-feats (index.csv only), -real, -fakes.
"""

# CELL 1: configuration, mounts, and the split definition
import os
import sys
import glob
import time
import math

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn

CHECKPOINT = "facebook/wav2vec2-xls-r-300m"
LAYER = 20
SR = 16000
SEED = 1337
N_FIX = 8

SPAN_S = 5.0
SPAN = int(SPAN_S * SR)

RANK = 8
ALPHA = 16
EPOCHS = 4
BATCH = 8
LR_ADAPTER = 1e-4
LR_HEAD = 1e-3

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
CORPORA = ("hindi_english", "bengali_english")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


def find(name):
    """Kaggle mounts inputs under the notebook slug, not the dataset name."""
    return next(
        (p for depth in ("*", "*/*", "*/*/*")
         for p in sorted(glob.glob(f"/kaggle/input/{depth}/{name}")) if os.path.isdir(p)),
        None)


FEAT_DIR = find("feats")
FAKE_DIR = find("fakes")

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"

if FEAT_DIR is None or FAKE_DIR is None or not os.path.isdir(REAL_ROOT):
    sys.exit("attach codeswitch-deepfake-feats, -real and -fakes")

IDX = pd.read_csv(f"{FEAT_DIR}/index.csv")
Y = (IDX.label == "fake").values.astype(int)

IDX["path"] = [
    f"{FAKE_DIR}/{os.path.basename(p)}" if lab == "fake"
    else f"{REAL_ROOT}/audio/{os.path.basename(p)}"
    for p, lab in zip(IDX.file, IDX.label)
]

missing = [p for p in IDX.path.sample(200, random_state=SEED) if not os.path.exists(p)]
if missing:
    sys.exit(f"{len(missing)} of 200 sampled paths do not resolve, first {missing[0]}")

BASE_TRAIN = ((IDX.role == "train_source") & (IDX.split == "train")).values
DEV = ((IDX.role == "train_source") & (IDX.split == "dev")).values
CONTROL = (IDX.role == "control_indomain").values

SPK = {c: sorted(IDX[IDX.set == c].speaker.dropna().unique()) for c in CORPORA}
rng = np.random.RandomState(SEED)
ADDED = {c: sorted(rng.choice(SPK[c], N_FIX, replace=False)) for c in CORPORA}

mask_added = np.zeros(len(IDX), dtype=bool)
for c, spks in ADDED.items():
    mask_added |= ((IDX.set == c) & IDX.speaker.isin(spks)).values

HELD = IDX.set.isin(list(CORPORA)).values & ~mask_added
TRAIN = {"baseline": BASE_TRAIN, "fixed": BASE_TRAIN | mask_added}

print(f"{DEVICE}, {len(IDX):,} utterances, {Y.sum():,} fake")
print(f"baseline trains on {int(BASE_TRAIN.sum()):,}, fixed on {int(TRAIN['fixed'].sum()):,}")


# CELL 2: adapters and the model, built before any data is touched
from transformers import AutoModel


class LoRALinear(nn.Module):
    """A frozen projection plus a trainable low-rank correction.

    B starts at zero so the adapted model is exactly the pretrained one at step zero. Without
    that, training begins by undoing a random perturbation of a well-trained network.
    """

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.a.weight, std=0.02)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class Detector(nn.Module):
    """Truncated XLS-R with adapters, pooled to mean and standard deviation, then one logit."""

    def __init__(self):
        super().__init__()
        enc = AutoModel.from_pretrained(CHECKPOINT, attn_implementation="eager")
        assert enc.config.do_stable_layer_norm, "the trailing norm rule below assumes this"
        enc.encoder.layers = nn.ModuleList(list(enc.encoder.layers[:LAYER]))
        # The encoder's final layer_norm belongs to hidden state 24 alone. Hidden state 20 is a
        # raw block output, so keeping the norm would produce a different tensor than every
        # other result in this project was computed from.
        enc.encoder.layer_norm = nn.Identity()
        enc.config.num_hidden_layers = LAYER

        # The convolutional feature extractor stays frozen and unadapted throughout. It is the
        # part most likely to memorise a recording chain, and leaving it fixed keeps the
        # comparison against the frozen probe honest.
        for p in enc.parameters():
            p.requires_grad = False

        # Patched by walking the module tree rather than by attribute path, so a rename in the
        # library surfaces as a failed count here instead of as a model that silently trained
        # with no adapters at all.
        patched = 0
        for block in enc.encoder.layers:
            attn = block.attention
            for proj in ("q_proj", "v_proj"):
                if not hasattr(attn, proj):
                    continue
                setattr(attn, proj, LoRALinear(getattr(attn, proj), RANK, ALPHA))
                patched += 1
        assert patched == 2 * LAYER, f"adapted {patched} projections, expected {2 * LAYER}"

        self.enc = enc
        self.head = nn.Linear(2 * enc.config.hidden_size, 1)

    def forward(self, wav):
        h = self.enc(wav).last_hidden_state
        # Biased standard deviation, matching numpy and therefore matching what every other
        # result in this project pooled. Torch defaults to the unbiased estimator.
        pooled = torch.cat([h.mean(1), h.std(1, unbiased=False)], dim=-1)
        return self.head(pooled).squeeze(-1)


probe = Detector()
n_train = sum(p.numel() for p in probe.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in probe.parameters())
print(f"{n_train:,} trainable of {n_all:,} parameters, {100*n_train/n_all:.2f}%")
del probe


# CELL 3: data
from torch.utils.data import Dataset, DataLoader


def take_span(wav, train):
    """Wrapped rather than zero-padded when short, because generated clips run systematically
    shorter than the reals and the amount of padding would otherwise carry the label."""
    if len(wav) < SPAN:
        wav = np.pad(wav, (0, SPAN - len(wav)), mode="wrap")
        return wav[:SPAN]
    start = np.random.randint(0, len(wav) - SPAN + 1) if train else (len(wav) - SPAN) // 2
    return wav[start: start + SPAN]


class Clips(Dataset):
    def __init__(self, rows, train):
        self.rows = np.asarray(rows)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        wav, sr = sf.read(IDX.path.values[r], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
        wav = take_span(wav, self.train)
        wav = (wav - wav.mean()) / math.sqrt(wav.var() + 1e-7)
        return torch.from_numpy(wav.astype(np.float32)), float(Y[r]), r


def loader(rows, train):
    return DataLoader(Clips(rows, train), batch_size=BATCH, shuffle=train,
                      num_workers=2, pin_memory=True, drop_last=False)


# CELL 4: train both heads
def evaluate(model, dl):
    model.eval()
    logits, labels, rows = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        for wav, y, r in dl:
            logits.append(model(wav.to(DEVICE, non_blocking=True)).float().cpu())
            labels.append(y)
            rows.append(r)
    return (torch.cat(logits).numpy(), torch.cat(labels).numpy(),
            torch.cat(rows).numpy().astype(int))


MODELS, CURVES = {}, []
dev_dl = loader(np.flatnonzero(DEV), False)
lossfn = nn.BCEWithLogitsLoss()

for name, train_mask in TRAIN.items():
    torch.manual_seed(SEED)
    model = Detector().to(DEVICE)
    opt = torch.optim.AdamW([
        {"params": [p for n_, p in model.enc.named_parameters() if p.requires_grad],
         "lr": LR_ADAPTER},
        {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
    trainable = {n_ for n_, p in model.named_parameters() if p.requires_grad}
    train_dl = loader(np.flatnonzero(train_mask), True)

    best, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for wav, y, _ in train_dl:
            wav, y = wav.to(DEVICE, non_blocking=True), y.to(DEVICE).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = lossfn(model(wav), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss) * len(y)
            seen += len(y)

        z, yy, _ = evaluate(model, dev_dl)
        val_loss = float(nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(z), torch.from_numpy(yy)))
        val_acc = float(((z > 0) == (yy > 0.5)).mean())
        print(f"{name} epoch {epoch}: loss {total/seen:.4f} val_loss {val_loss:.4f} "
              f"val_acc {val_acc:.4f} in {time.time()-t0:.0f}s")
        CURVES.append(dict(model=f"lora_{name}", epoch=epoch, loss=total / seen,
                           val_loss=val_loss, val_accuracy=val_acc))

        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    MODELS[name] = model
    # Only the adapters and the head. The pretrained weights are unchanged and reloadable from
    # the checkpoint, so storing them again would make a 1.2 GB file out of 2.6 MB of training.
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if k in trainable},
               f"{OUT}/lora_{name}.pt")

pd.DataFrame(CURVES).to_csv(f"{OUT}/curves_lora.csv", index=False)
print(f"wrote curves_lora.csv, {len(CURVES)} epochs total")


# CELL 5: score every utterance
from sklearn.metrics import roc_curve, roc_auc_score

all_dl = loader(np.arange(len(IDX)), False)

SCORES = {}
for name, model in MODELS.items():
    t0 = time.time()
    z, _, rows = evaluate(model, all_dl)
    raw = np.empty(len(IDX), dtype=np.float32)
    raw[rows] = z

    d = raw[DEV]
    lo, hi = d[Y[DEV] == 1].min(), d[Y[DEV] == 0].max()
    if lo > hi:
        thr, sep = float((lo + hi) / 2), "separable"
    else:
        fpr, tpr, cut = roc_curve(Y[DEV], d)
        thr, sep = float(cut[int(np.nanargmin(np.abs((1 - tpr) - fpr)))]), "overlapping"
    SCORES[f"lora_{name}"] = raw - thr
    print(f"{name}: dev {sep}, threshold {thr:+.3f}, scored in {time.time()-t0:.0f}s")


# CELL 6: metrics, in the same shape as notebook 16 so the tables concatenate
CS = IDX.cs_bin.fillna("").values

CONDITIONS = [
    ("in-domain dev, unseen speakers", DEV),
    ("in-domain control, official test partition", CONTROL),
    ("Hindi-English corpus, all", HELD & (IDX.set == "hindi_english").values),
    ("Hindi-English corpus, monolingual utterances only",
     HELD & (IDX.set == "hindi_english").values & (CS == "mono")),
    ("Hindi-English corpus, code-switched utterances only",
     HELD & (IDX.set == "hindi_english").values & (CS != "mono")),
    ("Bengali-English corpus, all", HELD & (IDX.set == "bengali_english").values),
    ("Bengali-English corpus, monolingual utterances only",
     HELD & (IDX.set == "bengali_english").values & (CS == "mono")),
    ("Bengali-English corpus, code-switched utterances only",
     HELD & (IDX.set == "bengali_english").values & (CS != "mono")),
    ("out of corpus, pooled", HELD),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def eer(y, s):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    return 100 * float((fpr + fnr)[int(np.nanargmin(np.abs(fnr - fpr)))] / 2)


def metrics(y, s):
    pred = (s > 0).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    n_fake, n_real = tp + fn, fp + tn
    nan = float("nan")
    prec = tp / (tp + fp) if tp + fp else nan
    rec = tp / n_fake if n_fake else nan
    f1 = 2 * prec * rec / (prec + rec) if tp + fp and n_fake and prec + rec else nan
    far = fn / n_fake if n_fake else nan
    frr = fp / n_real if n_real else nan
    frr_lo, frr_hi = wilson(fp, n_real)
    far_lo, far_hi = wilson(fn, n_fake)
    both = n_fake > 0 and n_real > 0
    return dict(
        n=len(y), n_genuine=n_real, n_fake=n_fake, tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=100 * (tp + tn) / len(y) if len(y) else nan,
        balanced_accuracy=100 * ((rec + (1 - frr)) / 2) if both else nan,
        precision=100 * prec if prec == prec else nan,
        recall=100 * rec if rec == rec else nan,
        f1=100 * f1 if f1 == f1 else nan,
        far=100 * far if far == far else nan, far_lo=far_lo, far_hi=far_hi,
        frr=100 * frr if frr == frr else nan, frr_lo=frr_lo, frr_hi=frr_hi,
        auc=100 * roc_auc_score(y, s) if both else nan,
        eer=eer(y, s) if both else nan,
    )


rows = []
for model, score in SCORES.items():
    for label, mask in CONDITIONS:
        if mask.sum():
            rows.append(dict(model=model, condition=label, **metrics(Y[mask], score[mask])))

M = pd.DataFrame(rows)
M.to_csv(f"{OUT}/metrics_lora.csv", index=False)

LEAF = [(l, m) for l, m in CONDITIONS if "corpus, all" not in l and "pooled" not in l]
cond_of = np.full(len(IDX), "", dtype=object)
for label, mask in LEAF:
    cond_of[mask] = label
keep = cond_of != ""

RAW = pd.DataFrame({"row": np.flatnonzero(keep), "set": IDX.set.values[keep],
                    "speaker": IDX.speaker.values[keep], "cs_bin": CS[keep],
                    "condition": cond_of[keep], "label": IDX.label.values[keep], "y": Y[keep]})
for model, score in SCORES.items():
    RAW[f"score_{model}"] = score[keep]
RAW.to_csv(f"{OUT}/scores_lora.csv", index=False)


def fmt(v, dp=1):
    return "-" if v != v else f"{v:.{dp}f}"


print("\n| Model | Condition | n | Acc | AUC | EER | FAR | FRR (95% CI) |")
print("|---|---|---|---|---|---|---|---|")
for _, r in M.iterrows():
    print(f"| {r.model} | {r.condition} | {r.n:,} | {fmt(r.accuracy)} | {fmt(r.auc, 2)} | "
          f"{fmt(r.eer, 2)} | {fmt(r.far, 2)} | {fmt(r.frr)} "
          f"({fmt(r.frr_lo)}-{fmt(r.frr_hi)}) |")

print("\nthe question this notebook exists to answer\n")
for model in SCORES:
    g = M[M.model == model].set_index("condition")
    dev = g.loc["in-domain dev, unseen speakers"]
    pooled = g.loc["out of corpus, pooled"]
    print(f"{model}: in-domain EER {fmt(dev.eer, 2)}%, out of corpus accuracy "
          f"{fmt(pooled.accuracy)}%, misses {fmt(pooled.far, 2)}% of fakes and rejects "
          f"{fmt(pooled.frr)}% of genuine speech")

print("\nThe frozen probe rejected 29.2% of out-of-corpus genuine speech as the baseline. If an "
      "adapted encoder trained on the same corpus rejects a comparable fraction, then freezing "
      "is not what caused the collapse, and the remaining explanation is the training data.")
