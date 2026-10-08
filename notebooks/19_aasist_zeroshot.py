"""Run a published anti-spoofing model on this data without training it.

The three detectors compared so far were all trained here, on one corpus, by one person. A
reader who suspects the training setup rather than the data is answered only by a model that
had nothing to do with it. AASIST is graph attention over raw waveform, 297k parameters,
released with weights trained on ASVspoof2019 LA. It runs as released, and only the threshold
is fitted on in-domain dev.

Two limits stated plainly. AASIST has seen neither this corpus nor its bandwidth, since every
file here is bandlimited at 4 kHz where ASVspoof2019 is full-band, so a poor result cannot be
pinned on the Indian corpus specifically, and it may simply miss attacks postdating its
training. It also consumes raw amplitude with no per-utterance normalisation, which made level
invisible to every other model here, so both released and peak-normalised protocols are scored.

Requirements
    Kaggle notebook, GPU optional, internet on for the clone. Under 15 minutes.
    Attached inputs: codeswitch-deepfake-feats, -real, -fakes.
"""

# CELL 1: configuration, mounts, and the split definition
import os
import sys
import glob
import json
import time
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SEED = 1337
N_FIX = 8
SR = 16000

# The released model consumes a fixed 64,600 samples, about 4.04 seconds.
NB_SAMP = 64600
BATCH = 32

REPO = "/kaggle/working/aasist"
REPO_URL = "https://github.com/clovaai/aasist.git"

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
CORPORA = ("hindi_english", "bengali_english")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(SEED)
torch.manual_seed(SEED)


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

DEV = ((IDX.role == "train_source") & (IDX.split == "dev")).values
CONTROL = (IDX.role == "control_indomain").values

SPK = {c: sorted(IDX[IDX.set == c].speaker.dropna().unique()) for c in CORPORA}
rng = np.random.RandomState(SEED)
ADDED = {c: sorted(rng.choice(SPK[c], N_FIX, replace=False)) for c in CORPORA}

mask_added = np.zeros(len(IDX), dtype=bool)
for c, spks in ADDED.items():
    mask_added |= ((IDX.set == c) & IDX.speaker.isin(spks)).values

# Those speakers are excluded here too. AASIST never trained on them, so this costs nothing in
# fairness, but it keeps every row of the comparison table drawn from the same population.
HELD = IDX.set.isin(list(CORPORA)).values & ~mask_added

print(f"{DEVICE}, {len(IDX):,} utterances, {Y.sum():,} fake, {int(HELD.sum()):,} held out")


# CELL 2: the released model, loaded and verified before any audio is read
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
sys.path.insert(0, REPO)

CONF = json.load(open(f"{REPO}/config/AASIST.conf"))
WEIGHTS = os.path.normpath(os.path.join(REPO, CONF["model_path"]))

from models.AASIST import Model as AASIST

model = AASIST(CONF["model_config"]).to(DEVICE)
state = torch.load(WEIGHTS, map_location=DEVICE)
# Released under DataParallel, so keys may carry a module prefix the bare model does not have.
state = {k.replace("module.", "", 1): v for k, v in state.items()}
report = model.load_state_dict(state, strict=False)

# A silent partial load would leave a randomly initialised graph attention block scoring
# everything, and the result would look like a finding.
assert not report.missing_keys, f"missing weights: {report.missing_keys[:5]}"
assert not report.unexpected_keys, f"unexpected weights: {report.unexpected_keys[:5]}"
model.eval()

print(f"AASIST loaded from {WEIGHTS}")
print(f"{sum(p.numel() for p in model.parameters()):,} parameters, all weights matched")


# CELL 3: audio, in the released model's own protocol and in a level control
from torch.utils.data import Dataset, DataLoader


def pad(x, n):
    """Repeat the signal to length rather than pad with zeros, as the released evaluation does."""
    if len(x) >= n:
        return x[:n]
    return np.tile(x, int(n / len(x)) + 1)[:n]


class Clips(Dataset):
    def __init__(self):
        self.paths = IDX.path.values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wav, sr = sf.read(self.paths[i], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
        wav = pad(wav, NB_SAMP)
        peak = float(np.abs(wav).max()) or 1.0
        return torch.from_numpy(wav), torch.from_numpy(wav / peak), i


loader = DataLoader(Clips(), batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)


# CELL 4: score every utterance twice
def bonafide(out):
    """The released code scores with the bonafide logit, so higher means more genuine."""
    if isinstance(out, (tuple, list)):
        out = out[-1]
    return out[:, 1]


raw_scores = np.empty(len(IDX), dtype=np.float32)
norm_scores = np.empty(len(IDX), dtype=np.float32)

t0 = time.time()
with torch.no_grad():
    for wav, wav_norm, rows in loader:
        r = rows.numpy()
        # Negated so that, as everywhere else in this project, a higher score means more fake.
        raw_scores[r] = -bonafide(model(wav.to(DEVICE))).float().cpu().numpy()
        norm_scores[r] = -bonafide(model(wav_norm.to(DEVICE))).float().cpu().numpy()
print(f"scored {len(IDX):,} utterances twice in {time.time()-t0:.0f}s")

agree = float(np.corrcoef(raw_scores, norm_scores)[0, 1])
print(f"released protocol against the peak-normalised control: correlation {agree:.4f}")
if agree < 0.9:
    print("level is moving this model's scores; read the two columns separately")


# CELL 5: threshold, metrics, and the tables
from sklearn.metrics import roc_curve, roc_auc_score

SCORES = {}
for name, raw in (("aasist_zeroshot", raw_scores), ("aasist_zeroshot_levelnorm", norm_scores)):
    d = raw[DEV]
    lo, hi = d[Y[DEV] == 1].min(), d[Y[DEV] == 0].max()
    if lo > hi:
        thr, sep = float((lo + hi) / 2), "separable"
    else:
        fpr, tpr, cut = roc_curve(Y[DEV], d)
        thr, sep = float(cut[int(np.nanargmin(np.abs((1 - tpr) - fpr)))]), "overlapping"
    SCORES[name] = raw - thr
    print(f"{name}: dev {sep}, threshold {thr:+.3f}")

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
for model_name, score in SCORES.items():
    for label, mask in CONDITIONS:
        if mask.sum():
            rows.append(dict(model=model_name, condition=label, **metrics(Y[mask], score[mask])))

M = pd.DataFrame(rows)
M.to_csv(f"{OUT}/metrics_aasist.csv", index=False)

LEAF = [(l, m) for l, m in CONDITIONS if "corpus, all" not in l and "pooled" not in l]
cond_of = np.full(len(IDX), "", dtype=object)
for label, mask in LEAF:
    cond_of[mask] = label
keep = cond_of != ""

RAW = pd.DataFrame({"row": np.flatnonzero(keep), "set": IDX.set.values[keep],
                    "speaker": IDX.speaker.values[keep], "cs_bin": CS[keep],
                    "condition": cond_of[keep], "label": IDX.label.values[keep], "y": Y[keep]})
for model_name, score in SCORES.items():
    RAW[f"score_{model_name}"] = score[keep]
RAW.to_csv(f"{OUT}/scores_aasist.csv", index=False)


def fmt(v, dp=1):
    return "-" if v != v else f"{v:.{dp}f}"


print("\n| Model | Condition | n | Acc | AUC | EER | FAR | FRR (95% CI) |")
print("|---|---|---|---|---|---|---|---|")
for _, r in M.iterrows():
    print(f"| {r.model} | {r.condition} | {r.n:,} | {fmt(r.accuracy)} | {fmt(r.auc, 2)} | "
          f"{fmt(r.eer, 2)} | {fmt(r.far, 2)} | {fmt(r.frr)} "
          f"({fmt(r.frr_lo)}-{fmt(r.frr_hi)}) |")

print("\nwhat this adds to the comparison\n")
for model_name in SCORES:
    g = M[M.model == model_name].set_index("condition")
    dev = g.loc["in-domain dev, unseen speakers"]
    pooled = g.loc["out of corpus, pooled"]
    print(f"{model_name}: on this project's in-domain dev, EER {fmt(dev.eer, 2)}%. "
          f"Out of corpus, accuracy {fmt(pooled.accuracy)}%, misses {fmt(pooled.far, 2)}% of "
          f"fakes and rejects {fmt(pooled.frr)}% of genuine speech")

print("""
Read the two rows against each other before drawing a conclusion. A model that misses most
fakes has failed in the ordinary cross-dataset way, because its training attacks predate these
generators, and that is a different result from the asymmetry this project documents. A model
that catches the fakes and rejects the genuine speech has reproduced it on weights nobody here
trained.
""")
