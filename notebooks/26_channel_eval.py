"""Score the channel grid with every architecture, at the threshold each was deployed on.

Notebook 23 put one set of utterances through five transmission channels. This scores them
with the four detectors and produces the channel by language table.

The decision rule is the point. Each threshold is fitted on clean in-domain dev and nothing
else, the way a deployment would have to fit it, then applied unchanged to every channel. A
threshold refitted per channel would measure the thresholds rather than the detectors.

The grid contains a clean channel, so clean-condition rejection has to reproduce the published
numbers. If it does not, the scoring path is wrong and nothing else in the run means anything.
Cell 7 checks that before the table is written.

Each architecture writes its own CSV as it finishes, so a session that dies partway still
leaves completed rows behind rather than nothing.

Requirements
    Kaggle notebook, GPU T4. Internet on for the XLS-R checkpoint and the AASIST clone.
    Attached inputs: notebook 23's output, 17's output, 18's output, 14's output,
    codeswitch-deepfake-feats, codeswitch-deepfake-real, codeswitch-deepfake-fakes.
"""

# CELL 1: mounts and config

import os
import sys
import glob
import json
import math
import time

import numpy as np
import pandas as pd
import soundfile as sf

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
OUT = "/kaggle/working"

SR = 16000
SEED = 1337
LAYER = 20
SPAN_S = 5.0
SPAN = int(SPAN_S * SR)

N_MFCC, N_FFT, HOP, FRAMES = 40, 400, 160, 500
RANK, ALPHA = 8, 16
NB_SAMP = 64600
CHECKPOINT = "facebook/wav2vec2-xls-r-300m"
SMOKE = 0


def find(pattern):
    """Mount paths depend on the producing notebook's slug, so locate by content."""
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        hits = sorted(glob.glob(f"/kaggle/input/{depth}/{pattern}"))
        if hits:
            return hits[0]
    return None


GRID_MAN = find("manifest_channels.csv")
FEAT_IDX = find("feats/index.csv") or find("index.csv")
HEADS_JSON = find("heads.json")
if GRID_MAN is None:
    sys.exit("manifest_channels.csv not found, attach notebook 23's output")
if FEAT_IDX is None:
    sys.exit("index.csv not found, attach codeswitch-deepfake-feats")
if HEADS_JSON is None:
    sys.exit("heads.json not found, attach notebook 14's output")

GRID_ROOT = os.path.dirname(GRID_MAN)
FAKE_DIR = find("fakes")
GRID = pd.read_csv(GRID_MAN)
GRID = GRID[GRID.status == "ok"].reset_index(drop=True)
# `path` is already "<channel>/<name>.flac". The audio sits under channels/ in the producing
# notebook's working directory, but a mount can land either side of that, so both are tried.
GRID["file"] = GRID_ROOT + "/channels/" + GRID.path
miss = ~GRID.file.map(os.path.exists)
GRID.loc[miss, "file"] = GRID_ROOT + "/" + GRID.path[miss]
bad = ~GRID.file.map(os.path.exists)
if bad.any():
    print(GRID.file[bad].head(3).to_string())
    sys.exit(f"{int(bad.sum())} grid clips not found under {GRID_ROOT}")
GRID["y"] = (GRID.label == "fake").astype(int)
if SMOKE:
    # Stratified, so every channel and both labels are still exercised.
    GRID = (GRID.groupby(["channel", "label"], group_keys=False)
                .head(max(1, SMOKE // (GRID.channel.nunique() * 2))).reset_index(drop=True))
    print(f"SMOKE RUN, {len(GRID)} grid clips. The cell 7 gate is meaningless at this size.")
print(f"grid {len(GRID):,} clips, {GRID.channel.nunique()} channels")


# CELL 2: the calibration set, which the grid does not contain

# Thresholds come from clean in-domain dev, so those clips are read from the original datasets
# rather than from the grid. The training split is also needed, for the MFCC statistics
# notebook 17 computed but did not save.
IDX = pd.read_csv(FEAT_IDX)
IDX["y"] = (IDX.label == "fake").astype(int)


def resolve(row):
    if row.label == "fake":
        return f"{FAKE_DIR}/{os.path.basename(str(row.file))}"
    return f"{REAL_ROOT}/audio/{os.path.basename(str(row.file))}"


DEV = IDX[(IDX.role == "train_source") & (IDX.split == "dev")].copy()
TRAIN = IDX[(IDX.role == "train_source") & (IDX.split == "train")].copy()
for d in (DEV, TRAIN):
    d["file"] = d.apply(resolve, axis=1)
DEV = DEV[DEV.file.map(os.path.exists)].reset_index(drop=True)
TRAIN = TRAIN[TRAIN.file.map(os.path.exists)].reset_index(drop=True)
if SMOKE:
    # Both labels kept, or the threshold rule has nothing to separate.
    DEV = DEV.groupby("label", group_keys=False).head(SMOKE // 2).reset_index(drop=True)
    TRAIN = TRAIN.groupby("label", group_keys=False).head(SMOKE // 2).reset_index(drop=True)
print(f"dev {len(DEV):,} for calibration, train {len(TRAIN):,} for MFCC statistics")
assert len(DEV) and len(TRAIN), "dev or train clips did not resolve, check the audio mounts"


def threshold_of(dev_scores, dev_y):
    """The rule notebooks 17 and 18 both used: the gap midpoint when dev separates, the equal
    error point when it does not."""
    from sklearn.metrics import roc_curve
    lo, hi = dev_scores[dev_y == 1].min(), dev_scores[dev_y == 0].max()
    if lo > hi:
        return float((lo + hi) / 2), "separable"
    fpr, tpr, cut = roc_curve(dev_y, dev_scores)
    return float(cut[int(np.nanargmin(np.abs((1 - tpr) - fpr)))]), "overlapping"


def emit(arch, variant, scores, thr):
    """One CSV per architecture as it finishes, so a dead session keeps what is done."""
    df = GRID[["channel", "condition_grid", "label", "speaker", "fname", "y"]].copy()
    df["arch"] = arch
    df["variant"] = variant
    df["score"] = scores - thr
    path = f"{OUT}/grid_scores_{arch}_{variant}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {os.path.basename(path)}")
    return df


PARTS = []


# CELL 3: frozen XLS-R probe, both heads

# The exported heads carry their threshold folded into the bias, so this path needs no
# calibration and is the one that can be checked against the shipped artifact directly.
import torch
from transformers import AutoFeatureExtractor, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEADS = json.load(open(HEADS_JSON))
print(f"device {DEVICE}, heads {list(HEADS['heads'])}, layer {HEADS['meta']['layer']}")

fe = AutoFeatureExtractor.from_pretrained(CHECKPOINT)
enc = AutoModel.from_pretrained(CHECKPOINT).to(DEVICE).eval()


@torch.no_grad()
def pooled_layer(paths, layer=LAYER, batch=8):
    """Mean and std over valid frames of one hidden state. Notebook 09's recipe, one layer."""
    out = np.empty((len(paths), 2 * enc.config.hidden_size), dtype=np.float32)
    for i in range(0, len(paths), batch):
        chunk = paths[i: i + batch]
        waves = []
        for p in chunk:
            w, _ = sf.read(p, dtype="float32", always_2d=False)
            waves.append(w.mean(1) if w.ndim > 1 else w)
        e = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True,
               return_attention_mask=True)
        iv, am = e.input_values.to(DEVICE), e.attention_mask.to(DEVICE).long()
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            h = enc(iv, attention_mask=am, output_hidden_states=True).hidden_states[layer]
        lens = enc._get_feat_extract_output_lengths(am.sum(-1)).to(torch.long)
        m = (torch.arange(h.shape[1], device=DEVICE)[None] < lens[:, None]).float()
        n = m.sum(1).clamp(min=1)[:, None]
        h = h.float() * m[:, :, None]
        mean = h.sum(1) / n
        var = (h.pow(2).sum(1) / n - mean.pow(2)).clamp(min=0)
        out[i: i + len(chunk)] = torch.cat([mean, var.sqrt()], 1).cpu().numpy()
        if i and i % 2000 == 0:
            print(f"    {i:,}/{len(paths):,}")
    return out


t0 = time.time()
F_GRID = pooled_layer(GRID.file.tolist())
print(f"frozen features in {time.time() - t0:.0f}s")

for name, h in HEADS["heads"].items():
    mu, sd, w, b = (np.asarray(h[k], dtype=np.float32) for k in ("mu", "sd", "w", "b"))
    s = ((F_GRID - mu) / sd) @ w + float(h["b"])
    PARTS.append(emit("xlsr_frozen", name, s, 0.0))

del enc
torch.cuda.empty_cache()


# CELL 4: XLS-R with LoRA adapters, both heads

import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class Detector(nn.Module):
    """Copied from notebook 18 rather than re-derived. A difference here would be a different
    model wearing the same weights."""

    def __init__(self):
        super().__init__()
        e = AutoModel.from_pretrained(CHECKPOINT, attn_implementation="eager")
        e.encoder.layers = nn.ModuleList(list(e.encoder.layers[:LAYER]))
        e.encoder.layer_norm = nn.Identity()
        e.config.num_hidden_layers = LAYER
        patched = 0
        for block in e.encoder.layers:
            for proj in ("q_proj", "v_proj"):
                if hasattr(block.attention, proj):
                    setattr(block.attention, proj,
                            LoRALinear(getattr(block.attention, proj), RANK, ALPHA))
                    patched += 1
        assert patched == 2 * LAYER, f"adapted {patched}, expected {2 * LAYER}"
        self.enc = e
        self.head = nn.Linear(2 * e.config.hidden_size, 1)

    def forward(self, wav):
        h = self.enc(wav).last_hidden_state
        return self.head(torch.cat([h.mean(1), h.std(1, unbiased=False)], -1)).squeeze(-1)


def take_span(wav):
    """Centred, and wrapped rather than zero-padded when short, as in notebooks 17 and 18."""
    if len(wav) < SPAN:
        return np.pad(wav, (0, SPAN - len(wav)), mode="wrap")[:SPAN]
    start = (len(wav) - SPAN) // 2
    return wav[start: start + SPAN]


def read_span(path):
    w, _ = sf.read(path, dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(1)
    w = take_span(w)
    return (w - w.mean()) / math.sqrt(w.var() + 1e-7)


@torch.no_grad()
def score_lora(model, paths, label, batch=8):
    out = np.empty(len(paths), dtype=np.float32)
    for i in range(0, len(paths), batch):
        wav = np.stack([read_span(p) for p in paths[i: i + batch]])
        t = torch.from_numpy(wav.astype(np.float32)).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out[i: i + len(wav)] = model(t).float().cpu().numpy()
        if i and i % 2000 == 0:
            print(f"    {label} {i:,}/{len(paths):,}")
    return out


for name in ("baseline", "fixed"):
    ckpt = find(f"lora_{name}.pt")
    if ckpt is None:
        print(f"lora_{name}.pt not mounted, skipping this head")
        continue
    m = Detector().to(DEVICE).eval()
    report = m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    assert not report.unexpected_keys, f"unexpected keys in lora_{name}.pt"
    t0 = time.time()
    print(f"lora {name}: scoring dev")
    d = score_lora(m, DEV.file.tolist(), f"{name} dev")
    thr, sep = threshold_of(d, DEV.y.values)
    print(f"lora {name}: dev {sep}, threshold {thr:+.3f}, now scoring the grid")
    g = score_lora(m, GRID.file.tolist(), f"{name} grid")
    print(f"lora {name}: done in {time.time() - t0:.0f}s")
    PARTS.append(emit("xlsr_lora", name, g, thr))
    del m
    torch.cuda.empty_cache()


# CELL 5: MFCC CNN-BiLSTM, both heads

import librosa
from joblib import Parallel, delayed


def mfcc_of(path):
    w, _ = sf.read(path, dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(1)
    w = take_span(w)
    m = librosa.feature.mfcc(y=w, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    # float16 as in notebook 17, which is also the dtype its statistics were computed through.
    return m[:, :FRAMES].astype(np.float16)


def mfcc_batch(paths, label):
    t0 = time.time()
    out = np.stack(Parallel(n_jobs=4, batch_size=64)(delayed(mfcc_of)(p) for p in paths))
    print(f"  {label} mfcc {out.shape} in {time.time() - t0:.0f}s")
    return out


# Notebook 17 standardised with statistics from the training split and did not save them.
# They are deterministic given the same split and parameters, so they are recomputed here.
X_TRAIN = mfcc_batch(TRAIN.file.tolist(), "train")
MU = X_TRAIN.astype(np.float32).mean((0, 2), keepdims=True)
SD = X_TRAIN.astype(np.float32).std((0, 2), keepdims=True) + 1e-6
del X_TRAIN
print(f"  mfcc stats recomputed from {len(TRAIN):,} training clips")

X_DEV = mfcc_batch(DEV.file.tolist(), "dev")
X_GRID = mfcc_batch(GRID.file.tolist(), "grid")

from tensorflow import keras


def predict_all(model, X):
    """Batched because the standardised copy is float32 and the grid array is large."""
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), 256):
        xb = (X[i: i + 256].astype(np.float32) - MU) / SD
        out[i: i + 256] = model.predict(xb[..., None], verbose=0).ravel()
    return out


for name in ("baseline", "fixed"):
    path = find(f"cnn_bilstm_{name}.keras")
    if path is None:
        print(f"cnn_bilstm_{name}.keras not mounted, skipping this head")
        continue
    model = keras.models.load_model(path, compile=False)
    thr, sep = threshold_of(predict_all(model, X_DEV), DEV.y.values)
    print(f"cnn_bilstm {name}: dev {sep}, threshold {thr:.4f}")
    PARTS.append(emit("cnn_bilstm", name, predict_all(model, X_GRID), thr))

del X_DEV, X_GRID


# CELL 6: AASIST, released weights, no training

# Released protocol is the first 64,600 samples with no normalisation. Peak-normalised is
# scored alongside because this model is the only one that can read absolute level.
AASIST_DIR = find("aasist") or "/kaggle/working/aasist"
if not os.path.isdir(AASIST_DIR):
    os.system("git clone -q https://github.com/clovaai/aasist /kaggle/working/aasist")
    AASIST_DIR = "/kaggle/working/aasist"

try:
    sys.path.insert(0, AASIST_DIR)
    from models.AASIST import Model as AASISTModel

    conf = json.load(open(f"{AASIST_DIR}/config/AASIST.conf"))
    net = AASISTModel(conf["model_config"]).to(DEVICE)
    net.load_state_dict(torch.load(os.path.join(AASIST_DIR, conf["model_path"]),
                                   map_location=DEVICE), strict=False)
    net.eval()

    def pad_fixed(w):
        """Repeat to length rather than zero-pad, as the released evaluation does."""
        if len(w) >= NB_SAMP:
            return w[:NB_SAMP]
        return np.tile(w, int(NB_SAMP / len(w)) + 1)[:NB_SAMP]

    def bonafide(out):
        """The released code scores with the bonafide logit, so higher means more genuine."""
        if isinstance(out, (tuple, list)):
            out = out[-1]
        return out[:, 1]

    @torch.no_grad()
    def score_aasist(paths, peak, batch=16):
        out = np.empty(len(paths), dtype=np.float32)
        for i in range(0, len(paths), batch):
            ws = []
            for p in paths[i: i + batch]:
                w, _ = sf.read(p, dtype="float32", always_2d=False)
                if w.ndim > 1:
                    w = w.mean(1)
                w = pad_fixed(w)
                if peak:
                    w = w / (float(np.abs(w).max()) or 1.0)
                ws.append(w)
            t = torch.from_numpy(np.stack(ws).astype(np.float32)).to(DEVICE)
            # Negated so that above zero means fake, matching every other score here.
            out[i: i + len(ws)] = -bonafide(net(t)).float().cpu().numpy()
        return out

    for variant, peak in (("released", False), ("peaknorm", True)):
        thr, sep = threshold_of(score_aasist(DEV.file.tolist(), peak), DEV.y.values)
        print(f"aasist {variant}: dev {sep}, threshold {thr:+.3f}")
        PARTS.append(emit("aasist", variant, score_aasist(GRID.file.tolist(), peak), thr))
except Exception as e:
    print(f"aasist skipped: {type(e).__name__}: {e}")


# CELL 7: the clean channel has to reproduce the published numbers

ALL = pd.concat(PARTS, ignore_index=True)
ALL.to_csv(f"{OUT}/grid_scores_all.csv", index=False)

clean = ALL[ALL.channel == "clean"]
print("\nclean channel, genuine speech rejected, against the published run")
for (arch, variant), g in clean.groupby(["arch", "variant"]):
    real = g[g.y == 0]
    ind = real[real.condition_grid == "in-domain control"]
    out = real[real.condition_grid != "in-domain control"]
    print(f"  {arch:12s} {variant:9s} in-domain {100 * (ind.score > 0).mean():6.2f}%  "
          f"out of corpus {100 * (out.score > 0).mean():6.2f}%")
print("\nxlsr_frozen baseline out of corpus should land near 33.5% and fixed near 0.2%.")
print("Those are the published per-clip scores reweighted the way this grid samples, equal")
print("weight per speaker and per corpus. The headline 29.2% and 0.11% are the same scores")
print("weighted by clip count, where the larger and easier corpus dominates.")
print("A large gap means the scoring path differs from the run those numbers came from.")


# CELL 8: the table

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


rows = []
for (arch, variant, cond, chan), g in ALL.groupby(
        ["arch", "variant", "condition_grid", "channel"]):
    real, fake = g[g.y == 0], g[g.y == 1]
    frr_k, far_k = int((real.score > 0).sum()), int((fake.score <= 0).sum())
    lo, hi = wilson(frr_k, len(real))
    rows.append(dict(arch=arch, variant=variant, condition=cond, channel=chan,
                     n=len(g), n_genuine=len(real), n_fake=len(fake),
                     frr=100 * frr_k / len(real) if len(real) else float("nan"),
                     frr_lo=lo, frr_hi=hi,
                     far=100 * far_k / len(fake) if len(fake) else float("nan")))

T = pd.DataFrame(rows).sort_values(["arch", "variant", "condition", "channel"])
T.to_csv(f"{OUT}/channel_grid.csv", index=False)
print(f"\nwrote channel_grid.csv, {len(T)} rows")
print(T.pivot_table(index=["arch", "variant", "condition"], columns="channel",
                    values="frr").round(2).to_string())
