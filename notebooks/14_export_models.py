"""Export the two detectors as artifacts the API can serve, and measure what the export cost.

Everything so far was measured on cached fp16 features from a GPU pass over 16 kHz mono audio.
The deployed service gets a browser upload, resamples it, runs an int8 ONNX graph on one CPU
thread, and dots the result with a stored weight vector. Four things change along that path and
each can move a score across the boundary, so this reproduces the path exactly.

Two heads are exported: the baseline trained on one corpus, which is the exhibit, and the fixed
head with a few speakers per code-switched corpus added, which is the product. Both hold those
speakers out of every number printed here.

Requirements
    Kaggle notebook, CPU. Internet on for the checkpoint and the onnx install in cell 1.
    Attached inputs: codeswitch-deepfake-feats, -real, -fakes. Cells 5 and 6 dominate runtime."""

# CELL 1: configuration and mounts
import os
import subprocess
import sys
import glob
import json
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

# The Kaggle image carries torch and transformers but not the onnx side of this.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime", "onnx"], check=True)

CHECKPOINT = "facebook/wav2vec2-xls-r-300m"
LAYER = 20
SR = 16000
SEED = 1337

# Target-corpus speakers per corpus entering the fixed head's training.
N_FIX = 8

# Clips pushed through the full deployment path in cell 5.
N_CHECK = 240

OPSET = 17

# The fp32 graph is about a gigabyte and only exists to be quantised.
KEEP_FP32 = False

# Inference splits a long clip into spans of this length. Declared here rather than beside the
# cell that tests it, so the artifacts can be rebuilt without re-running the slow checks.
WINDOW_S = 5.0
HOP_S = 4.0

OUT = "/kaggle/working/models"
FP32_ONNX = f"{OUT}/frontend_fp32.onnx"
INT8_ONNX = f"{OUT}/frontend_int8.onnx"
HEADS_JSON = f"{OUT}/heads.json"

os.makedirs(OUT, exist_ok=True)


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

if FEAT_DIR is None:
    sys.exit("attach codeswitch-deepfake-feats")
if FAKE_DIR is None or not os.path.isdir(REAL_ROOT):
    sys.exit("attach codeswitch-deepfake-real and codeswitch-deepfake-fakes")

IDX = pd.read_csv(f"{FEAT_DIR}/index.csv")
FEATS = np.load(f"{FEAT_DIR}/pooled_meanstd_fp16.npy", mmap_mode="r")
X = np.asarray(FEATS[:, LAYER], dtype=np.float32)
Y = (IDX.label == "fake").values.astype(int)

# The index stores the paths the extraction notebook saw. Those mount points carry a different
# slug in this notebook, so only the filename survives.
IDX["file"] = [
    f"{FAKE_DIR}/{os.path.basename(p)}" if lab == "fake"
    else f"{REAL_ROOT}/audio/{os.path.basename(p)}"
    for p, lab in zip(IDX.file, IDX.label)
]

# A wrong path here would only surface in cell 5, after the export has already been written.
missing = [p for p in IDX.file.sample(200, random_state=SEED) if not os.path.exists(p)]
if missing:
    sys.exit(f"{len(missing)} of 200 sampled paths do not resolve, first {missing[0]}")

print(f"{len(IDX):,} rows, layer {LAYER}, feature dim {X.shape[1]}")


# CELL 2: train the two heads on the cache
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

BASE_TRAIN = ((IDX.role == "train_source") & (IDX.split == "train")).values
DEV = ((IDX.role == "train_source") & (IDX.split == "dev")).values
CONTROL = (IDX.role == "control_indomain").values
CORPORA = ("hindi_english", "bengali_english")

SPK = {c: sorted(IDX[IDX.set == c].speaker.dropna().unique()) for c in CORPORA}

rng = np.random.RandomState(SEED)
ADDED = {c: sorted(rng.choice(SPK[c], N_FIX, replace=False)) for c in CORPORA}

mask_added = np.zeros(len(IDX), dtype=bool)
for c, spks in ADDED.items():
    mask_added |= ((IDX.set == c) & IDX.speaker.isin(spks)).values

TRAIN = {"baseline": BASE_TRAIN, "fixed": BASE_TRAIN | mask_added}
# Held out by speaker. Nobody the fixed head trained on appears in any number below, which is
# what makes the two columns comparable at all.
HELD = IDX.set.isin(list(CORPORA)).values & ~mask_added


def eer_of(score, mask):
    y = Y[mask]
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score[mask])
    fnr = 1 - tpr
    return 100 * float((fpr + fnr)[int(np.nanargmin(np.abs(fnr - fpr)))] / 2)


def rejected(score, mask):
    """Genuine speech called fake, at the shipped threshold of zero."""
    v = score[mask & (Y == 0)]
    return 100 * float((v > 0).mean()) if len(v) else float("nan")


HEADS = {}
for name, train in TRAIN.items():
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    clf = LogisticRegression(max_iter=1000).fit((X[train] - mu) / sd, Y[train])
    raw = clf.decision_function((X - mu) / sd)

    # The threshold comes from in-domain dev, because a deployment has no labelled target data.
    # It is folded into the bias so the service can keep its rule as a plain score above a half,
    # rather than carrying a second number that could drift away from these weights.
    d = raw[DEV]
    lo, hi = d[Y[DEV] == 1].min(), d[Y[DEV] == 0].max()
    if lo > hi:
        thr = float((lo + hi) / 2)
        sep = "separable"
    else:
        # Not separable, so the midpoint is meaningless and the equal-error point is used.
        fpr, tpr, cut = roc_curve(Y[DEV], d)
        thr = float(cut[int(np.nanargmin(np.abs((1 - tpr) - fpr)))])
        sep = "overlapping"

    score = raw - thr
    HEADS[name] = dict(mu=mu, sd=sd, w=clf.coef_[0].astype(np.float64),
                       b=float(clf.intercept_[0] - thr), score=score, train=train)

    print(f"\n{name}: n_train {int(train.sum()):,}, dev {sep}, threshold {thr:+.3f}")
    print(f"dev EER {eer_of(score, DEV):.2f}%")
    print(f"in-domain control rejects {rejected(score, CONTROL):.1f}% of genuine speech")
    print(f"held-out corpora reject {rejected(score, HELD):.1f}%, "
          f"EER {eer_of(score, HELD):.2f}%")
    for c in CORPORA:
        m = (IDX.set == c).values & HELD
        print(f"{c} rejects {rejected(score, m):.1f}%")

print(f"\nfixed head trained on {N_FIX} added speakers per corpus: "
      + ", ".join(f"{c} {[int(s) for s in ADDED[c]]}" for c in CORPORA))


# CELL 3: truncate the frontend to layer 20 and export it
from transformers import AutoModel

model = AutoModel.from_pretrained(CHECKPOINT, attn_implementation="eager").eval()
N_FULL = len(model.encoder.layers)

# A reference taken before surgery, from the same object, so the comparison cannot drift on a
# second download or a different weight init path.
probe_rows = np.random.RandomState(SEED).choice(len(IDX), 2, replace=False)
probe_wavs = [sf.read(IDX.file.values[i], dtype="float32")[0][: SR * 4] for i in probe_rows]
probe_x = [torch.from_numpy((w - w.mean()) / np.sqrt(w.var() + 1e-7))[None, :] for w in probe_wavs]

with torch.no_grad():
    REF = [model(x, output_hidden_states=True).hidden_states[LAYER] for x in probe_x]

assert model.config.do_stable_layer_norm, "the trailing norm rule below assumes stable layernorm"
model.encoder.layers = torch.nn.ModuleList(list(model.encoder.layers[:LAYER]))
if LAYER < N_FULL:
    # In this architecture the encoder's final layer_norm is applied after the last block and
    # belongs to hidden state 24 alone. Hidden state 20 is a raw block output, so keeping the
    # norm here would export a different tensor than the one the heads were trained on.
    model.encoder.layer_norm = torch.nn.Identity()
model.config.num_hidden_layers = LAYER

with torch.no_grad():
    for x, ref in zip(probe_x, REF):
        got = model(x).last_hidden_state
        rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        print(f"truncated vs hidden_states[{LAYER}]: max relative difference {rel:.2e}")
        assert rel < 1e-4, "the truncated model is not returning the layer the heads expect"

kept = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{kept:.0f}M parameters kept of {N_FULL} layers truncated to {LAYER}")


class Frontend(torch.nn.Module):
    """A plain tensor in, a plain tensor out. The exporter handles dataclass outputs poorly."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_values):
        return self.base(input_values).last_hidden_state


wrapper = Frontend(model).eval()
dummy = torch.randn(1, SR * 4)

kwargs = dict(
    input_names=["input_values"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_values": {1: "samples"}, "last_hidden_state": {1: "frames"}},
    opset_version=OPSET,
    do_constant_folding=True,
)
try:
    torch.onnx.export(wrapper, (dummy,), FP32_ONNX, dynamo=False, **kwargs)
except TypeError:
    # Torch below 2.6 has no dynamo switch on this entry point.
    torch.onnx.export(wrapper, (dummy,), FP32_ONNX, **kwargs)

print(f"fp32 graph {os.path.getsize(FP32_ONNX)/2**20:.0f} MiB")


# CELL 4: quantise to int8 and price the damage
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


# Only MatMul. Dynamic quantisation of the convolutional feature extractor costs accuracy on
# the waveform side without buying much size, since the weight mass is in the transformer.
quantize_dynamic(FP32_ONNX, INT8_ONNX, weight_type=QuantType.QInt8,
                 op_types_to_quantize=["MatMul"],
                 extra_options={"MatMulConstBOnly": True})
print(f"int8 graph {os.path.getsize(INT8_ONNX)/2**20:.0f} MiB")

sess32, sess8 = session(FP32_ONNX, 4), session(INT8_ONNX, 4)


def pool(sess, wav):
    """The service's feature path, reproduced exactly."""
    x = wav.astype(np.float32)
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    h = sess.run(None, {"input_values": x[None, :]})[0][0]
    return np.concatenate([h.mean(0), h.std(0)])


for ref, wav in zip(REF, probe_wavs):
    # Torch pools with an unbiased standard deviation by default and numpy does not, so the
    # reference has to be told which one the service computes.
    t = np.concatenate([ref[0].mean(0).numpy(), ref[0].std(0, unbiased=False).numpy()])
    a, b = pool(sess32, wav), pool(sess8, wav)
    print(f"torch vs fp32 onnx: {np.abs(t-a).max() / (np.abs(t).max() + 1e-9):.2e} "
          f"fp32 vs int8 onnx: {np.abs(a-b).max() / (np.abs(a).max() + 1e-9):.2e}")

if not KEEP_FP32:
    del sess32
    os.remove(FP32_ONNX)


# CELL 5: the whole deployment path, on real audio
# The only number that describes what a real user will see. It runs the int8 graph over files
# the fixed head never trained on and scores them with the exported weights, so drift between
# the cache and the runtime shows up as a changed verdict rather than a small relative error.
def head_score(name, f):
    h = HEADS[name]
    return float(((f - h["mu"]) / h["sd"] * h["w"]).sum() + h["b"])


pools = [("in-domain genuine", CONTROL & (Y == 0)),
         ("in-domain fake", DEV & (Y == 1)),
         ("held-out genuine, Hindi-English", HELD & (Y == 0) & (IDX.set == "hindi_english").values),
         ("held-out genuine, Bengali-English", HELD & (Y == 0) & (IDX.set == "bengali_english").values)]

rng = np.random.RandomState(SEED)
rows = []
t0 = time.time()
for label, mask in pools:
    pick = rng.choice(np.flatnonzero(mask), min(N_CHECK // len(pools), int(mask.sum())),
                      replace=False)
    for i in pick:
        wav = sf.read(IDX.file.values[i], dtype="float32")[0]
        f = pool(sess8, wav)
        rows.append(dict(pool=label, row=int(i), y=int(Y[i]),
                         cached_baseline=float(HEADS["baseline"]["score"][i]),
                         cached_fixed=float(HEADS["fixed"]["score"][i]),
                         runtime_baseline=head_score("baseline", f),
                         runtime_fixed=head_score("fixed", f)))
    print(f"{label}: {len(pick)} clips, {time.time()-t0:.0f}s elapsed")

RUN = pd.DataFrame(rows)
RUN.to_csv("/kaggle/working/export_runtime_check.csv", index=False)

print("\npercentages are baseline / fixed, cached first then through the int8 graph\n")
for label, _ in pools:
    g = RUN[RUN.pool == label]
    if not len(g):
        continue
    if g.y.iloc[0] == 1:
        # Fakes are scored by miss rate, the opposite error.
        cb, cf = (g.cached_baseline <= 0).mean(), (g.cached_fixed <= 0).mean()
        rb, rf = (g.runtime_baseline <= 0).mean(), (g.runtime_fixed <= 0).mean()
        kind = "missed"
    else:
        cb, cf = (g.cached_baseline > 0).mean(), (g.cached_fixed > 0).mean()
        rb, rf = (g.runtime_baseline > 0).mean(), (g.runtime_fixed > 0).mean()
        kind = "rejected"
    print(f"{label}, n={len(g)}, % {kind}: "
          f"cached {100*cb:.1f} / {100*cf:.1f}, runtime {100*rb:.1f} / {100*rf:.1f}")

flips = ((RUN.cached_fixed > 0) != (RUN.runtime_fixed > 0)).mean()
corr = np.corrcoef(RUN.cached_fixed, RUN.runtime_fixed)[0, 1]
print(f"\nfixed head: {100*flips:.1f}% of clips change verdict between the cache and the "
      f"runtime, score correlation {corr:.4f}")
assert flips < 0.05, "quantisation is moving verdicts, export fp32 or quantise fewer ops"

# Latency at the thread count the container actually has, which is not the one used above.
solo = session(INT8_ONNX, 1)
for secs in (3, 10, 30):
    clip = np.random.RandomState(0).randn(SR * secs).astype(np.float32) * 0.05
    t = time.time()
    pool(solo, clip)
    print(f"{secs}s clip on one thread: {1000*(time.time()-t):.0f} ms")


# CELL 6: does windowing survive a long clip
# A live upload will be half a minute, and pooling mean and standard deviation over thirty
# seconds produces a different statistic rather than a longer one, so the service cuts the clip
# into training-length spans. Utterances with known verdicts are glued up and rescored.
def windows(audio):
    """Mirror of the function in app/api/service.py.

    Duplicated rather than imported because this file is pasted into Kaggle, where the
    repository does not exist. If one changes, change both.
    """
    n, hop = int(WINDOW_S * SR), int(HOP_S * SR)
    if len(audio) <= n:
        return [audio]
    starts = list(range(0, len(audio) - n + 1, hop))
    if starts[-1] + n < len(audio):
        starts.append(len(audio) - n)
    spans = [audio[s: s + n] for s in starts]
    ref = float(np.sqrt(np.mean(audio ** 2)))
    loud = [s for s in spans if float(np.sqrt(np.mean(s ** 2))) >= 0.1 * ref]
    return loud or spans


def windowed(name, audio):
    return float(np.mean([head_score(name, pool(sess8, w)) for w in windows(audio)]))


def stitch(mask, n_clips=6, per_clip=6):
    """Glue consecutive utterances from one speaker into a clip of demo length."""
    idx = np.flatnonzero(mask)
    spk = IDX.speaker.values[idx]
    out = []
    for s in pd.unique(spk):
        members = idx[spk == s][:per_clip]
        if len(members) < 3:
            continue
        wav = np.concatenate([sf.read(IDX.file.values[r], dtype="float32")[0] for r in members])
        out.append((s, members, wav))
        if len(out) == n_clips:
            break
    return out


cases = [("genuine, held out", HELD & (Y == 0) & (IDX.set == "hindi_english").values),
         ("fake, in domain", DEV & (Y == 1))]

rows = []
for label, mask in cases:
    for spk, members, wav in stitch(mask):
        rows.append(dict(
            pool=label, speaker=spk, n_utts=len(members),
            secs=round(len(wav) / SR, 1), n_win=len(windows(wav)),
            # The verdict these utterances already have, one at a time, from the cache.
            cached=float(HEADS["fixed"]["score"][members].mean()),
            whole=head_score("fixed", pool(sess8, wav)),
            windowed=windowed("fixed", wav)))

STITCH = pd.DataFrame(rows)
STITCH.to_csv("/kaggle/working/export_window_check.csv", index=False)
print(STITCH.round(3).to_string(index=False))

print()
for label, _ in cases:
    g = STITCH[STITCH.pool == label]
    agree_win = ((g.windowed > 0) == (g.cached > 0)).mean()
    agree_whole = ((g.whole > 0) == (g.cached > 0)).mean()
    print(f"{label}: agrees with the per-utterance verdict "
          f"{100*agree_win:.1f}% windowed against {100*agree_whole:.1f}% whole clip, "
          f"drift {abs(g.windowed-g.cached).mean():.2f} against "
          f"{abs(g.whole-g.cached).mean():.2f}")


# CELL 7: write the artifacts
spec = {
    "meta": {
        "checkpoint": CHECKPOINT,
        "layer": LAYER,
        "feature": "mean and std over frames of hidden state 20, concatenated",
        "dim": int(X.shape[1]),
        "sample_rate": SR,
        "bottleneck_hz": 8000,
        "threshold": "folded into the bias, so a probability above 0.5 means fake",
        "window_s": WINDOW_S,
        "hop_s": HOP_S,
        "n_added_speakers_per_corpus": N_FIX,
        "quantisation": f"dynamic int8, MatMul only, opset {OPSET}",
    },
    "heads": {
        name: {"mu": h["mu"].tolist(), "sd": h["sd"].tolist(),
               "w": h["w"].tolist(), "b": h["b"]}
        for name, h in HEADS.items()
    },
}
with open(HEADS_JSON, "w") as fh:
    json.dump(spec, fh)
print(f"heads.json {os.path.getsize(HEADS_JSON)/2**20:.1f} MiB")

card = """# codeswitch-deepfake-models

Artifacts served by `app/api/service.py`.

* `frontend_int8.onnx`, `{CHECKPOINT}` truncated to {LAYER} transformer blocks, dynamically
  quantised to int8 over MatMul. One input, `input_values`, shape (1, samples) at {SR} Hz,
  already normalised to zero mean and unit variance. One output, `last_hidden_state`, which is
  hidden state {LAYER} of the untruncated model.
* `heads.json`. Two logistic heads over the mean and standard deviation of that output,
  {X.shape[1]} dimensions. Each carries `mu`, `sd`, `w`, `b`. The decision threshold is folded
  into `b`, so the service's rule of "probability above a half means fake" is the calibrated
  rule and not an accident of where logistic regression puts its own zero.

## Length

Scores are pooled over frames, and the training utterances average under five seconds, so a
long upload is cut into {WINDOW_S:.0f} second spans at a {HOP_S:.0f} second hop and the spans
are averaged as logits. Spans quieter than a tenth of the clip's own level are dropped, since
each span is normalised on its own and silence would otherwise be amplified into noise.

## Preprocessing that is not optional

Every training file was resampled native to 8 kHz to 16 kHz, so no energy above 4 kHz has ever
been seen by these weights. An upload arriving at 44.1 kHz must pass the same bottleneck. This
is the exact axis the project measured, and skipping it puts the input out of distribution in
the way the paper is about.

## What the two heads are

`baseline` is trained on one corpus. It reaches 0% error in domain and then rejects a large
fraction of genuine speech recorded anywhere else. It is the exhibit, not the product.

`fixed` adds {N_FIX} speakers per code-switched corpus to the same training set. Those speakers
are held out of every number reported, so the figures describe these weights rather than a
sibling.

## Known limits

Neither head has heard a phone microphone. Both corpora used for adaptation are read speech
from studio-grade collections, so a live recording is a third recording condition and not a
familiar one. The acceptance test on real phone audio is a separate step and its result is not
predicted by anything here.
"""
with open(f"{OUT}/MODEL_CARD.md", "w", encoding="utf-8") as fh:
    fh.write(card)

print("""
Upload, from a machine with the Kaggle output downloaded and modal installed. The volume has
to exist before anything can be written into it; the service declares it with
create_if_missing, but that only runs at deploy time, which is after the upload.

    modal volume create codeswitch-deepfake-models
    modal volume put codeswitch-deepfake-models frontend_int8.onnx /frontend_int8.onnx
    modal volume put codeswitch-deepfake-models heads.json /heads.json
    modal deploy app/api/service.py

The service reports mode "onnx" once both files are present, and "stub" until then. Files added
to the volume after a deploy are picked up by the next container rather than the running one.
""")
