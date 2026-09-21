"""Cache frozen XLS-R features for every real and fake utterance.

Every experiment in weeks 3 to 5 reads this cache instead of running the frontend, so the cost
is paid once. The frontend is frozen throughout, which is what makes a cache valid at all.

Frame-level features are not storable here: 31,922 utterances at 50 fps and 1024 dimensions is
roughly 16 GB in fp16, past the working disk. What is stored instead is the mean and standard
deviation over time of every one of the 25 hidden states. Pooling loses temporal detail, but
keeping all layers keeps the one choice that matters open, spoofing artifacts in a
self-supervised model are not concentrated in the final layer, and a sweep over layers on the
dev split costs nothing once this file exists, whereas testing one more layer by re-extracting
costs the entire pass again.

The cache serves the baseline, the go/no-go evaluation and the condition grids. Adapting
the frontend itself reads audio rather than this file and is unaffected by the pooling.

Requirements
    Kaggle notebook, GPU T4. Internet: On for the checkpoint (~1.2 GB).
    Attached inputs: codeswitch-deepfake-real and codeswitch-deepfake-fakes.
    Output is about 3 GB, so commit rather than running interactively.
    To resume a killed run, attach the previous commit's output as a third input; the `feats`
    directory is found the same way the fakes directory is.
"""

# CELL 1: configuration and mounts
import os
import sys
import glob
import json
import time
import shutil

import numpy as np
import pandas as pd
import soundfile as sf
import torch

CHECKPOINT = "facebook/wav2vec2-xls-r-300m"

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
WORK = "/kaggle/working/feats"
FEAT_NPY = f"{WORK}/pooled_meanstd_fp16.npy"
DONE_NPY = f"{WORK}/done.npy"
INDEX_CSV = f"{WORK}/index.csv"
META_JSON = f"{WORK}/meta.json"

SR = 16000
# padded audio per batch, so long clips batch less deeply
MAX_BATCH_SAMPLES = SR * 60
MAX_BATCH = 16
FLUSH_EVERY = 1000
# set to e.g. 200 for a smoke run; the verification cell relaxes to match
LIMIT = None

os.makedirs(WORK, exist_ok=True)

# The fakes mount path depends on the notebook slug rather than the dataset name, so it is
# found by looking for the directory the generator wrote. Same rule as 07.
FAKE_DIR = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/fakes")) if os.path.isdir(p)),
    None)
RESUME_ROOT = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/feats")) if os.path.isdir(p)),
    None)

if not os.path.isdir(REAL_ROOT):
    sys.exit(f"real dataset not mounted at {REAL_ROOT}, use + Add Input")
if FAKE_DIR is None:
    sys.exit("no `fakes` directory among the mounted inputs, attach codeswitch-deepfake-fakes")

FAKE_ROOT = os.path.dirname(FAKE_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {DEVICE}")
print(f"fakes at {FAKE_DIR}")
print(f"resume source {RESUME_ROOT}")


# CELL 2: one index over both halves
real = pd.read_csv(f"{REAL_ROOT}/manifest_real.csv")
fake = pd.read_csv(f"{FAKE_ROOT}/manifest_fake.csv")

real["file"] = REAL_ROOT + "/" + real.path
real["duration"] = real.dur
real["attack"] = "none"
real["voiced_by"] = real.speaker
real["pair"] = real.fname

# The fake manifest inherits `path` and `dur` from the real row it was generated from, so
# neither describes the fake file. Its own audio is `fake_fname` and its own length `dur_fake`.
fake["file"] = FAKE_DIR + "/" + fake.fake_fname
fake["duration"] = fake.dur_fake
fake["pair"] = fake.fname

KEEP = ["file", "pair", "label", "attack", "set", "lang", "split", "role", "speaker",
        "voiced_by", "condition", "cs_bin", "eng_ratio", "cmi", "switches", "n_words",
        "duration"]
IDX = pd.concat([real[KEEP], fake[KEEP]], ignore_index=True)

# A fake whose real counterpart is missing would sit in a condition unpaired, which is the one
# thing the balance of this dataset depends on.
assert set(fake.pair) <= set(real.fname), "a fake references a real utterance not in the manifest"
assert IDX.file.is_unique

# Listing each directory once rather than stat-ing 31,922 paths: a file missing here costs the
# whole pass, and the check is worth seconds rather than an hour.
have = {f"{REAL_ROOT}/audio/{f}" for f in os.listdir(f"{REAL_ROOT}/audio")}
have |= {f"{FAKE_DIR}/{f}" for f in os.listdir(FAKE_DIR) if f.endswith(".flac")}
missing = IDX[~IDX.file.isin(have)]
if len(missing):
    print(missing.head(10).file.to_string())
    sys.exit(f"{len(missing)} indexed files are not on disk")

N = len(IDX)
print(f"{N:,} utterances, {IDX.duration.sum()/3600:.1f} h")
print(IDX.groupby(["set", "split", "label"]).size().rename("utts").to_string())


# CELL 3: the frozen frontend
from transformers import AutoFeatureExtractor, AutoModel

fe = AutoFeatureExtractor.from_pretrained(CHECKPOINT)
model = AutoModel.from_pretrained(CHECKPOINT).to(DEVICE).eval()

# hidden_states includes the conv output
N_LAYERS = model.config.num_hidden_layers + 1
HIDDEN = model.config.hidden_size
DIM = 2 * HIDDEN

assert fe.sampling_rate == SR, "frontend sample rate differs from the dataset"
# Normalisation is per utterance. Without a mask it would be computed across the padding too,
# which makes a file's features depend on what it was batched with.
assert fe.do_normalize and fe.return_attention_mask, "batched pooling needs masked normalisation"

print(f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
      f"{N_LAYERS} hidden states of {HIDDEN}")
print(f"cache will be {N * N_LAYERS * DIM * 2 / 2**30:.2f} GiB")


@torch.no_grad()
def pooled(paths):
    """Mean and std over valid frames, for every layer. Returns (n, N_LAYERS, DIM) fp16."""
    waves = [sf.read(p, dtype="float32")[0] for p in paths]
    enc = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True,
             return_attention_mask=True)
    iv = enc.input_values.to(DEVICE)
    # The extractor returns an int32 mask; the encoder builds tensors from its dtype and wants
    # a long.
    am = enc.attention_mask.to(DEVICE).long()

    with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        out = model(iv, attention_mask=am, output_hidden_states=True)

    # The conv stack strides the waveform down, so the number of real frames per item is not
    # the padded length. This is the model's own accounting of that stride.
    lens = model._get_feat_extract_output_lengths(am.sum(-1)).to(torch.long)
    frames = out.hidden_states[0].shape[1]
    mask = (torch.arange(frames, device=DEVICE)[None, :] < lens[:, None]).float()
    n = mask.sum(1).clamp(min=1)[:, None]

    stats = torch.empty(len(waves), N_LAYERS, DIM, dtype=torch.float32, device=DEVICE)
    for li, h in enumerate(out.hidden_states):
        # Accumulated in fp32: autocast leaves the activations in fp16, where summing a
        # thousand frames loses precision the stored value would then keep.
        h = h.float() * mask[:, :, None]
        mean = h.sum(1) / n
        var = (h.pow(2).sum(1) / n - mean.pow(2)).clamp(min=0)
        stats[:, li] = torch.cat([mean, var.sqrt()], dim=1)

    return stats.cpu().numpy().astype(np.float16)


# CELL 4: extract, resume-safe
from numpy.lib.format import open_memmap

FINGERPRINT = dict(n=N, layers=N_LAYERS, dim=DIM, checkpoint=CHECKPOINT,
                   first=IDX.file.iloc[0], last=IDX.file.iloc[-1])


def carry_forward():
    """Copy a previous commit's partial cache in, but only if it indexes the same files.

    Row position is the only key here, so a cache built against a different index would be
    silently mismatched rather than wrong in a way that shows up.
    """
    if not RESUME_ROOT:
        return
    prev = json.load(open(f"{RESUME_ROOT}/meta.json")) if \
        os.path.exists(f"{RESUME_ROOT}/meta.json") else None
    if prev != FINGERPRINT:
        print("resume source does not match this index, ignoring it and starting clean")
        return
    for f in ("pooled_meanstd_fp16.npy", "done.npy"):
        src = f"{RESUME_ROOT}/{f}"
        if os.path.exists(src) and not os.path.exists(f"{WORK}/{f}"):
            t0 = time.time()
            shutil.copyfile(src, f"{WORK}/{f}")
            print(f"carried {f} forward in {time.time()-t0:.0f}s")


carry_forward()
IDX.to_csv(INDEX_CSV, index=False)
json.dump(FINGERPRINT, open(META_JSON, "w"))

exists = os.path.exists(FEAT_NPY) and os.path.exists(DONE_NPY)
feats = open_memmap(FEAT_NPY, mode="r+" if exists else "w+",
                    dtype=np.float16, shape=(N, N_LAYERS, DIM))
done = np.load(DONE_NPY) if exists else np.zeros(N, dtype=bool)
print(f"{int(done.sum()):,}/{N:,} already cached")

# Sorted by duration so a batch is padded to nearly its own length. Unsorted, one 20 s clip
# drags fifteen 2 s clips up to its length and most of the compute is spent on silence.
order = np.argsort(IDX.duration.values, kind="stable")
todo = [i for i in order if not done[i]]
if LIMIT:
    # Spread across the duration range rather than taking the head, so a smoke run exercises
    # the deepest batches and the shallowest ones rather than only the shortest clips.
    todo = todo[:: max(1, len(todo) // LIMIT)][:LIMIT]
    print(f"LIMIT set, extracting {len(todo)} of {N:,}")

batches, cur = [], []
for i in todo:
    # Ascending order means the item being added is always the longest in the batch, so the
    # padded size is its length times the count.
    padded = (len(cur) + 1) * int(IDX.duration.values[i] * SR)
    if cur and (padded > MAX_BATCH_SAMPLES or len(cur) >= MAX_BATCH):
        batches.append(cur)
        cur = []
    cur.append(i)
if cur:
    batches.append(cur)
print(f"{len(todo):,} to extract in {len(batches):,} batches")

failed, seen, t0 = [], 0, time.time()
for b, rows in enumerate(batches, 1):
    paths = [IDX.file.values[i] for i in rows]
    try:
        stats = pooled(paths)
    except Exception as e:
        # One unreadable file should cost one file, not the fifteen batched beside it.
        print(f"batch {b} failed ({type(e).__name__}: {e}), retrying singly")
        stats = np.zeros((len(rows), N_LAYERS, DIM), dtype=np.float16)
        keep = np.zeros(len(rows), dtype=bool)
        for k, p in enumerate(paths):
            try:
                stats[k] = pooled([p])[0]
                keep[k] = True
            except Exception as e2:
                failed.append(dict(file=p, err=f"{type(e2).__name__}: {e2}"))
        for k, i in enumerate(rows):
            if keep[k]:
                feats[i] = stats[k]
                done[i] = True
        seen += len(rows)
        continue

    if not np.isfinite(stats).all():
        # Silently storing a NaN row would surface as an unexplainable training failure days
        # later, so it is caught where the cause is still visible.
        bad = [paths[k] for k in range(len(rows)) if not np.isfinite(stats[k]).all()]
        failed.extend(dict(file=p, err="non-finite features") for p in bad)
        for k, i in enumerate(rows):
            if np.isfinite(stats[k]).all():
                feats[i] = stats[k]
                done[i] = True
    else:
        for k, i in enumerate(rows):
            feats[i] = stats[k]
            done[i] = True

    seen += len(rows)
    if seen % FLUSH_EVERY < len(rows) or b == len(batches):
        feats.flush()
        np.save(DONE_NPY, done)
        el = time.time() - t0
        print(f"{seen:,}/{len(todo):,} {el/60:.1f} min elapsed, "
              f"{el/max(seen,1)*(len(todo)-seen)/60:.0f} min left")

feats.flush()
np.save(DONE_NPY, done)
print(f"\n{int(done.sum()):,}/{N:,} cached, {len(failed)} failed, "
      f"{(time.time()-t0)/3600:.2f} GPU-h")
if failed:
    pd.DataFrame(failed).to_csv(f"{WORK}/failures.csv", index=False)
    print(pd.DataFrame(failed).err.value_counts().head().to_string())


# CELL 5: verify before anything downstream trusts this
if not LIMIT:
    assert done.all(), f"{int((~done).sum())} utterances missing, rerun before publishing"

# Chunked: materialising all 1.6e9 values at once to test them costs several GB for no reason.
# Only cached rows are checked, so the same cell serves a smoke run and a full one.
bad_rows = []
for a in range(0, N, 2000):
    sl = slice(a, min(a + 2000, N))
    chunk = np.asarray(feats[sl], dtype=np.float32).reshape(sl.stop - sl.start, -1)
    ok = np.isfinite(chunk).all(1) & (np.abs(chunk).sum(1) > 0)
    bad_rows.extend((a + np.flatnonzero(~ok & done[sl])).tolist())
assert not bad_rows, f"{len(bad_rows)} rows are non-finite or all zero, first {bad_rows[:5]}"

# Batching changes padding, and padding is exactly what the mask is there to undo. If the two
# paths disagree, every pooled value depends on batch composition and the cache is worthless.
rng = np.random.RandomState(1337)
probe = rng.choice(np.flatnonzero(done), 16, replace=False)
single = np.stack([pooled([IDX.file.values[i]])[0] for i in probe]).astype(np.float32)
batched = np.asarray(feats)[probe].astype(np.float32)
rel = np.abs(single - batched).max() / (np.abs(single).max() + 1e-9)
print(f"batched vs single: max relative difference {rel:.2e}")
assert rel < 0.02, "pooling depends on batch composition, the attention mask is not being applied"

# A smoke test, not the layer-sweep result: one middle layer, no tuning. It answers
# whether the cache carries any spoofing signal at all, while the session is still open.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve


def eer(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return (fpr[i] + fnr[i]) / 2


LAYER = N_LAYERS // 2
tr = ((IDX.role == "train_source") & (IDX.split == "train")).values & done
dv = ((IDX.role == "train_source") & (IDX.split == "dev")).values & done
y = (IDX.label == "fake").values.astype(int)

if tr.sum() >= 200 and dv.sum() >= 100 and len(set(y[dv])) == 2:
    X = np.asarray(feats)[:, LAYER].astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((X[tr] - mu) / sd, y[tr])
    s = clf.decision_function((X[dv] - mu) / sd)
    print(f"layer {LAYER} smoke test on dev: acc {clf.score((X[dv]-mu)/sd, y[dv]):.3f}, "
          f"EER {100*eer(y[dv], s):.2f}%, n_train {int(tr.sum())}, n_dev {int(dv.sum())}")
else:
    print(f"not enough cached train/dev rows for the probe ({int(tr.sum())}/{int(dv.sum())})")


# CELL 6: dataset card
card = """# codeswitch-deepfake-feats

Frozen `{CHECKPOINT}` features for every utterance in `codeswitch-deepfake-real` and
`codeswitch-deepfake-fakes`.

## Contents

* `pooled_meanstd_fp16.npy`, ({N:,}, {N_LAYERS}, {DIM}) fp16. Axis 1 is the hidden state
  index: 0 is the projected convolutional output as it enters the transformer, and
  {N_LAYERS - 1} is the final transformer layer.
  Axis 2 is the mean over frames concatenated with the standard deviation over frames,
  {HIDDEN} dimensions each.
* `index.csv`, one row per utterance in the same order as axis 0. Carries label, attack,
  set, split, role, speaker, and the code-switching metrics.
* `meta.json`. The fingerprint a resuming run checks before reusing a partial cache.

## How it was pooled

Utterances are normalised individually and batched with an attention mask, so a file's
features do not depend on what it was batched with; the equivalence against unbatched
extraction is asserted in `09_extract_features.py`. Statistics are accumulated in fp32 over
valid frames only, using the model's own conv-stride accounting, then stored as fp16.

All 25 hidden states are kept because the layer carrying spoofing artifacts is not known in
advance and re-extracting to test one costs the whole pass again.

## What it does not support

Frame-level modelling, and anything that updates the frontend. Both read audio instead.
"""
with open(f"{WORK}/DATASET_CARD.md", "w", encoding="utf-8") as f:
    f.write(card)
print(card)
