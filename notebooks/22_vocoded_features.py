"""Cache frozen XLS-R features for the vocoded fakes, in the same space as the existing cache.

Notebook 21 wrote the resynthesised audio. Nothing can be trained on it until it has been
through the same frontend, with the same pooling, as the 31,922 clips already cached by
notebook 09. Any difference in checkpoint, sample rate, normalisation or masking would put
these rows in a different space, and a head trained across both would be fitting the
difference between two extractors rather than the difference between real and fake.

The recipe is therefore copied rather than rewritten: facebook/wav2vec2-xls-r-300m, per
utterance normalisation with an attention mask, mean and standard deviation over the valid
frames only, accumulated in fp32 and stored as fp16.

One thing does differ, deliberately. Notebook 09 stored all 25 hidden states because the layer
had not been chosen yet. Layer 20 has been settled since, and is locked, so only that layer is
stored here. The cache is 143 MB instead of 3.5 GB, which matters because this output has to
be attached alongside the existing one.

Requirements
    Kaggle notebook, GPU T4. Internet: On for the checkpoint (~1.2 GB).
    Attached inputs: the output of notebook 21, and codeswitch-deepfake-feats for the
    fingerprint check.
    Resumable. Re-attach the previous commit's output to continue a killed run.
    Writes voc_layer20_fp16.npy and voc_index.csv.
"""

# CELL 1: configuration and mounts
import os
import sys
import glob
import json
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

CHECKPOINT = "facebook/wav2vec2-xls-r-300m"
SR = 16000
LAYER = 20

# Matched to notebook 09, so batch composition cannot change a stored value.
MAX_BATCH_SAMPLES = SR * 60
MAX_BATCH = 16

# Set to a few hundred for a smoke run. The verification cell relaxes to match.
LIMIT = None

WORK = "/kaggle/working"
FEAT_NPY = f"{WORK}/voc_layer20_fp16.npy"
DONE_NPY = f"{WORK}/voc_done.npy"
INDEX_CSV = f"{WORK}/voc_index.csv"
META_JSON = f"{WORK}/voc_meta.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {DEVICE}")


def find(name):
    """Kaggle mounts inputs under the notebook slug, not the dataset name."""
    return next(
        (p for depth in ("*", "*/*", "*/*/*")
         for p in sorted(glob.glob(f"/kaggle/input/{depth}/{name}")) if os.path.isdir(p)),
        None)


VOC_DIR = find("vocoded")
if VOC_DIR is None:
    sys.exit("attach the output of notebook 21")

VOC_ROOT = os.path.dirname(VOC_DIR)
VOC = pd.read_csv(f"{VOC_ROOT}/manifest_vocoded.csv")
VOC["file"] = [f"{VOC_DIR}/{f}" for f in VOC.fname]

missing = [p for p in VOC.file if not os.path.exists(p)]
if missing:
    print(f"{len(missing)} manifest rows have no audio, dropping them")
    VOC = VOC[VOC.file.apply(os.path.exists)].reset_index(drop=True)

N = len(VOC)
print(f"{N:,} vocoded clips")
print(VOC.groupby(["vocoder", "role"]).size().to_string())


# CELL 2: the frozen frontend, identical to notebook 09
from transformers import AutoFeatureExtractor, AutoModel

fe = AutoFeatureExtractor.from_pretrained(CHECKPOINT)
model = AutoModel.from_pretrained(CHECKPOINT).to(DEVICE).eval()

N_LAYERS = model.config.num_hidden_layers + 1
HIDDEN = model.config.hidden_size
DIM = 2 * HIDDEN

assert fe.sampling_rate == SR, "frontend sample rate differs from the audio"
# Per utterance normalisation. Without a mask it would run across the padding too, and a
# file's features would then depend on what it was batched with.
assert fe.do_normalize and fe.return_attention_mask, "batched pooling needs masked normalisation"
assert LAYER < N_LAYERS, f"layer {LAYER} does not exist in a model with {N_LAYERS} states"

print(f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
      f"{N_LAYERS} hidden states of {HIDDEN}, storing layer {LAYER} only")
print(f"cache will be {N * DIM * 2 / 2**20:.0f} MiB")


@torch.no_grad()
def pooled(paths):
    """Mean and std over the valid frames of one layer. Returns (n, DIM) fp16."""
    waves = [sf.read(p, dtype="float32")[0] for p in paths]
    enc = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True,
             return_attention_mask=True)
    iv = enc.input_values.to(DEVICE)
    am = enc.attention_mask.to(DEVICE).long()

    with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        out = model(iv, attention_mask=am, output_hidden_states=True)

    lens = model._get_feat_extract_output_lengths(am.sum(-1)).to(torch.long)
    frames = out.hidden_states[0].shape[1]
    mask = (torch.arange(frames, device=DEVICE)[None, :] < lens[:, None]).float()
    n = mask.sum(1).clamp(min=1)[:, None]

    # fp32 for the accumulation. Autocast leaves activations in fp16, where summing a thousand
    # frames loses precision the stored value would otherwise keep.
    h = out.hidden_states[LAYER].float() * mask[:, :, None]
    mean = h.sum(1) / n
    var = (h.pow(2).sum(1) / n - mean.pow(2)).clamp(min=0)
    return torch.cat([mean, var.sqrt()], dim=1).cpu().numpy().astype(np.float16)


# CELL 3: extract, resume-safe
from numpy.lib.format import open_memmap

FINGERPRINT = dict(n=N, layer=LAYER, dim=DIM, checkpoint=CHECKPOINT,
                   first=VOC.fname.iloc[0], last=VOC.fname.iloc[-1])

RESUME_ROOT = find("voc_feats") or (VOC_ROOT if os.path.exists(f"{VOC_ROOT}/voc_done.npy") else None)
if RESUME_ROOT:
    prev = json.load(open(f"{RESUME_ROOT}/voc_meta.json")) \
        if os.path.exists(f"{RESUME_ROOT}/voc_meta.json") else None
    if prev == FINGERPRINT:
        import shutil
        for f in ("voc_layer20_fp16.npy", "voc_done.npy"):
            src = f"{RESUME_ROOT}/{f}"
            if os.path.exists(src) and not os.path.exists(f"{WORK}/{f}"):
                shutil.copyfile(src, f"{WORK}/{f}")
                print(f"carried {f} forward")
    else:
        print("resume source indexes a different set, starting clean")

VOC.to_csv(INDEX_CSV, index=False)
json.dump(FINGERPRINT, open(META_JSON, "w"))

exists = os.path.exists(FEAT_NPY) and os.path.exists(DONE_NPY)
feats = open_memmap(FEAT_NPY, mode="r+" if exists else "w+",
                    dtype=np.float16, shape=(N, DIM))
done = np.load(DONE_NPY) if exists else np.zeros(N, dtype=bool)
print(f"{int(done.sum()):,} already cached, {N - int(done.sum()):,} to go")

# Ascending duration, so the item being added to a batch is always its longest and the padded
# size is that length times the count.
order = np.argsort(VOC.dur.values, kind="stable")
todo = [i for i in order if not done[i]]
if LIMIT:
    todo = todo[:: max(1, len(todo) // LIMIT)][:LIMIT]
    print(f"LIMIT set, extracting {len(todo)} of {N:,}")

batches, cur = [], []
for i in todo:
    padded = (len(cur) + 1) * int(VOC.dur.values[i] * SR)
    if cur and (padded > MAX_BATCH_SAMPLES or len(cur) >= MAX_BATCH):
        batches.append(cur)
        cur = []
    cur.append(i)
if cur:
    batches.append(cur)
print(f"{len(todo):,} to extract in {len(batches):,} batches")

failed, seen, t0 = [], 0, time.time()
for b, rows in enumerate(batches, 1):
    paths = [VOC.file.values[i] for i in rows]
    try:
        stats = pooled(paths)
        keep = np.ones(len(rows), dtype=bool)
    except Exception as exc:
        # One unreadable file should cost one file, not the fifteen batched beside it.
        print(f"batch {b} failed ({type(exc).__name__}: {exc}), retrying singly")
        stats = np.zeros((len(rows), DIM), dtype=np.float16)
        keep = np.zeros(len(rows), dtype=bool)
        for k, p in enumerate(paths):
            try:
                stats[k] = pooled([p])[0]
                keep[k] = True
            except Exception as exc2:
                failed.append({"file": p, "err": f"{type(exc2).__name__}: {exc2}"})

    for k, i in enumerate(rows):
        if keep[k]:
            feats[i] = stats[k]
            done[i] = True
    seen += len(rows)

    if b % 50 == 0 or b == len(batches):
        np.save(DONE_NPY, done)
        el = time.time() - t0
        rate = seen / max(el, 1e-9)
        print(f"{seen:,}/{len(todo):,} {el/60:.1f} min elapsed, "
              f"eta {(len(todo)-seen)/max(rate,1e-9)/60:.0f} min", flush=True)

feats.flush()
np.save(DONE_NPY, done)
print(f"\n{int(done.sum()):,} of {N:,} cached in {(time.time()-t0)/60:.1f} min")
if failed:
    pd.DataFrame(failed).to_csv(f"{WORK}/voc_feat_failures.csv", index=False)
    print(f"{len(failed)} failures, see voc_feat_failures.csv")


# CELL 4: verify the new rows share a space with the old cache
# A separable-looking result would follow automatically from an extractor mismatch, so this
# runs before any of it reaches a classifier.
OLD = find("feats")
if OLD is None:
    print("codeswitch-deepfake-feats not attached, skipping the cross-check")
else:
    old_idx = pd.read_csv(f"{OLD}/index.csv")
    old_feats = np.load(f"{OLD}/pooled_meanstd_fp16.npy", mmap_mode="r")
    old_real = np.asarray(old_feats[(old_idx.label == "real").values][:400, LAYER],
                          dtype=np.float32)
    new = np.asarray(feats[done][:400], dtype=np.float32)

    print(f"old real rows: mean {old_real.mean():+.4f}, sd {old_real.std():.4f}")
    print(f"new vocoded  : mean {new.mean():+.4f}, sd {new.std():.4f}")
    print("\nThese are different classes so the numbers should not match, but an order of "
          "magnitude apart would mean the extractor differs rather than the audio.")

    ratio = new.std() / max(old_real.std(), 1e-9)
    assert 0.2 < ratio < 5.0, f"scale ratio {ratio:.2f} suggests a different feature space"
    print(f"scale ratio {ratio:.2f}, within range")

# A vocoded fake and its real source differ only by resynthesis, so their features should be
# close but not identical. Identical would mean a path is a no-op and is teaching nothing.
if OLD is not None:
    src_names = {os.path.basename(f): i for i, f in enumerate(old_idx.file)}
    pairs = []
    for i in np.where(done)[0][:2000]:
        j = src_names.get(os.path.basename(str(VOC.source_path.values[i])))
        if j is not None:
            pairs.append((i, j))
    if pairs:
        a = np.asarray(feats[[p[0] for p in pairs]], dtype=np.float32)
        b = np.asarray(old_feats[[p[1] for p in pairs], LAYER], dtype=np.float32)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        print(f"\n{len(pairs)} fake/source pairs, cosine similarity "
              f"median {np.median(cos):.4f}, min {cos.min():.4f}, max {cos.max():.4f}")
        print("Near 1.000 across the board would mean the resynthesis left no trace in the "
              "representation, and no head could separate them.")


# CELL 5: the README for the output dataset
with open(f"{WORK}/README.md", "w", encoding="utf-8") as fh:
    fh.write(f"""# codeswitch-deepfake-voc-feats

Frozen XLS-R layer {LAYER} features for the vocoded fakes written by notebook 21.

Extracted with `{CHECKPOINT}`, per utterance normalisation with an attention mask, mean and
standard deviation pooled over valid frames only, accumulated in fp32 and stored as fp16.
Identical to the recipe in `codeswitch-deepfake-feats`, so rows from the two can be stacked.

* `voc_layer20_fp16.npy`, ({N}, {DIM}) fp16, row order matches `voc_index.csv`
* `voc_index.csv`, one row per clip, carrying `vocoder`, `held_out`, `set`, `role`, `split`,
  `speaker` and `source_path`

Only layer {LAYER} is stored. The layer sweep is settled, and keeping all 25 would be 3.5 GB
rather than {N * DIM * 2 / 2**20:.0f} MB.

All rows are label=fake. The genuine counterpart of each one is the file named in
`source_path`, which lives in `codeswitch-deepfake-real`.
""")
print("README written")
