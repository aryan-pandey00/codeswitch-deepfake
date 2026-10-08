"""Close the two gaps the feature cache's own checks left open.

Notebook 09 verified pooled features against unbatched extraction, which establishes row
alignment but not that the attention mask is applied: 09 batches in duration order, so a broken
mask has one or two percent padding to corrupt and the error would sit near fp16 noise.

This pads deliberately. A short utterance is extracted alone and again batched with the longest
file in the dataset; with a correct mask the two match to floating-point precision. The second
gap: the extractor validates the sample rate it is handed rather than resampling to it, so a
file at another rate would produce plausible, wrong features silently.

Requirements
    Kaggle notebook, CPU, internet on for the checkpoint.
    Attached inputs: codeswitch-deepfake-feats, and the real and fake audio datasets."""

# CELL 1: mounts
import os
import sys
import glob

import numpy as np
import pandas as pd
import soundfile as sf
import torch

CHECKPOINT = "facebook/wav2vec2-xls-r-300m"

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
SR = 16000

FEAT_DIR = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/feats")) if os.path.isdir(p)),
    None)
FAKE_DIR = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/fakes")) if os.path.isdir(p)),
    None)
if FEAT_DIR is None or FAKE_DIR is None:
    sys.exit("attach codeswitch-deepfake-feats and codeswitch-deepfake-fakes")

IDX = pd.read_csv(f"{FEAT_DIR}/index.csv")
FEATS = np.load(f"{FEAT_DIR}/pooled_meanstd_fp16.npy", mmap_mode="r")

base = IDX.file.map(os.path.basename)
IDX["file"] = np.where(IDX.label == "real", f"{REAL_ROOT}/audio/" + base, f"{FAKE_DIR}/" + base)
print(f"{len(IDX):,} rows, features {FEATS.shape}")


# CELL 2: every file is 16 kHz mono
from concurrent.futures import ThreadPoolExecutor


def header(path):
    i = sf.info(path)
    return i.samplerate, i.channels, i.duration


with ThreadPoolExecutor(max_workers=8) as ex:
    H = list(ex.map(header, IDX.file.values))

H = pd.DataFrame(H, columns=["sr", "ch", "dur_file"])
print(H.sr.value_counts().to_string())
print(H.ch.value_counts().to_string())
assert (H.sr == SR).all(), "a file is not 16 kHz, its features were extracted at the wrong rate"
assert (H.ch == 1).all(), "a file is not mono"

# The manifest duration is what batching and every per-condition summary is computed from. A
# drift between it and the file would not corrupt the features, but it would misreport the
# corpus in the paper.
drift = (H.dur_file.values - IDX.duration.values)
print(f"manifest vs file duration: max |drift| {np.abs(drift).max():.4f} s")
assert np.abs(drift).max() < 0.05


# CELL 3: the adversarial padding test
from transformers import AutoFeatureExtractor, AutoModel

fe = AutoFeatureExtractor.from_pretrained(CHECKPOINT)
model = AutoModel.from_pretrained(CHECKPOINT).eval()
N_LAYERS = model.config.num_hidden_layers + 1
HIDDEN = model.config.hidden_size


@torch.no_grad()
def pooled(paths):
    """The pooling from notebook 09, unchanged, on CPU in fp32."""
    waves = [sf.read(p, dtype="float32")[0] for p in paths]
    enc = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True,
             return_attention_mask=True)
    am = enc.attention_mask.long()
    out = model(enc.input_values, attention_mask=am, output_hidden_states=True)

    lens = model._get_feat_extract_output_lengths(am.sum(-1)).to(torch.long)
    frames = out.hidden_states[0].shape[1]
    mask = (torch.arange(frames)[None, :] < lens[:, None]).float()
    n = mask.sum(1).clamp(min=1)[:, None]

    stats = torch.empty(len(waves), N_LAYERS, 2 * HIDDEN, dtype=torch.float32)
    for li, h in enumerate(out.hidden_states):
        h = h.float() * mask[:, :, None]
        mean = h.sum(1) / n
        var = (h.pow(2).sum(1) / n - mean.pow(2)).clamp(min=0)
        stats[:, li] = torch.cat([mean, var.sqrt()], dim=1)
    return stats.numpy()


rng = np.random.RandomState(1337)
order = np.argsort(IDX.duration.values)
longest = int(order[-1])
shortest = [int(i) for i in rng.choice(order[:2000], 6, replace=False)]

print(f"padding file {IDX.duration.values[longest]:.1f}s against clips of "
      f"{[round(float(IDX.duration.values[i]), 1) for i in shortest]}s")

alone = np.stack([pooled([IDX.file.values[i]])[0] for i in shortest])
padded = pooled([IDX.file.values[i] for i in shortest] + [IDX.file.values[longest]])[:-1]

scale = np.abs(alone).max()
rel = np.abs(alone - padded).max() / (scale + 1e-9)
# With a batch this lopsided, dropping the mask would pull each short clip's pooled mean
# towards zero by the padded fraction, an error of the same order as the values themselves,
# not a rounding difference.
dilution = 1 - IDX.duration.values[shortest] / IDX.duration.values[longest]
print(f"padded fraction per clip: {np.round(dilution, 3)}")
print(f"alone vs heavily padded: max relative difference {rel:.2e}")
assert rel < 1e-3, "pooling changes when padding dominates, the attention mask is not working"


# CELL 4: the fresh CPU pass still agrees with what was stored
stored = np.asarray(FEATS[shortest], dtype=np.float32)
cross = np.abs(alone - stored).max() / (np.abs(alone).max() + 1e-9)
print(f"cpu fp32 vs stored gpu fp16: max relative difference {cross:.2e}")
# Looser by design: the cache was built under autocast fp16 on a T4 and stored as fp16, so
# agreement to a few parts in a thousand is the expected floor rather than a concern.
assert cross < 0.02, "the stored cache does not reproduce, investigate before using it"

print("\ncache verified: 16 kHz mono throughout, masking correct under heavy padding, "
      "stored values reproduce")
