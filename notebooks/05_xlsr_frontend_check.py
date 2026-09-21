"""Verify the XLS-R frontend loads and runs before training code depends on it.

Everything downstream assumes a wav2vec2-style self-supervised frontend is available on
Kaggle and produces frame-level features from the frozen real-audio dataset. This
confirms that end to end and records what the feature extractor needs: hidden size,
frame rate, parameter count.

The final cells also measure the deployed artifact size. Hosting stays free only if
the always-warm container fits in about 2 GiB, which the plan resolves by shipping
int8 ONNX rather than fp32 PyTorch. That is an arithmetic claim until something
weighs it, so this weighs it.

Requirements
    Kaggle notebook, CPU is sufficient (no GPU quota needed).
    Internet: On, for the checkpoint download (~1.2 GB).
    Attached input: the frozen real-audio dataset.
    Optional: an HF_TOKEN secret. The checkpoint is public, so this only lifts
    anonymous rate limits.
"""

# CELL 1: environment
import os
import sys
import time

CHECKPOINT = "facebook/wav2vec2-xls-r-300m"

KAGGLE_USER = "aryankumarpandey"
ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
WORK = "/kaggle/working/xlsr_check"
# the ceiling that keeps the always-warm container free
DEPLOY_BUDGET_GIB = 2.0

os.makedirs(WORK, exist_ok=True)

import importlib.metadata as md

for pkg in ("transformers", "torch", "numpy", "soundfile"):
    try:
        print(f"{pkg} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg} absent")

import torch
print(f"torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

if not os.path.isdir(ROOT):
    sys.exit(f"dataset not mounted at {ROOT}, use + Add Input")

# Optional; the checkpoint is public and downloads fine without it.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded")
except Exception:
    print("HF_TOKEN not set, continuing anonymously")


# CELL 2: load the checkpoint
from transformers import AutoFeatureExtractor, AutoModel

t0 = time.time()
fe = AutoFeatureExtractor.from_pretrained(CHECKPOINT)
model = AutoModel.from_pretrained(CHECKPOINT)
model.eval()
load_s = time.time() - t0

n_params = sum(p.numel() for p in model.parameters())
fp32_gib = n_params * 4 / 1024**3

print(f"loaded in {load_s:.0f}s")
print(f"parameters {n_params/1e6:.0f}M")
print(f"hidden size {model.config.hidden_size}, {model.config.num_hidden_layers} layers")
print(f"expects {fe.sampling_rate} Hz, normalise={fe.do_normalize}")
print(f"fp32 weights {fp32_gib:.2f} GiB")

# The dataset was frozen at 16 kHz specifically to match this.
assert fe.sampling_rate == 16000, "frontend sample rate differs from the dataset"


# CELL 3: forward pass on a real clip
import pandas as pd
import soundfile as sf

MAN = pd.read_csv(f"{ROOT}/manifest_real.csv")
row = MAN[MAN.dur.between(4.0, 6.0)].iloc[0]

audio, sr = sf.read(f"{ROOT}/{row.path}", dtype="float32")
inputs = fe(audio, sampling_rate=sr, return_tensors="pt")

t0 = time.time()
with torch.no_grad():
    out = model(**inputs).last_hidden_state
infer_s = time.time() - t0

frames, hidden = out.shape[1], out.shape[2]
print(f"clip {row.dur:.2f}s @ {sr} Hz, condition={row.condition}")
print(f"features {tuple(out.shape)} = (batch, frames, hidden)")
print(f"frame rate {frames/row.dur:.0f} fps, {1000*row.dur/frames:.1f} ms per frame")
print(f"cpu inference {infer_s:.1f}s, RTF {infer_s/row.dur:.2f}")

# wav2vec2 strides to 20 ms frames. A large deviation means the config isn't what the
# pooling and sequence-length assumptions downstream are written against.
assert 45 < frames / row.dur < 55, "unexpected frame rate for a wav2vec2 frontend"


# CELL 4: deployment size, measured rather than estimated
import onnx
import numpy as np
from onnxruntime.quantization import QuantType, quantize_dynamic

# Export is done on a fixed-length dummy input with a dynamic time axis, so one
# exported graph serves clips of any duration.
dummy = torch.zeros(1, 16000 * 5)
fp32_path = f"{WORK}/xlsr_fp32.onnx"

t0 = time.time()
# dynamo=False picks the TorchScript exporter. Recent torch defaults to the dynamo path,
# which needs onnxscript and wants dynamic_shapes instead of the dynamic_axes used here.
torch.onnx.export(
    model, (dummy,), fp32_path,
    input_names=["input_values"], output_names=["last_hidden_state"],
    dynamic_axes={"input_values": {0: "batch", 1: "samples"},
                  "last_hidden_state": {0: "batch", 1: "frames"}},
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print(f"onnx export {time.time()-t0:.0f}s")

int8_path = f"{WORK}/xlsr_int8.onnx"
t0 = time.time()
# per_channel=True was measured and made no difference here (cosine 0.9566 vs 0.9591) while
# taking 4x longer, so it stays off. If the drift ever needs reducing, the untried lever is
# restricting quantisation to MatMul and leaving the conv feature extractor in fp32.
quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
print(f"quantisation {time.time()-t0:.0f}s")


def size_gib(path):
    """ONNX may write weights beside the graph, so sum the whole directory."""
    d = os.path.dirname(path)
    stem = os.path.basename(path).replace(".onnx", "")
    return sum(os.path.getsize(os.path.join(d, f))
               for f in os.listdir(d) if f.startswith(stem)) / 1024**3


fp32_onnx = size_gib(fp32_path)
int8_onnx = size_gib(int8_path)

print(f"\nfp32 torch weights {fp32_gib:.2f} GiB")
print(f"fp32 onnx {fp32_onnx:.2f} GiB")
print(f"int8 onnx {int8_onnx:.2f} GiB")

# Runtime and activations sit on top of the weights, so the container needs headroom
# beyond the model file itself.
overhead_gib = 0.5
projected = int8_onnx + overhead_gib
verdict = "fits" if projected < DEPLOY_BUDGET_GIB else "OVER BUDGET"
print(f"\nint8 + ~{overhead_gib} GiB runtime and activations = {projected:.2f} GiB "
      f"against a {DEPLOY_BUDGET_GIB} GiB ceiling -> {verdict}")


# CELL 5: confirm the quantised graph still produces usable features
import onnxruntime as ort

sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
t0 = time.time()
onnx_out = sess.run(None, {"input_values": inputs["input_values"].numpy()})[0]
onnx_s = time.time() - t0

torch_out = out.numpy()
n = min(onnx_out.shape[1], torch_out.shape[1])
cos = float(np.mean([
    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    for a, b in zip(onnx_out[0, :n], torch_out[0, :n])
]))

print(f"onnx int8 output {onnx_out.shape}")
print(f"cpu inference {onnx_s:.1f}s, RTF {onnx_s/row.dur:.2f}")
print(f"cosine vs fp32 {cos:.4f}")

# This is a sanity check, not an acceptance test. Drift compounds across 24 layers, so a
# cosine in the mid-0.90s is ordinary for dynamic int8 and does not by itself rule out
# deploying the quantised graph.
# 
# It only becomes a problem if the classifier is trained on fp32 features and the int8 graph
# is substituted at deploy time, because the calibrated threshold then sits in the wrong
# place, which is exactly the failure mode where genuine speech gets called fake. Calibrate
# the threshold on whichever graph ships and the drift stops mattering.
# 
# The real acceptance test is downstream EER on held-out code-switched data, run in W7.
# fp32 ONNX also fits the memory ceiling, so staying unquantised is always available.
print(f"cosine {cos:.4f}, "
      + ("close to fp32" if cos > 0.99 else
         "moderate drift, fine if the threshold is calibrated on this graph" if cos > 0.90
         else "large drift, deploy fp32 ONNX instead"))
