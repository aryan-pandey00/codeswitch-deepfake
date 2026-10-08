"""Score real phone recordings with the exported detectors, without the audio leaving the machine.

Every number so far comes from two studio-grade read-speech corpora. The deployed model gets a
phone, in a room, over a codec, from someone talking rather than reading. That is a recording
condition neither head has heard, and an unfamiliar recording condition is the whole finding of
this project, so this is the test most likely to disagree with everything reported so far.

It runs locally on purpose. The clips are other people's voices and a private Kaggle dataset is
still an upload to a third party. Nothing writes audio anywhere; the only output is a CSV.

Scoring is byte-for-byte the path in app/api/service.py. A disagreement between this file and
the deployed service would mean one of them is wrong.

Requirements
    Python 3.11 (onnxruntime has no 3.14 wheels).
    pip install onnxruntime soundfile soxr imageio-ffmpeg pandas numpy
    frontend_int8.onnx and heads.json from notebook 14, in MODEL_DIR.
    Clips in CLIP_DIR named speaker_language_source_index_style.ext, style read or talk.
"""

import io
import re
import sys
import json
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import onnxruntime as ort
import imageio_ffmpeg

MODEL_DIR = Path(r"E:\CodeSwitch-Deepfake\models")
CLIP_DIR = Path(r"E:\CodeSwitch-Deepfake\data\realvoice")
OUT_CSV = Path(r"E:\CodeSwitch-Deepfake\data\realvoice_scores.csv")

SR = 16000
BOTTLENECK_HZ = 8000
WINDOW_S = 5.0
HOP_S = 4.0
MAX_WINDOWS = 8
CORES = 4

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# The index is optional, because one clip per speaker-language-source combination is the
# normal case and numbering a set of ones adds nothing.
NAME = re.compile(r"^(spk\d+)_([a-z-]+)_([a-z-]+)(?:_(\d+))?(?:_(read|talk))?$", re.I)

# Everything this project measured is Indian speech. A speaker from outside that population is
# worth scoring and worth reporting, but not worth averaging into a rate that claims to be
# about Indian speech.
OUT_OF_SCOPE = {"english-foreign"}


def decode(path):
    """Mono float32 at the file's own sample rate.

    The rate is kept rather than resampled here because the bottleneck below has to start from
    whatever the recorder actually produced. Handing ffmpeg a target rate would hide the thing
    being controlled for.
    """
    out = subprocess.run(
        [FFMPEG, "-v", "quiet", "-i", str(path), "-", "wav", "-ac", "1", "pipe:1"],
        capture_output=True)
    if out.returncode != 0 or not out.stdout:
        raise ValueError("ffmpeg could not read this file")
    audio, sr = sf.read(io.BytesIO(out.stdout), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def bottleneck(x, sr):
    """The sample rate path every training file took."""
    return soxr.resample(soxr.resample(x, sr, BOTTLENECK_HZ), BOTTLENECK_HZ, SR)


def band_ratio(x, sr):
    """Energy above 4 kHz against the speech band below it, on the file as delivered.

    This is what separates a call recording from a phone recorder even when both arrive as
    m4a. Measured before the bottleneck, since the bottleneck erases it by design.
    """
    if sr <= 8000 or len(x) < 1024:
        return 0.0
    n = min(len(x), sr * 5)
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n))) ** 2
    freq = np.fft.rfftfreq(n, 1 / sr)
    hi = spec[freq > 4000].sum()
    lo = spec[(freq > 300) & (freq <= 4000)].sum()
    return float(hi / (lo + 1e-12))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def windows(audio):
    """Mirror of app/api/service.py. If one changes, change both."""
    n, hop = int(WINDOW_S * SR), int(HOP_S * SR)
    if len(audio) <= n:
        return [audio]

    starts = list(range(0, len(audio) - n + 1, hop))
    if starts[-1] + n < len(audio):
        starts.append(len(audio) - n)
    spans = [audio[s: s + n] for s in starts]

    ref = rms(audio)
    loud = [s for s in spans if rms(s) >= 0.1 * ref] or spans

    if len(loud) > MAX_WINDOWS:
        step = (len(loud) - 1) / (MAX_WINDOWS - 1)
        loud = [loud[int(round(i * step))] for i in range(MAX_WINDOWS)]
    return loud


class Detector:
    def __init__(self, model_dir):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = CORES
        self.session = ort.InferenceSession(str(model_dir / "frontend_int8.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        spec = json.loads((model_dir / "heads.json").read_text())
        self.heads = {name: {k: np.asarray(v, dtype=np.float32) if isinstance(v, list) else v
                             for k, v in head.items()}
                      for name, head in spec["heads"].items()}
        self.meta = spec.get("meta", {})

    def features(self, chunk):
        x = chunk.astype(np.float32)
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        h = self.session.run(None, {"input_values": x[None, :]})[0][0]
        return np.concatenate([h.mean(0), h.std(0)])

    def score(self, audio):
        """Returns the mean logit per head. Above zero means the detector called it fake."""
        feats = [self.features(w) for w in windows(audio)]
        out = {}
        for name, head in self.heads.items():
            z = [float(((f - head["mu"]) / head["sd"] * head["w"]).sum() + head["b"])
                 for f in feats]
            out[name] = float(np.mean(z))
        return out, len(feats)


def parse(stem):
    m = NAME.match(stem)
    if not m:
        return None
    spk, lang, src, n, style = m.groups()
    return dict(speaker=spk.lower(), language=lang.lower(), source=src.lower(),
                index=int(n) if n else 1, style=(style or "unknown").lower())


def main():
    if not MODEL_DIR.exists():
        sys.exit(f"put frontend_int8.onnx and heads.json in {MODEL_DIR}")
    if not CLIP_DIR.exists():
        sys.exit(f"put the renamed clips in {CLIP_DIR}")

    files = sorted(p for p in CLIP_DIR.iterdir() if p.is_file())
    if not files:
        sys.exit(f"no files in {CLIP_DIR}")

    bad = [p.name for p in files if parse(p.stem) is None]
    if bad:
        print(f"{len(bad)} files do not match speaker_language_source_index_style:")
        for name in bad[:10]:
            print(f"{name}")
        print("rename them or move them out, then run again")
        return

    det = Detector(MODEL_DIR)
    print(f"layer {det.meta.get('layer')}, heads {list(det.heads)}")
    print(f"{len(files)} clips\n")

    rows, t0 = [], time.time()
    for p in files:
        meta = parse(p.stem)
        try:
            audio, sr = decode(p)
        except Exception as exc:
            print(f"skipped {p.name}: {exc}")
            continue

        prepared = bottleneck(audio, sr)
        scores, n_win = det.score(prepared)
        rows.append(dict(file=p.name, **meta, sr=sr, secs=round(len(audio) / sr, 1),
                         n_win=n_win, band_ratio=band_ratio(audio, sr),
                         **{f"z_{k}": v for k, v in scores.items()}))

    df = pd.DataFrame(rows)
    for head in det.heads:
        df[f"rejected_{head}"] = df[f"z_{head}"] > 0
    df.to_csv(OUT_CSV, index=False)
    print(f"scored {len(df)} clips in {time.time()-t0:.0f}s, wrote {OUT_CSV}\n")

    # Every clip here is a real human, so a positive score is always an error. There is no
    # miss rate to report and no EER, only the false rejection rate.
    scope = df[~df.language.isin(OUT_OF_SCOPE)]
    held = df[df.language.isin(OUT_OF_SCOPE)]
    n_spk = scope.speaker.nunique()
    for head in det.heads:
        print(f"{head}: rejects {100*scope[f'rejected_{head}'].mean():.1f}% of "
              f"{len(scope)} genuine clips from {n_spk} speakers")
    if len(held):
        print("\nheld out of the rate above, outside the population this project measured:")
        for _, r in held.iterrows():
            calls = " ".join(f"{h} z={r[f'z_{h}']:+.2f}" for h in det.heads)
            print(f"{r.file} {calls}")

    for axis in ("source", "language", "style", "speaker"):
        print(f"\nby {axis}")
        g = df.groupby(axis).agg(n=("file", "size"),
                                 **{h: (f"rejected_{h}", "mean") for h in det.heads})
        for name, row in g.iterrows():
            calls = " ".join(f"{h} {100*row[h]:.0f}%" for h in det.heads)
            print(f"{name}, n={int(row['n'])}: {calls}")

    # Whether the three sources really are three channels, rather than three names for one.
    print("\nenergy above 4 kHz before the bottleneck, by source")
    for name, row in df.groupby("source").band_ratio.agg(["median", "size"]).iterrows():
        print(f"{name}, n={int(row['size'])}: {row['median']:.4f}")

    wrong = df[df.rejected_fixed].sort_values("z_fixed", ascending=False)
    if len(wrong):
        print(f"\n{len(wrong)} clips the shipped model got wrong, worst first. Listen to these:")
        for _, r in wrong.iterrows():
            print(f"{r.file} z={r.z_fixed:+.2f} {r.secs}s {r.n_win} windows")
    else:
        print("\nthe shipped model accepted every clip")


if __name__ == "__main__":
    main()
