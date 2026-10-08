"""Resynthesise real speech through several vocoders, to teach synthesis rather than a generator.

XTTS-v2 and FreeVC both decode through HiFi-GAN, so between them they showed the model one
vocoder family, and a downloaded clip from an unknown generator came back genuine at full
margin. Passing a real recording through a vocoder isolates the resynthesis: utterance,
speaker, words and channel are held fixed, so nothing can act as a proxy for fakeness.
Published approach, Wang and Yamagishi, arXiv 2309.06014.

Four paths, deliberately unalike, since three variations of one idea teach one artifact three
times. encodec is a neural codec with an adversarial decoder, griffinlim is classical phase
reconstruction with no learned decoder, mel_hifigan is the family the generators already use,
lowrank is spectral rank truncation. One is held out as an unseen attack, and which one is a
parameter so the claim does not rest on a lucky choice.

Requirements
    Kaggle notebook, GPU T4 for encodec and the mel vocoder, internet on (~300 MB).
    Attached input: codeswitch-deepfake-real. Resumable by re-attaching the previous output.
"""

# CELL 1: configuration and mounts
import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import soundfile as sf

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"

WORK = "/kaggle/working"
OUT = f"{WORK}/vocoded"
MANIFEST = f"{WORK}/manifest_vocoded.csv"

SEED = 1337
BOTTLENECK_HZ = 8000
TARGET_SR = 16000

# Training side needs enough to matter against 11,410 existing rows; evaluation only a stable rate.
N_TRAIN_SOURCE = 6000
N_PER_TEST_CORPUS = 1500

CORPORA = ("hindi_english", "bengali_english")

os.makedirs(OUT, exist_ok=True)
np.random.seed(SEED)

if not os.path.isdir(REAL_ROOT):
    sys.exit(f"real dataset not mounted at {REAL_ROOT}")

MAN = pd.read_csv(f"{REAL_ROOT}/manifest_real.csv")
print(f"{len(MAN):,} real utterances available")


def find(name):
    """Kaggle mounts a previous commit's output under its notebook slug."""
    return next(
        (p for depth in ("*", "*/*", "*/*/*")
         for p in sorted(glob.glob(f"/kaggle/input/{depth}/{name}")) if os.path.isdir(p)),
        None)


RESUME = find("vocoded")
if RESUME:
    print(f"resuming against {RESUME}")


# CELL 2: choose the source clips
# Per role, since a fake inherits its source's role and would otherwise leak training audio.
train_pool = MAN[(MAN.role == "train_source") & (MAN.split == "train")]
picked = [train_pool.sample(min(N_TRAIN_SOURCE, len(train_pool)), random_state=SEED)]

for corpus in CORPORA:
    pool = MAN[MAN.set == corpus]
    if len(pool):
        picked.append(pool.sample(min(N_PER_TEST_CORPUS, len(pool)), random_state=SEED))

SRC = pd.concat(picked, ignore_index=True)
print(f"\n{len(SRC):,} source clips selected")
print(SRC.groupby(["set", "role"]).size().to_string())


# CELL 3: the resynthesis paths
import soxr
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {DEVICE}")


def bottleneck(x, sr):
    """The same band every file in this project was forced through."""
    if sr != BOTTLENECK_HZ:
        x = soxr.resample(x, sr, BOTTLENECK_HZ)
    return soxr.resample(x, BOTTLENECK_HZ, TARGET_SR)


_encodec = {}


def path_encodec(x, sr):
    """Neural codec round trip. Quantised latent, adversarially trained decoder."""
    from transformers import EncodecModel, AutoProcessor
    if not _encodec:
        _encodec["proc"] = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        _encodec["model"] = EncodecModel.from_pretrained("facebook/encodec_24khz").to(DEVICE).eval()
    proc, model = _encodec["proc"], _encodec["model"]
    wav = soxr.resample(x, sr, 24000)
    inputs = proc(raw_audio=wav, sampling_rate=24000, return_tensors="pt")
    with torch.no_grad():
        enc = model.encode(inputs["input_values"].to(DEVICE),
                           inputs["padding_mask"].to(DEVICE), bandwidth=6.0)
        dec = model.decode(enc.audio_codes, enc.audio_scales,
                           inputs["padding_mask"].to(DEVICE))[0]
    return dec.squeeze().float().cpu().numpy(), 24000


_mel_basis = {}


def path_griffinlim(x, sr, n_iter=16):
    """Magnitude mel, phase discarded and reconstructed. No learned decoder anywhere.

    librosa's mel_to_audio inverts the mel filterbank with a non-negative least squares solve,
    which measured 15.5 s on a 3.4 s clip and would put this one path past thirty hours over
    the full set. A precomputed pseudo-inverse replaces it. The reconstruction is cruder, which
    for this purpose is not a cost: the artifact is the product, and a rougher mel inversion
    produces more of it, not less.
    """
    import librosa
    key = (sr, 1024, 80)
    if key not in _mel_basis:
        basis = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=80)
        _mel_basis[key] = (basis, np.linalg.pinv(basis))
    basis, inv_basis = _mel_basis[key]

    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256))
    mel = basis @ S
    approx = np.maximum(inv_basis @ mel, 0.0)
    return librosa.griffinlim(approx, n_iter=n_iter, hop_length=256, n_fft=1024,
                              length=len(x)), sr


def path_lpc(x, sr, order=16, frame=512, hop=256):
    """Source-filter resynthesis, the oldest kind of vocoder there is.

    Each frame is reduced to a short all-pole filter and re-excited by its own residual. It
    keeps the speaker and the words while discarding the fine spectral detail, which is what a
    low-capacity decoder does. It shares no machinery with the neural paths and needs no
    dependency beyond scipy, so it also survives an environment where the others do not.
    """
    import scipy.signal
    import librosa
    out = np.zeros(len(x) + frame, dtype=np.float32)
    window = np.hanning(frame).astype(np.float32)
    for start in range(0, max(len(x) - frame, 1), hop):
        seg = x[start:start + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        seg = seg * window
        if np.abs(seg).max() < 1e-6:
            continue
        try:
            a = librosa.lpc(seg.astype(np.float64), order=order)
        except Exception:
            continue
        residual = scipy.signal.lfilter(a, [1.0], seg)
        rebuilt = scipy.signal.lfilter([1.0], a, residual)
        out[start:start + frame] += rebuilt.astype(np.float32)
    out = out[:len(x)]
    peak = float(np.abs(out).max()) or 1.0
    return (out / peak) * float(np.abs(x).max() or 1.0), sr


_hifigan = {}


def path_mel_hifigan(x, sr):
    """A learned mel vocoder, the family XTTS and FreeVC both decode through."""
    from speechbrain.inference.vocoders import HIFIGAN
    import torchaudio
    if not _hifigan:
        _hifigan["v"] = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir="/kaggle/working/hifigan", run_opts={"device": DEVICE})
        _hifigan["mel"] = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80,
            f_min=0, f_max=8000, power=1).to(DEVICE)
    wav = torch.from_numpy(soxr.resample(x, sr, 16000)).float().to(DEVICE)[None]
    mel = _hifigan["mel"](wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    with torch.no_grad():
        out = _hifigan["v"].decode_batch(mel)
    return out.squeeze().float().cpu().numpy(), 16000


def path_lowrank(x, sr, keep=48):
    """Spectral rank truncation. Not a vocoder, and that is the point.

    An SVD of the magnitude spectrogram truncated to `keep` components, resynthesised with the
    original phase. It produces the smearing a low-capacity decoder produces without sharing
    any machinery with the three above, which is what makes it a fair held-out attack rather
    than a fourth sample of the same artifact.
    """
    import librosa
    S = librosa.stft(x, n_fft=1024, hop_length=256)
    mag, phase = np.abs(S), np.exp(1j * np.angle(S))
    u, s, vt = np.linalg.svd(mag, full_matrices=False)
    k = min(keep, len(s))
    mag_hat = (u[:, :k] * s[:k]) @ vt[:k]
    return librosa.istft(mag_hat * phase, hop_length=256, length=len(x)), sr


# Five defined, the smoke test keeps whichever survive. lpc is insurance against mel_hifigan.
PATHS = {
    "encodec": path_encodec,
    "griffinlim": path_griffinlim,
    "mel_hifigan": path_mel_hifigan,
    "lpc": path_lpc,
    "lowrank": path_lowrank,
}

# Held out and scored only as an unseen attack, named here so it travels in the manifest.
HELD_OUT = "lowrank"
print(f"paths: {list(PATHS)}, held out for evaluation: {HELD_OUT}")


# CELL 4: a smoke test before committing to the full pass
# One clip per path, so a missing dependency surfaces in seconds rather than forty minutes in.
probe, probe_sr = sf.read(f"{REAL_ROOT}/{SRC.iloc[0].path}", dtype="float32")
print(f"probe: {len(probe)/probe_sr:.1f}s at {probe_sr} Hz\n")

ok, timings = {}, {}
for name, fn in PATHS.items():
    t0 = time.time()
    try:
        y, y_sr = fn(probe, probe_sr)
        # Timed on the second call, since the first pays for a checkpoint download.
        t0 = time.time()
        y, y_sr = fn(probe, probe_sr)
        y = bottleneck(np.asarray(y, dtype=np.float32), y_sr)
        assert len(y) > 0.5 * len(probe) * TARGET_SR / probe_sr, "output far shorter than input"
        ok[name] = True
        timings[name] = time.time() - t0
        print(f"{name}: {timings[name]:.2f}s, {len(y)/TARGET_SR:.1f}s out, "
              f"rms {np.sqrt(np.mean(y**2)):.4f}")
    except Exception as exc:
        ok[name] = False
        print(f"{name}: FAILED, {type(exc).__name__}: {exc}")

PATHS = {k: v for k, v in PATHS.items() if ok.get(k)}
if HELD_OUT not in PATHS:
    sys.exit(f"the held-out path {HELD_OUT} failed, so there is no unseen attack to measure")
if len(PATHS) < 3:
    sys.exit("fewer than three paths survived, not enough to train on and hold one out")

# The full pass is this multiplied out, so an estimate here is cheaper than finding out later.
per_clip = sum(timings.get(k, 0) for k in PATHS)
print(f"\n{per_clip:.2f}s per source clip across {len(PATHS)} paths, "
      f"{len(SRC):,} clips, roughly {per_clip * len(SRC) / 3600:.1f} hours")
if per_clip * len(SRC) / 3600 > 6:
    print("Over six hours. Drop the slowest path or lower N_TRAIN_SOURCE before committing.")


# CELL 5: generate
# Atomic writes, skipped if present, so a session killed at the nine hour cap resumes.
rows, failed = [], []
t_start = time.time()
total = len(SRC) * len(PATHS)
done = 0

for r in SRC.itertuples():
    try:
        x, sr = sf.read(f"{REAL_ROOT}/{r.path}", dtype="float32")
    except Exception as exc:
        failed.append({"path": r.path, "err": f"read: {exc}"})
        continue
    if x.ndim > 1:
        x = x.mean(axis=1)

    for name, fn in PATHS.items():
        fname = f"voc_{name}__{os.path.splitext(os.path.basename(r.path))[0]}.flac"
        dst = os.path.join(OUT, fname)
        carried = os.path.join(RESUME, fname) if RESUME else None

        if not os.path.exists(dst) and carried and os.path.exists(carried):
            import shutil
            shutil.copyfile(carried, dst)

        if not os.path.exists(dst):
            try:
                y, y_sr = fn(x, sr)
                y = bottleneck(np.asarray(y, dtype=np.float32), y_sr)
                peak = float(np.abs(y).max()) or 1.0
                if peak > 0.99:
                    # Resampling can overshoot and clipping would be its own artifact.
                    y = y * (0.99 / peak)
                tmp = dst + ".part"
                sf.write(tmp, y, TARGET_SR, format="FLAC", subtype="PCM_16")
                os.replace(tmp, dst)
            except Exception as exc:
                failed.append({"path": r.path, "attack": name, "err": str(exc)[:200]})
                continue

        rows.append({
            "fname": fname,
            "source_path": r.path,
            "attack": f"vocoded_{name}",
            "vocoder": name,
            "held_out": name == HELD_OUT,
            "set": r.set,
            "role": r.role,
            "split": r.split,
            "speaker": r.speaker,
            "transcript": getattr(r, "transcript", None),
            "dur": sf.info(dst).duration,
        })
        done += 1

    if done and done % 500 < len(PATHS):
        el = time.time() - t_start
        rate = done / max(el, 1e-9)
        print(f"{done:,}/{total:,} {el/60:.1f} min elapsed, "
              f"eta {(total-done)/max(rate,1e-9)/60:.0f} min", flush=True)

VOC = pd.DataFrame(rows)
VOC.to_csv(MANIFEST, index=False)
print(f"\n{len(VOC):,} files written in {(time.time()-t_start)/60:.1f} min")
if failed:
    pd.DataFrame(failed).to_csv(f"{WORK}/vocoded_failures.csv", index=False)
    print(f"{len(failed)} failures, see vocoded_failures.csv")


# CELL 6: the confound checks, before any of this reaches a model
print(VOC.groupby(["vocoder", "role"]).size().to_string())

import librosa


def band_ratio(y, sr=TARGET_SR, split_hz=BOTTLENECK_HZ // 2):
    n = min(len(y), sr * 4)
    spec = np.abs(np.fft.rfft(y[:n] * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    return float(spec[freqs >= split_hz].sum() / (spec[freqs < split_hz].sum() + 1e-12))


sample = VOC.sample(min(40, len(VOC)), random_state=SEED)
print("\nenergy above 4 kHz, which must match the real side or bandwidth separates the classes")
for name, g in sample.groupby("vocoder"):
    vals = [band_ratio(sf.read(os.path.join(OUT, f), dtype="float32")[0]) for f in g.fname]
    print(f"{name}: median {np.median(vals):.2e}")

real_sample = SRC.sample(min(20, len(SRC)), random_state=SEED)
vals = [band_ratio(sf.read(f"{REAL_ROOT}/{p}", dtype="float32")[0]) for p in real_sample.path]
print(f"real: median {np.median(vals):.2e}")

print("\nduration against the source, which must not shift by class")
merged = VOC.merge(SRC[["path", "dur"]].rename(columns={"path": "source_path", "dur": "src_dur"}),
                   on="source_path", how="left")
merged["ratio"] = merged.dur / merged.src_dur
print(merged.groupby("vocoder").ratio.describe()[["mean", "std", "min", "max"]].round(3).to_string())
print("\nA mean far from 1.00 means a path is truncating or padding, and length alone would "
      "then separate the classes.")


# CELL 7: listen
# A pipeline nobody has listened to is not verified.
import IPython.display as ipd

demo = VOC.groupby("vocoder", group_keys=False).apply(
    lambda g: g.sample(1, random_state=SEED))

src_row = SRC[SRC.path == demo.iloc[0].source_path].iloc[0]
print("source")
ipd.display(ipd.Audio(f"{REAL_ROOT}/{src_row.path}"))
for r in demo.itertuples():
    print(f"{r.vocoder}{' (held out)' if r.held_out else ''}")
    ipd.display(ipd.Audio(os.path.join(OUT, r.fname)))

print("""
Each one should be the same voice saying the same words, degraded. If a path is unintelligible
it is teaching the detector to hear damage rather than resynthesis, and it should be dropped
rather than kept for the sake of having four. If a path is indistinguishable from the source,
it carries no artifact to learn and is equally useless.
""")
