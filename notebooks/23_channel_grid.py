"""Degrade a paired sample through four transmission channels.

Corpus comparisons confound room, microphone and session at once. This makes the channel the
only variable: one set of utterances through real telephony codecs, scored against itself.
Degradation is CPU and ffmpeg while the evaluation needs a GPU, so they are separate files.

The grid is paired, so the same utterances pass through every channel and the effect is
within-utterance. It is sampled because all 31,922 through four channels is five times the
cost of the original feature pass.

Bandwidth is neutralised and the codecs cannot undo it. Everything already passed a native,
8 kHz, 16 kHz bottleneck, so the clean set carries no measurable energy above 4 kHz at all
(median 0.000000). The codecs add a little coding noise back rather than removing speech
bandwidth: AMR-NB 2e-6, G.711 3e-6, Opus 6e-6, Opus with loss 1.0e-4. Separability between
channels is the treatment. The confound would be a channel landing differently on real and
fake, which cell 6 measures before any GPU is spent.

Requirements
    Kaggle notebook, CPU.
    Attached inputs: codeswitch-deepfake-real and codeswitch-deepfake-fakes.
    Output is roughly 4 GB of FLAC.
"""

# CELL 1: configuration and mounts

import os
import sys
import glob
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
WORK = "/kaggle/working/channels"
MANIFEST_CSV = "/kaggle/working/manifest_channels.csv"

SR = 16000
SEED = 1337
N_PER_CONDITION = 600

# Named as notebook 20 names them, so the two read against each other directly.
CONDITIONS = {
    "in-domain control": lambda d: d["role"] == "control_indomain",
    "Hindi-English": lambda d: d["set"] == "hindi_english",
    "Bengali-English": lambda d: d["set"] == "bengali_english",
}

FAKE_DIR = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/fakes")) if os.path.isdir(p)),
    None)
if not os.path.isdir(REAL_ROOT):
    sys.exit(f"real dataset not mounted at {REAL_ROOT}, use + Add Input")
if FAKE_DIR is None:
    sys.exit("no `fakes` directory among the mounted inputs, attach codeswitch-deepfake-fakes")
FAKE_ROOT = os.path.dirname(FAKE_DIR)
print(f"real  {REAL_ROOT}")
print(f"fakes {FAKE_DIR}")


# CELL 2: the channels, probed before the sampling

# Loss rides on Opus: it is a property of packet transport.
LOSS_RATE = 0.10
LOSS_FRAME_MS = 20

CHANNELS = {
    "clean": None,
    "opus_12k": ["-c:a", "libopus", "-b:a", "12k", "-ar", "16000", "-f", "ogg"],
    "amrnb_6k7": ["-c:a", "libopencore_amrnb", "-b:a", "6.7k", "-ar", "8000", "-ac", "1",
                  "-f", "amr"],
    "g711_ulaw": ["-c:a", "pcm_mulaw", "-ar", "8000", "-ac", "1", "-f", "wav"],
    "opus_12k_loss10": ["-c:a", "libopus", "-b:a", "12k", "-ar", "16000", "-f", "ogg"],
}
NEEDED = sorted({a[a.index("-c:a") + 1] for a in CHANNELS.values() if a})


def encoders_of(exe):
    try:
        return subprocess.run([exe, "-hide_banner", "-encoders"],
                              capture_output=True, text=True).stdout
    except OSError:
        return ""


def pick_ffmpeg():
    """Kaggle's system ffmpeg is built without libopencore_amrnb; the imageio-ffmpeg static
    build carries it. Choose on capability rather than on which one is first on PATH."""
    seen = []
    cands = [shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        print("imageio-ffmpeg not installed; pip install it if the system build is short")
    for exe in [c for c in cands if c]:
        missing = [c for c in NEEDED if f" {c} " not in encoders_of(exe)]
        print(f"{exe}: {'all codecs' if not missing else 'missing ' + ', '.join(missing)}")
        seen.append((exe, missing))
    for exe, missing in seen:
        if not missing:
            return exe
    sys.exit("no available ffmpeg carries every codec. `pip install -q imageio-ffmpeg` with "
             "internet on, or drop amrnb_6k7 from CHANNELS and note the gap in the results.")


FFMPEG = pick_ffmpeg()
print(f"using {FFMPEG}")

encoders = encoders_of(FFMPEG)
for name, args in CHANNELS.items():
    if args is None:
        continue
    codec = args[args.index("-c:a") + 1]
    ok = f" {codec} " in encoders
    print(f"{name:18s} {codec:22s} {'available' if ok else 'MISSING from this ffmpeg build'}")
    if not ok:
        sys.exit(f"encoder {codec} missing from this ffmpeg build. A missing channel is a "
                 f"missing column of the grid, so install it rather than run with a hole.")


# CELL 3: the paired sample

real = pd.read_csv(f"{REAL_ROOT}/manifest_real.csv")
fake = pd.read_csv(f"{FAKE_ROOT}/manifest_fake.csv")

picked = []
for label, test in CONDITIONS.items():
    pool = real[test(real)]
    # Spread across speakers; a sample concentrated in a few sessions would measure those
    # sessions, which is this project's own failure mode.
    per_spk = max(1, int(np.ceil(N_PER_CONDITION / pool.speaker.nunique())))
    # Not groupby().apply(sample): recent pandas drops the grouping column there, so it loses
    # `speaker` on Kaggle while working locally. head() reads from the original frame.
    spread = (pool.sample(frac=1, random_state=SEED)
                  .groupby("speaker", sort=False).head(per_spk))
    take = spread.sample(min(N_PER_CONDITION, len(spread)), random_state=SEED)
    take = take.assign(condition_grid=label)
    picked.append(take)
    print(f"{label:18s} {len(take):4d} of {len(pool):5d} available, "
          f"{take.speaker.nunique()} speakers")

sample_real = pd.concat(picked, ignore_index=True)

# One fake per real, so the pairing is exact: both sides of a cell are the same utterances.
paired = fake[fake.fname.isin(set(sample_real.fname))]
assert paired.fname.is_unique, "more than one fake per real utterance"
missing = set(sample_real.fname) - set(paired.fname)
if missing:
    print(f"{len(missing)} sampled reals have no fake and are dropped from both sides")
    sample_real = sample_real[~sample_real.fname.isin(missing)]
    paired = fake[fake.fname.isin(set(sample_real.fname))]

grid = pd.concat([
    sample_real.assign(label="real", src=REAL_ROOT + "/" + sample_real.path,
                       out_name=sample_real.fname),
    paired.merge(sample_real[["fname", "condition_grid"]], on="fname")
          .assign(label="fake", src=lambda d: FAKE_DIR + "/" + d.fake_fname,
                  out_name=lambda d: d.fake_fname.str.replace(".flac", "", regex=False)),
], ignore_index=True)[["fname", "out_name", "src", "label", "condition_grid", "speaker"]]

print(f"\n{len(grid):,} utterances × {len(CHANNELS)} channels = "
      f"{len(grid) * len(CHANNELS):,} files")
print(grid.groupby(["condition_grid", "label"]).size().rename("utts").to_string())


# CELL 4: degrade

def band_ratio(x, sr=SR):
    """Energy above 4 kHz as a fraction of the total."""
    if len(x) < 512:
        return float("nan")
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    total = spec.sum()
    return float(spec[freqs > 4000].sum() / total) if total > 0 else float("nan")


def drop_frames(x, rate=LOSS_RATE, frame_ms=LOSS_FRAME_MS, rs=None):
    """Zero whole frames at random: loss with no concealment. Worst case, not typical, since a
    real decoder interpolates."""
    n = int(SR * frame_ms / 1000)
    y = x.copy()
    for start in range(0, len(y) - n, n):
        if rs.rand() < rate:
            y[start:start + n] = 0.0
    return y


def through(src, args, lossy, rs):
    """One encode/decode round trip, returning 16 kHz mono float32."""
    x, sr = sf.read(src, dtype="float32")
    if args is None:
        out = x
    else:
        with tempfile.TemporaryDirectory() as td:
            raw, enc = f"{td}/in.wav", f"{td}/enc"
            sf.write(raw, x, sr)
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", raw, *args, enc],
                           check=True)
            dec = f"{td}/out.wav"
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", enc,
                            "-ar", str(SR), "-ac", "1", dec], check=True)
            out, _ = sf.read(dec, dtype="float32")
    if lossy:
        out = drop_frames(out, rs=rs)
    return out


rows = []
for ci, (channel, args) in enumerate(CHANNELS.items()):
    out_dir = f"{WORK}/{channel}"
    os.makedirs(out_dir, exist_ok=True)
    # Seeded off position, not hash(channel): Python randomises string hashes per process, so
    # the loss pattern would differ between runs of an otherwise reproducible notebook.
    rs = np.random.RandomState(SEED + ci)
    for i, r in enumerate(grid.itertuples()):
        dst = f"{out_dir}/{r.out_name}.flac"
        if os.path.exists(dst):
            continue
        try:
            y = through(r.src, args, channel.endswith("loss10"), rs)
        except subprocess.CalledProcessError:
            rows.append(dict(channel=channel, fname=r.fname, status="ffmpeg failed"))
            continue
        sf.write(dst, y, SR, format="FLAC", subtype="PCM_16")
        rows.append(dict(channel=channel, fname=r.fname, out_name=r.out_name,
                         label=r.label, condition_grid=r.condition_grid, speaker=r.speaker,
                         path=f"{channel}/{r.out_name}.flac", secs=len(y) / SR,
                         band_ratio=band_ratio(y), status="ok"))
        if i and i % 500 == 0:
            print(f"{channel} {i:,}/{len(grid):,}")
    print(f"{channel} done")

MAN = pd.DataFrame(rows)
MAN.to_csv(MANIFEST_CSV, index=False)
failed = MAN[MAN.status != "ok"]
if len(failed):
    print(f"\n{len(failed)} files failed, by channel:")
    print(failed.groupby("channel").size().to_string())
print(f"\nwrote {MANIFEST_CSV}, {len(MAN):,} rows")


# CELL 5: did the channels do something different

# Codecs that left the audio unchanged give a clean-looking grid that means nothing.
ok = MAN[MAN.status == "ok"]
print("\nmedian energy above 4 kHz, by channel")
for channel, g in ok.groupby("channel"):
    print(f"  {channel:18s} n={len(g):5d}  {g.band_ratio.median():.6f}")

print("\nmedian duration, by channel. A codec that changed the length changed the alignment")
for channel, g in ok.groupby("channel"):
    print(f"  {channel:18s} {g.secs.median():.2f}s")


# CELL 6: the confound check, on CPU, before any GPU is spent

# Not "is the channel detectable", it is, that is the treatment. "Does it land differently on
# real and fake", which would let a detector separate them on coding artifacts. Method from
# nb 21b, which is also why a median gap is not trusted here.
def auc(y, s):
    """Rank AUC, no sklearn import needed."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    if y.all() or not y.any():
        return float("nan")
    r = pd.Series(s).rank().values
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


print("\nband-ratio AUC separating real from fake, within each channel")
print("0.5 is chance. Above ~0.65 the channel introduced an artifact that tracks the label.")
for channel, g in ok.groupby("channel"):
    a = auc(g.label.values == "fake", g.band_ratio.values)
    flag = "" if abs(a - 0.5) < 0.15 else "   <-- check"
    print(f"  {channel:18s} n={len(g):5d}  AUC {a:.3f}{flag}")

# `clean` is the reference and should not have moved. If it has, the cause predates the grid.


# CELL 7: one clip per cell, to listen to

# Unintelligible speech is not a channel, it is noise, and rejecting it would be correct
# rather than broken. Only the ear settles that.
import IPython.display as ipd

demo = ok[ok.label == "real"].groupby(["condition_grid", "channel"], group_keys=False).head(1)
for r in demo.itertuples():
    print(f"\n[{r.condition_grid} | {r.channel} | {r.secs:.1f}s | "
          f"band {r.band_ratio:.6f}]")
    ipd.display(ipd.Audio(f"{WORK}/{r.path}"))


# CELL 8: does the band-ratio gap also order genuine speech by corpus

# Cell 6 put fakes below reals on the clean channel, which is a property of the frozen dataset
# rather than of the grid. It only threatens the main finding if genuine speech recorded
# elsewhere also sits low, because then a band-ratio shortcut would predict rejection of
# genuine out-of-corpus speech and would be an alternative explanation for it.
MAN = pd.read_csv(MANIFEST_CSV)
ok = MAN[MAN.status == "ok"]
C = ok[ok.channel == "clean"]

print("clean channel, band ratio by condition and label")
print(C.groupby(["condition_grid", "label"]).band_ratio
      .agg(n="size", median="median", mean="mean", zeros=lambda s: int((s == 0).sum()))
      .to_string())

real = C[C.label == "real"]
out = (real.condition_grid != "in-domain control").values
# Fakes score low on this scalar, so the fake-like direction is negated band ratio.
a_corpus = auc(out, -real.band_ratio.values)
a_label = auc((C.label == "fake").values, -C.band_ratio.values)
print(f"\nfake vs real, fake-like direction:                 AUC {a_label:.3f}")
print(f"out-of-corpus genuine vs in-domain genuine, same:  AUC {a_corpus:.3f}")
print("\nNear 0.5 on the second means the scalar does not order genuine speech by corpus, so it")
print("cannot explain rejection and the first number is a discrimination shortcut only.")
print("Above ~0.65 it is a live alternative explanation and belongs in the limitations.")
