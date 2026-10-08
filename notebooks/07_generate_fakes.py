"""Generate the synthetic half of the dataset with two attack systems.

Every real utterance gets one fake, half cloned with XTTS-v2 and half converted with FreeVC, so
the training condition holds two generator architectures. A later collapse on Hinglish produced
by the same two generators then cannot be explained as generator novelty.

Two matching rules keep the classes separable only by synthesis. Speakers are matched within
condition, since a Hinglish fake voiced by a monolingual train speaker would be an identity in
no real file. And every file passes native -> 8 kHz -> 16 kHz before writing: real audio holds
nothing above 4 kHz where generators emit 24 kHz, a gap measured at 2.3 million on one clip.

Bengali is FreeVC-only, since XTTS has no `bn`.

Requirements
    Kaggle notebook, GPU T4, internet on (~2.5 GB). Attached input: the real dataset.
    Roughly 7 GPU-hours, so the two generation cells commit independently. Set LIMIT first.
"""

# CELL 1: configuration and environment
import os
import sys
import json
import glob
import time
import shutil
import subprocess

KAGGLE_USER = "aryankumarpandey"
ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
OUT = "/kaggle/working/fakes"
PLAN_CSV = "/kaggle/working/plan_fakes.csv"
MANIFEST_CSV = "/kaggle/working/manifest_fake.csv"

# A previous run's output, re-attached as an input. Found by looking for a `fakes` directory
# among the mounted inputs rather than by hand: the mount path depends on the notebook slug,
# and a path that is wrong or left unset re-renders everything already generated, at full cost,
# without erroring.
RESUME_ROOT = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/fakes")) if os.path.isdir(p)),
    None)

# The two passes together exceed a single session, so each is committed separately: run with
# "xtts", attach that run's output, then run with "both", the finished files are skipped, so
# the second commit costs only the outstanding ones.
RUN_PASS = "xtts"

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
VC_MODEL = "voice_conversion_models/multilingual/vctk/freevc24"
BOTTLENECK_HZ = 8000
TARGET_HZ = 16000
# the seed notebook 02 recorded for the splits
SEED = 1337
# set to e.g. 24 for a smoke run over both attacks
LIMIT = None

os.makedirs(OUT, exist_ok=True)
if not os.path.isdir(ROOT):
    sys.exit(f"dataset not mounted at {ROOT}, use + Add Input")

os.environ["COQUI_TOS_AGREED"] = "1"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "coqui-tts"], check=True)
# XTTS expands digits in the transcript through num2words, which raises a bare
# NotImplementedError for any language its installed version does not cover. Every utterance
# containing a digit then fails for a reason unrelated to synthesis.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "num2words"], check=True)

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {DEVICE}")

MAN = pd.read_csv(f"{ROOT}/manifest_real.csv")
print(f"{len(MAN):,} real utterances, {MAN.dur.sum()/3600:.1f} h")


# CELL 2: assign an attack and a voice to every real utterance
RNG = np.random.RandomState(SEED)

plan = MAN.copy()
plan["attack"] = ""

# XTTS speaks Hindi but not Bengali, so the Bengali condition can only be served by the
# language-agnostic system. Everything else splits evenly, stratified within
# (set, split, speaker) so both generators cover every voice, one generator claiming whole
# speakers would reintroduce identity as a class cue.
for key, g in plan.groupby(["set", "split", "speaker"], sort=True):
    n = len(g)
    if key[0] == "bengali_english":
        a = np.array(["freevc"] * n)
    else:
        a = np.array(["xtts", "freevc"] * ((n + 1) // 2))[:n]
        RNG.shuffle(a)
    plan.loc[g.index, "attack"] = a

# XTTS expands digits through num2words, which has no Hindi converter and raises rather than
# falling back, so every transcript containing one fails. FreeVC never reads the text, so those
# utterances move to it rather than being dropped: a fake missing from a condition leaves its
# real counterpart unpaired there, and all of these fall in the code-switched test set.
digits = plan.transcript.fillna("").str.contains(r"\d", regex=True)
plan.loc[digits & (plan.attack == "xtts"), "attack"] = "freevc"

# FreeVC converts one real speaker towards another. The target has to sit in the same
# condition *and* the same split: a train-split fake wearing a dev speaker's identity would
# leak that speaker across the boundary the splits exist to enforce.
plan["target_speaker"] = ""
for key, g in plan[plan.attack == "freevc"].groupby(["set", "split"], sort=True):
    pool = sorted(MAN[(MAN.set == key[0]) & (MAN.split == key[1])].speaker.dropna().unique())
    if len(pool) < 2:
        raise SystemExit(f"{key} has {len(pool)} speakers, cannot convert across speakers")
    for spk, gs in g.groupby("speaker", sort=True):
        others = [s for s in pool if s != spk]
        # Cycled rather than sampled, so every speaker is used as a target about equally and
        # the fake-side identity distribution matches the real-side one.
        plan.loc[gs.index, "target_speaker"] = [others[i % len(others)] for i in range(len(gs))]

# The suffix is appended after stripping any existing one, not substituted into the name. The
# manifest's fname carries no extension, and a substitution silently no-ops on such names,
# leaving files the suffix-based resume check cannot see.
plan["fake_fname"] = ("fake_" + plan.attack + "__"
                      + plan.fname.astype(str).str.replace(r"\.[^.]+$", "", regex=True)
                      + ".flac")

# XTTS clones the utterance's own speaker, so speaker and transcript are both held constant
# and synthesis is the only thing that differs from the paired real file.
plan["voiced_by"] = np.where(plan.attack == "xtts", plan.speaker, plan.target_speaker)

assert (plan.attack != "").all()
assert (plan[plan.attack == "freevc"].target_speaker != "").all()
assert (plan[plan.attack == "freevc"].target_speaker != plan[plan.attack == "freevc"].speaker).all()

voiced_in_condition = plan.merge(
    MAN[["set", "split", "speaker"]].drop_duplicates(),
    left_on=["set", "split", "voiced_by"], right_on=["set", "split", "speaker"],
    how="left", suffixes=("", "_real"))
assert voiced_in_condition.speaker_real.notna().all(), \
    "a fake is voiced by an identity absent from the reals of its own condition and split"

plan.to_csv(PLAN_CSV, index=False)
print(plan.groupby(["set", "split", "attack"]).size().rename("fakes").to_string())
mean_dur = float(MAN.dur.mean())
print(f"\n{len(plan):,} fakes planned, 1 per real")
print(f"xtts {int((plan.attack=='xtts').sum()):,} at RTF 0.60, about "
      f"{(plan.attack=='xtts').sum()*mean_dur*0.60/3600:.1f} GPU-h")
print(f"freevc {int((plan.attack=='freevc').sum()):,} at RTF 0.43, about "
      f"{(plan.attack=='freevc').sum()*mean_dur*0.43/3600:.1f} GPU-h")


# CELL 3: shared helpers
def bottleneck(x, sr):
    """The path every real file took in notebook 02. Not optional, see the module docstring."""
    return soxr.resample(soxr.resample(x, sr, BOTTLENECK_HZ), BOTTLENECK_HZ, TARGET_HZ)


def write_fake(x, sr, fname):
    """Bottleneck, then store in the same container and bit depth as the reals."""
    y = bottleneck(x, sr)
    peak = np.max(np.abs(y))
    if peak > 0.99:
        # resampling can overshoot; clipping would be its own artifact
        y = y * (0.99 / peak)
    tmp = os.path.join(OUT, f".{fname}.part")
    sf.write(tmp, y.astype("float32"), TARGET_HZ, subtype="PCM_16", format="FLAC")
    # atomic, so a killed session leaves no stub
    os.replace(tmp, os.path.join(OUT, fname))


def already_done():
    return {f for f in os.listdir(OUT) if f.endswith(".flac")}


# A previous pass's output is attached read-only, and the dataset that gets published has to be
# one directory, so those files are carried into OUT rather than read from two places. Anything
# written before the suffix was corrected arrives without an extension and is normalised on the
# way in. The audio is FLAC either way, since the format was set explicitly at write time.
def carry_forward():
    if not RESUME_ROOT:
        print("RESUME_ROOT not set, every planned file will be rendered from scratch")
        return
    if not os.path.isdir(RESUME_ROOT):
        sys.exit(f"RESUME_ROOT set but nothing mounted at {RESUME_ROOT}")
    carried = 0
    for f in os.listdir(RESUME_ROOT):
        src = os.path.join(RESUME_ROOT, f)
        if not os.path.isfile(src) or not f.startswith("fake_"):
            continue
        dst = os.path.join(OUT, f if f.endswith(".flac") else f + ".flac")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            carried += 1
    print(f"carried {carried:,} files forward from {RESUME_ROOT}")


carry_forward()


def reference_for(speaker, in_set, exclude_fname=None, n_clips=4):
    """Concatenate a speaker's longest clips into the >=6 s reference XTTS and FreeVC want.

    The utterance being generated is excluded from its own reference. Otherwise the model is
    handed the exact content it is being asked to produce, and those few clips would come out
    unlike the rest.
    """
    pool = MAN[(MAN.speaker == speaker) & (MAN.set == in_set)]
    if exclude_fname is not None:
        pool = pool[pool.fname != exclude_fname]
    pool = pool.sort_values("dur", ascending=False).head(n_clips)
    buf = []
    for r in pool.itertuples():
        x, sr = sf.read(f"{ROOT}/{r.path}", dtype="float32")
        buf.append(x)
        buf.append(np.zeros(int(0.15 * sr), dtype="float32"))
    ref = np.concatenate(buf)
    # The excluded variant gets its own filename; writing both to one path would overwrite the
    # cached reference that every other utterance by this speaker is still pointing at.
    tag = "" if exclude_fname is None else "_excl"
    path = os.path.join(REF_DIR, f"ref_{in_set}_{speaker}{tag}.wav")
    sf.write(path, ref, TARGET_HZ, subtype="PCM_16")
    return path, len(ref) / TARGET_HZ


REF_DIR = "/kaggle/working/refs"
os.makedirs(REF_DIR, exist_ok=True)

# Built once per speaker and reused. The handful of utterances that appear in their own
# speaker's reference get a second reference drawn from the next-longest clips instead.
REF_CACHE = {}


def cached_reference(speaker, in_set, own_fname):
    key = (speaker, in_set)
    if key not in REF_CACHE:
        REF_CACHE[key] = reference_for(speaker, in_set)
        top = (MAN[(MAN.speaker == speaker) & (MAN.set == in_set)]
               .sort_values("dur", ascending=False).head(4).fname.tolist())
        REF_CACHE[(speaker, in_set, "top")] = set(top)
    if own_fname in REF_CACHE[(speaker, in_set, "top")]:
        alt = (MAN[(MAN.speaker == speaker) & (MAN.set == in_set)]
               .sort_values("dur", ascending=False).iloc[4:12])
        if len(alt) >= 2:
            return reference_for(speaker, in_set, exclude_fname=own_fname)[0]
    return REF_CACHE[key][0]


def run_pass(rows, synth, label):
    """Render `rows`, skipping what exists, recording one status per utterance."""
    done = already_done()
    todo = [r for r in rows.itertuples() if r.fake_fname not in done]
    print(f"{label}: {len(rows):,} planned, {len(rows)-len(todo):,} already present, {len(todo):,} to render")
    if LIMIT:
        todo = todo[:LIMIT]
        print(f"LIMIT set, rendering {len(todo)}")

    log, t_start = [], time.time()
    for i, r in enumerate(todo, 1):
        try:
            x, sr = synth(r)
            write_fake(x, sr, r.fake_fname)
            log.append(dict(fake_fname=r.fake_fname, status="ok"))
        except Exception as e:
            log.append(dict(fake_fname=r.fake_fname, status=f"fail: {type(e).__name__}: {e}"))
        if i % 100 == 0 or i == len(todo):
            el = time.time() - t_start
            print(f"{i}/{len(todo)} {el/60:.1f} min elapsed, "
                  f"{el/i*(len(todo)-i)/60:.0f} min left")
            pd.DataFrame(log).to_csv(f"/kaggle/working/log_{label}.csv", index=False)
    # Columns are named explicitly: a pass with nothing outstanding logs no rows, and an empty
    # frame built from an empty list has no columns to select on.
    out = pd.DataFrame(log, columns=["fake_fname", "status"])
    out.to_csv(f"/kaggle/working/log_{label}.csv", index=False)
    bad = out[out.status != "ok"]
    print(f"{label} done: {len(out)-len(bad)} ok, {len(bad)} failed")
    if len(bad):
        print(bad.status.str.slice(0, 80).value_counts().head().to_string())
    return out


# CELL 4: attack A, XTTS-v2 speaker cloning
from TTS.api import TTS

TMP_WAV = "/kaggle/working/_xtts_tmp.wav"

if RUN_PASS in ("xtts", "both"):
    t0 = time.time()
    xtts = TTS(XTTS_MODEL).to(DEVICE)
    print(f"xtts loaded in {time.time()-t0:.0f}s")

    def synth_xtts(r):
        ref = cached_reference(r.speaker, r.set, r.fname)
        # 'hi' for the Hinglish transcripts too; the gate confirmed English inside a
        # Devanagari sentence is pronounced as English in this mode.
        xtts.tts_to_file(text=r.transcript, speaker_wav=ref, language="hi", file_path=TMP_WAV)
        return sf.read(TMP_WAV, dtype="float32")

    xtts_log = run_pass(plan[plan.attack == "xtts"], synth_xtts, "xtts")
    del xtts
    torch.cuda.empty_cache()
else:
    print(f"xtts pass skipped, RUN_PASS={RUN_PASS}")


# CELL 5: attack B, FreeVC voice conversion
VC_TMP = "/kaggle/working/_vc_tmp.wav"

if RUN_PASS in ("freevc", "both"):
    t0 = time.time()
    vc = TTS(model_name=VC_MODEL).to(DEVICE)
    print(f"freevc loaded in {time.time()-t0:.0f}s")

    def synth_freevc(r):
        tgt = cached_reference(r.target_speaker, r.set, own_fname=None)
        vc.voice_conversion_to_file(source_wav=f"{ROOT}/{r.path}", target_wav=tgt,
                                    file_path=VC_TMP)
        return sf.read(VC_TMP, dtype="float32")

    vc_log = run_pass(plan[plan.attack == "freevc"], synth_freevc, "freevc")
    del vc
    torch.cuda.empty_cache()
else:
    print(f"freevc pass skipped, RUN_PASS={RUN_PASS}")


# CELL 6: verify before anything is published
def fake_path(fname):
    return os.path.join(OUT, fname)


rendered = already_done()
fake = plan[plan.fake_fname.isin(rendered)].copy()
fake["dur_fake"] = [sf.info(fake_path(f)).duration for f in fake.fake_fname]

print(f"{len(fake):,}/{len(plan):,} rendered")

# Unrendered rows leave their real counterpart unpaired, so the reason matters. Digits are the
# known cause: XTTS routes them through num2words, which covers fewer languages than XTTS does.
absent = plan[~plan.fake_fname.isin(rendered)]
if len(absent):
    digits = absent.transcript.fillna("").str.contains(r"\d", regex=True)
    print(f"{len(absent):,} absent, {int(digits.sum())} of which contain digits")
print(fake.groupby(["set", "split", "attack"]).size().rename("fakes").to_string())

# Class balance per split, which is what the training code will assume.
bal = pd.DataFrame({"real": MAN.groupby(["set", "split"]).size(),
                    "fake": fake.groupby(["set", "split"]).size()}).fillna(0).astype(int)
bal["ratio"] = (bal.fake / bal.real).round(3)
print("\n" + bal.to_string())


def band_ratio(x, sr, split_hz=BOTTLENECK_HZ // 2):
    n = min(len(x), sr * 4)
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    return float(spec[freqs >= split_hz].sum() / (spec[freqs < split_hz].sum() + 1e-12))


# The confound this whole pipeline is arranged around, checked on the files as written rather
# than on arrays in memory: above 4 kHz all that survives is the 16-bit quantisation floor,
# which a float32 array does not have.
sample = fake.groupby("attack", group_keys=False).head(40)
real_br = np.median([band_ratio(*sf.read(f"{ROOT}/{r.path}", dtype="float32"))
                     for r in MAN.sample(40, random_state=SEED).itertuples()])
print(f"\nreal median above 4 kHz {real_br:.2e}")
for atk, g in sample.groupby("attack"):
    br = np.median([band_ratio(*sf.read(fake_path(f), dtype="float32"))
                    for f in g.fake_fname])
    dec = abs(np.log10(max(br, 1e-30) / max(real_br, 1e-30)))
    print(f"{atk} median {br:.2e}, {dec:.1f} decades from real, "
          + ("matched" if dec < 1 else "SEPARABLE, do not publish this version"))

# The XTTS gate spelled `ok` out letter by letter. If that happens consistently, utterances
# containing short Latin tokens come out longer than their reals, and a detector can learn
# "spelled letters = fake" instead of learning synthesis. Duration inflation is the cheap
# proxy: no ASR needed, and it separates the affected utterances from the rest.
short_latin = fake.transcript.fillna("").str.split().apply(
    lambda ws: any(w.isascii() and w.isalpha() and len(w) <= 3 for w in ws))
fake["inflation"] = fake.dur_fake / fake.dur
for atk, g in fake.groupby("attack"):
    m = short_latin.loc[g.index]
    if m.any() and (~m).any():
        print(f"\n{atk} duration ratio: with short Latin tokens {g.inflation[m].median():.2f} "
              f"(n={int(m.sum())}), without {g.inflation[~m].median():.2f} (n={int((~m).sum())})")

fake["label"] = "fake"
fake["role"] = fake.role.astype(str)
fake.to_csv(MANIFEST_CSV, index=False)
print(f"\nwrote {MANIFEST_CSV}")


# CELL 7: dataset card
card = """# codeswitch-deepfake-fakes

Synthetic counterpart to `codeswitch-deepfake-real`. One fake per real utterance,
{len(fake):,} files, 16 kHz mono FLAC.

## Attack systems

| Attack | Model | Architecture | Coverage |
|---|---|---|---|
| A | XTTS-v2 | autoregressive tokens + HiFi-GAN | Hindi, Hinglish |
| B | FreeVC (freevc24) | VITS flow + HiFi-GAN over WavLM content | all three conditions |

A third system is held out entirely and never appears here, so attack generalisation can be
measured against data no training run has seen.

Bengali is attack B only, because XTTS does not support `bn`. Comparisons involving Bengali
are restricted to the FreeVC subset.

Attacks are otherwise split evenly within every (set, split, speaker) group, with one
exception: transcripts containing digits are always attack B. XTTS expands numbers through
num2words, which has no Hindi converter and raises rather than falling back. All 86 such
utterances fall in the code-switched test set, which therefore carries 1,609 B against 1,454 A
rather than an even split.

## Known characteristics

XTTS output runs about 9% shorter than its paired real file on code-switched transcripts and
about 10% longer on monolingual ones. This is not content loss: binning the code-switched set
by Latin-script share leaves the ratio flat at ~0.91 whatever proportion of the transcript is
English, and listening confirms the English words are present. FreeVC sits at 0.996 throughout,
since conversion cannot alter timing.

Above 4 kHz both attacks sit below the reals after the bottleneck, XTTS by a factor of 4,
FreeVC by 8, against a real median of 8.3e-09. Both are within the one-decade tolerance this
dataset was built to, but a detector reaching implausible accuracy should be tested against
that single feature before the result is believed.

## Controls

Speakers are matched within condition and split. An XTTS fake is voiced by the same speaker
and speaks the same transcript as the real file it pairs with, so synthesis is the only
difference. A FreeVC fake converts a real utterance towards another speaker of the same
condition and split, cycled so the fake-side identity distribution matches the real side.

Every file passed through native -> 8 kHz -> 16 kHz, identical to the real audio, which was
frozen through that path and holds no energy above 4 kHz. Without it, the raw 24 kHz output
of both generators carried roughly 2.3e6 times the real files' energy above 4 kHz.

## Provenance

Source speech: MUCS / OpenSLR 104. Seed {SEED}. Splits inherited from the real dataset and
not recomputed. Generation notebook: `07_generate_fakes.py`.

## Licences

XTTS-v2 is CPML, non-commercial; academic research is within scope. FreeVC is MIT, and its
checkpoint is trained on VCTK (CC BY 4.0), which carries an attribution requirement. Neither
generator is shipped in the deployed demo, which runs the detector only.
"""
with open("/kaggle/working/DATASET_CARD.md", "w", encoding="utf-8") as f:
    f.write(card)
print(card)
