"""Decide how every fake gets generated, by testing one thing.

Whether XTTS-v2 can speak a sentence containing both Devanagari and Latin script and pronounce
the English as English. If it can, the Hindi and Hinglish fakes come from XTTS cloning. If not,
everything moves to voice conversion, which is language-agnostic and so also covers Bengali,
and is a recognised ASVspoof attack category. Either answer works.

Synthesis speed is measured here too, since it sets the GPU budget for generation.

Requirements
    Kaggle notebook, GPU T4, internet on for the ~1.8 GB download.
    Attached input: codeswitch-deepfake-real. Runtime ~15 min, mostly the download.
"""


# CELL 1: install (read the note, this is the fiddly part)
# Kaggle runs Python 3.12. The original `TTS` package from Coqui pins
# python <3.12 and will not install. Use `coqui-tts`, the maintained Idiap fork
#, same `TTS` import, same models, Python 3.12 support.
import sys, subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "coqui-tts"],
               check=True)
print("installed")

import importlib.metadata as md
for p in ("coqui-tts", "torch", "transformers", "numpy"):
    try:
        print(f"{p} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p} (absent)")

import torch
print(f"\ncuda available: {torch.cuda.is_available()}")
print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# A numpy or transformers ABI error on the CELL 2 import clears with Restart &
# Clear Cell Outputs, then a run from CELL 2. This cell can be skipped on that
# second pass, since the package is already in the session's site-packages.


# CELL 2: load XTTS-v2 and confirm which languages it claims
import os, time
# XTTS is CPML-licensed; the terms are spelled out in CELL 6
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

t0 = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"loaded in {time.time()-t0:.0f}s")

langs = list(tts.languages)
print(f"\n{len(langs)} languages: {langs}")
print(f"hindi 'hi' supported: {'hi' in langs}")
print(f"bengali 'bn' supported: {'bn' in langs} (expected False)")


# CELL 3: build a reference clip from a real MUCS speaker
# XTTS clones from 6+ seconds of reference audio and these utterances average ~3.3 s, so
# several clips from one speaker are concatenated. The speaker is taken from
# the train split. The fakes have to come from the same voices as the reals,
# otherwise the detector could separate them on speaker identity alone.
import pandas as pd, numpy as np, soundfile as sf, soxr

KAGGLE_USER = "aryankumarpandey"
ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
MAN = pd.read_csv(f"{ROOT}/manifest_real.csv")

WORK = "/kaggle/working/xtts_gate"
os.makedirs(WORK, exist_ok=True)

train = MAN[(MAN.role == "train_source") & (MAN.split == "train")]
# the speaker with the most audio, so the reference clip clears XTTS's 6 s minimum
spk = train.groupby("speaker").dur.sum().idxmax()
clips = train[train.speaker == spk].sort_values("dur", ascending=False).head(4)

buf = []
for r in clips.itertuples():
    x, sr = sf.read(f"{ROOT}/{r.path}", dtype="float32")
    buf.append(x)
    # a short gap, so the concatenation does not read as one run-on utterance
    buf.append(np.zeros(int(0.15 * sr), dtype="float32"))
ref = np.concatenate(buf)

REF = f"{WORK}/ref_{spk}.wav"
sf.write(REF, ref, 16000, subtype="PCM_16")
print(f"reference speaker: {spk}")
print(f"clips used: {len(clips)}")
print(f"reference length: {len(ref)/16000:.1f}s, XTTS wants >=6s")

import IPython.display as ipd
print("\nreference voice, for comparison against the clone:")
ipd.display(ipd.Audio(REF))


# CELL 4: the gate
# Three real transcripts straight out of the corpus, not invented examples.
mono_txt = (MAN[(MAN.set == "hindi_mono_train") & (MAN.n_words.between(8, 16))]
              .transcript.iloc[0])
cs_txt = (MAN[(MAN.set == "hindi_english") & (MAN.cs_bin == "cs_high")
                & (MAN.n_words.between(8, 16))]
              .sort_values("switches", ascending=False).transcript.iloc[0])
# NB: notebook 02 kept only rows that have a `cs_bin`, and `english_only` rows
# have none, so all 66 pure-English utterances were dropped from the frozen
# dataset. No real loss (far too few to be a bin), but it means test C has no
# corpus source. Fall back to a fixed sentence; any English will do here.
eng_pool = MAN[(MAN.condition == "english_only") & (MAN.n_words.between(6, 16))]
eng_txt = (eng_pool.transcript.iloc[0] if len(eng_pool) else
            "The committee will review the annual report before the next meeting.")

TESTS = [
    ("A_pure_hindi", mono_txt, "baseline, should obviously work"),
    ("B_CODE_SWITCHED", cs_txt, "the gate: mixed Devanagari and Latin"),
    ("C_pure_english", eng_txt, "control, does 'hi' mode mangle English?"),
]

rows = []
for name, text, why in TESTS:
    out = f"{WORK}/{name}.wav"
    t0 = time.time()
    tts.tts_to_file(text=text, speaker_wav=REF, language="hi", file_path=out)
    gpu_s = time.time() - t0
    aud_s = sf.info(out).duration
    rows.append(dict(test=name, gpu_s=round(gpu_s, 1), audio_s=round(aud_s, 1),
                     rtf=round(gpu_s / max(aud_s, 1e-9), 2), chars=len(text)))
    print(f"\n{name}, {why}")
    print(f"{text}")
    print(f"{gpu_s:.1f}s GPU -> {aud_s:.1f}s audio (RTF {gpu_s/aud_s:.2f})")
    ipd.display(ipd.Audio(out))

TIMING = pd.DataFrame(rows)
print("\n" + TIMING.to_string(index=False))


# CELL 5: GPU budget for generation, from the measured speed
# Target volume ~9,000 fakes, against a mean clip length of ~3.3 s.
rtf = TIMING.rtf.mean()
N_FAKES = 9000
AVG_SEC = float(MAN[MAN.role == "train_source"].dur.mean())
gpu_hrs = N_FAKES * AVG_SEC * rtf / 3600

print(f"mean RTF: {rtf:.2f} GPU seconds per second of audio")
print(f"mean utterance length: {AVG_SEC:.1f}s")
print(f"projected for {N_FAKES:,} fakes: {gpu_hrs:.1f} GPU-hours against a 30-hour weekly quota")
print(f"-> {'COMFORTABLE' if gpu_hrs < 8 else 'TIGHT (cut Bengali first' if gpu_hrs < 20 else 'TOO EXPENSIVE) batch or switch to VC'}")
print("\nNote: a 9-hour session cap applies, so generation must checkpoint to")
print("/kaggle/working and be resumable. Notebook 04 implements that.")


# CELL 6: the verdict, judged by ear
print("""
Test B decides the gate. Three things to judge, in order:

1. Does it sound like the reference speaker from CELL 3?
   If not, cloning is too weak for this corpus.

2. Are the English words pronounced as English?
   Listen for the words in Latin script. Read letter-by-letter, spelled out,
   skipped, or given a heavy Devanagari mispronunciation all count as a fail.

3. Is it fluent across the switch point, or does it stumble there?
   A small hesitation is realistic. A break is not.

Gate passes, on 1 and 2 both being fine:
    XTTS clones the Hindi and Hinglish fakes. Bengali still needs voice conversion
    since XTTS has no 'bn', so the design ends up with two attack types either way
    and attack diversity comes free.

Gate fails, on 2 being bad:
    Everything moves to voice conversion. That is a clean outcome rather than a
    setback. VC is language-agnostic, so one method covers Hindi, Hinglish and
    Bengali, and it is a standard ASVspoof attack category. Fakes also inherit the
    real recording channel, which removes a confound the paper would otherwise
    have to argue away.

Licence, and this belongs in the paper:
    XTTS-v2 ships under the Coqui Public Model Licence, which is non-commercial.
    Academic research is fine. State the model, version and licence in the data
    section. Keep XTTS out of the deployed demo, which runs only the detector.
""")
