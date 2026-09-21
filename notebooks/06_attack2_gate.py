"""Decide the second attack system by ear, and verify fakes match the real audio's band.

A detector trained against a single generator learns that generator. The second attack
therefore has to differ from XTTS-v2 by architecture rather than by weights. The candidate
here is FreeVC, a VITS-family voice conversion model: flow-based posterior encoder and a
HiFi-GAN decoder driven by WavLM content features, with no autoregressive token stage at
all. Being text-free it is also language-agnostic, so the same pipeline covers Bengali,
which XTTS cannot speak.

Voice conversion has a second property worth the switch. The fake inherits the source
recording's prosody, channel and switch points, so it cannot carry the text-frontend
artifacts that a TTS system introduces, the XTTS gate spelled `ok` out letter by letter,
which a detector could learn instead of learning deepfake detection.

The last cell checks a confound that applies to every attack system, not only this one.
The frozen real audio passed through native -> 8 kHz -> 16 kHz, so it holds no energy above
4 kHz. Generators emit 24 kHz. A fake that keeps its full band is separable from every real
file on bandwidth alone, which would make any result in this project uninterpretable.

A single-speaker VITS system, MMS-TTS Hindi, is also probed cheaply at the end as a
candidate for the held-out third attack. It needs no cloning and no new dependency.

Requirements
    Kaggle notebook, GPU T4 (the conversion speed measured here sets the generation budget).
    Internet: On, for FreeVC and its WavLM content encoder (~1.5 GB together).
    Attached input: the frozen real-audio dataset.
    Stop the session when finished; quota bills by session time with a GPU attached.
"""

# CELL 1: environment
import os
import sys
import time
import subprocess

KAGGLE_USER = "aryankumarpandey"
ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
WORK = "/kaggle/working/attack2_gate"
VC_MODEL = "voice_conversion_models/multilingual/vctk/freevc24"
MMS_MODEL = "facebook/mms-tts-hin"
# the rate every real file was forced through in notebook 02
BOTTLENECK_HZ = 8000
TARGET_HZ = 16000
# planned generation volume, used for the budget projection
N_FAKES = 9000

os.makedirs(WORK, exist_ok=True)

if not os.path.isdir(ROOT):
    sys.exit(f"dataset not mounted at {ROOT}, use + Add Input")

# Same fork as the XTTS gate: Kaggle is Python 3.12 and Coqui's original `TTS` pins <3.12.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "coqui-tts"], check=True)

import torch
print(f"cuda {torch.cuda.is_available()}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# CELL 2: pick a source utterance and a target speaker, both from the train split
import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import IPython.display as ipd

MAN = pd.read_csv(f"{ROOT}/manifest_real.csv")

# Speakers are matched within condition, not across the corpus. The code-switched sets carry
# role="test_source" because training is monolingual Hindi, so the real/fake comparison inside
# the Hinglish condition is between Hinglish speakers. Cloning those fakes from the monolingual
# train speakers instead would leave every fake in that condition voiced by someone who appears
# in no real file there. Separable on identity alone, and nothing to do with synthesis.
cs = MAN[MAN.set == "hindi_english"]
assert len(cs), "no hindi_english rows in the manifest"

# The source carries the content, so it has to be genuinely code-switched, that is the
# property the conversion must not destroy.
src_row = (cs[cs.n_words.between(8, 16)]
           .sort_values("switches", ascending=False).iloc[0])

# The target is a different speaker from the same condition.
targets = cs[cs.speaker != src_row.speaker]
tgt_spk = targets.groupby("speaker").dur.sum().idxmax()
tgt_clips = targets[targets.speaker == tgt_spk].sort_values("dur", ascending=False).head(4)

buf = []
for r in tgt_clips.itertuples():
    x, sr = sf.read(f"{ROOT}/{r.path}", dtype="float32")
    buf.append(x)
    buf.append(np.zeros(int(0.15 * sr), dtype="float32"))
TGT = f"{WORK}/target_{tgt_spk}.wav"
sf.write(TGT, np.concatenate(buf), TARGET_HZ, subtype="PCM_16")

SRC = f"{WORK}/source.wav"
src_audio, src_sr = sf.read(f"{ROOT}/{src_row.path}", dtype="float32")
sf.write(SRC, src_audio, src_sr, subtype="PCM_16")

print(f"source speaker {src_row.speaker} ({src_row.set}, {src_row.role}), "
      f"{src_row.dur:.1f}s, {src_row.switches} switch points")
print(f"{src_row.transcript}")
print(f"target speaker {tgt_spk}, {sum(sf.info(f'{ROOT}/{r.path}').duration for r in tgt_clips.itertuples()):.1f}s reference")

print("\nsource, the content and switching that must survive")
ipd.display(ipd.Audio(SRC))
print("target, the identity the conversion must move towards")
ipd.display(ipd.Audio(TGT))


# CELL 3: convert
from TTS.api import TTS

t0 = time.time()
vc = TTS(model_name=VC_MODEL).to(DEVICE)
print(f"model loaded in {time.time()-t0:.0f}s")

RAW = f"{WORK}/converted_raw.wav"
t0 = time.time()
vc.voice_conversion_to_file(source_wav=SRC, target_wav=TGT, file_path=RAW)
convert_s = time.time() - t0

raw_info = sf.info(RAW)
rtf = convert_s / raw_info.duration
print(f"{convert_s:.1f}s for {raw_info.duration:.1f}s audio, RTF {rtf:.2f}, output {raw_info.samplerate} Hz")

gpu_hrs = N_FAKES * float(MAN.dur.mean()) * rtf / 3600
print(f"projected {N_FAKES:,} conversions: {gpu_hrs:.1f} GPU-hours against a 30/week quota")

print("\nconverted, before the bottleneck")
ipd.display(ipd.Audio(RAW))


# CELL 4: force the fake through the same band as the reals
def bottleneck(x, sr):
    """Reproduce notebook 02's native -> 8 kHz -> 16 kHz path, which every real file took."""
    return soxr.resample(soxr.resample(x, sr, BOTTLENECK_HZ), BOTTLENECK_HZ, TARGET_HZ)


def band_ratio(x, sr, split_hz=BOTTLENECK_HZ // 2):
    """Energy above the bottleneck's Nyquist, relative to energy below it."""
    n = min(len(x), sr * 4)
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    return float(spec[freqs >= split_hz].sum() / (spec[freqs < split_hz].sum() + 1e-12))


raw_audio, raw_sr = sf.read(RAW, dtype="float32")

FAKE = f"{WORK}/converted_16k.wav"
sf.write(FAKE, bottleneck(raw_audio, raw_sr), TARGET_HZ, subtype="PCM_16")

# Read back rather than measuring the array in memory. Above 4 kHz the only thing left after
# the bottleneck is the 16-bit quantisation floor, which a float32 array does not have and
# every stored file does. Measuring one of each makes the fake look orders cleaner than the
# file that will actually be written.
fake, _ = sf.read(FAKE, dtype="float32")

real_ratio = band_ratio(src_audio, src_sr)
raw_ratio = band_ratio(raw_audio, raw_sr)
fake_ratio = band_ratio(fake, TARGET_HZ)

print(f"real clip, above 4 kHz {real_ratio:.2e}")
print(f"fake before bottleneck {raw_ratio:.2e}")
print(f"fake after bottleneck {fake_ratio:.2e}")

# Both classes must sit in the same order of magnitude. If the raw fake is orders above the
# real, that gap alone is a perfect classifier and every downstream number would measure
# bandwidth rather than code-switching.
print(f"\nraw fake is {raw_ratio/max(real_ratio, 1e-30):.0f}x the real clip, "
      f"{'the bottleneck is load-bearing' if raw_ratio > 10 * real_ratio else 'gap is small, still apply it'}")

# Two-sided on purpose. A fake far below the real is as separable as one far above: either way
# a threshold on stopband energy splits the classes without touching synthesis.
decades = abs(np.log10(max(fake_ratio, 1e-30) / max(real_ratio, 1e-30)))
print(f"after the bottleneck {decades:.1f} decades from the real, "
      + ("matched" if decades < 1 else "STILL SEPARABLE, do not generate yet"))

print("\nconverted, after the bottleneck, this is what a fake will actually look like")
ipd.display(ipd.Audio(FAKE))


# CELL 5: MMS-TTS Hindi, a candidate for the held-out third attack
# Single-speaker, so it is unsuitable as a training attack but fine held out for evaluation.
# The question is whether its tokenizer can represent Latin script at all: MMS vocabularies
# are built per language, and characters outside the vocabulary are silently dropped, which
# would delete the English words from a code-switched sentence.
from transformers import AutoTokenizer, VitsModel

tok = AutoTokenizer.from_pretrained(MMS_MODEL)
mms = VitsModel.from_pretrained(MMS_MODEL).to(DEVICE).eval()

vocab = set(tok.get_vocab())
text = src_row.transcript
missing = sorted({c for c in text.lower() if not c.isspace()} - vocab)
latin_missing = [c for c in missing if c.isascii() and c.isalpha()]

print(f"vocabulary size {len(vocab)}, sampling rate {mms.config.sampling_rate} Hz")
print(f"characters absent from the vocabulary: {''.join(missing) or 'none'}")
if latin_missing:
    print(f"Latin letters dropped: {''.join(latin_missing)}, the English words will not be spoken")

inputs = tok(text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    wav = mms(**inputs).waveform[0].cpu().numpy()

MMS_OUT = f"{WORK}/mms_16k.wav"
sf.write(MMS_OUT, bottleneck(wav, mms.config.sampling_rate), TARGET_HZ, subtype="PCM_16")
print(f"\n{text}")
ipd.display(ipd.Audio(MMS_OUT))


# CELL 6: the decision this gate exists to make
print("""
Attack #2 passes only if all three hold for the bottlenecked conversion:

1. Identity moved. It sounds like the target speaker, not the source. A conversion that
   leaves the source voice intact is not a spoof and the detector would be right to
   call it real.

2. Content survived. The Hindi is intelligible and the English words are still English
   words. FreeVC's content encoder is WavLM, trained overwhelmingly on English, so
   degradation on Devanagari content is the plausible failure here.

3. Switch points intact. The conversion keeps the source prosody, so the switching
   should be audible in the same places. Smearing at the switch points would make the
   fakes easier than the reals for reasons unrelated to synthesis.

If it fails on 2, the fallback is Indic Parler-TTS: transformers-native, covers Hindi and
Bengali, and decodes through DAC rather than HiFi-GAN, so its artifacts are further from
XTTS than a VITS system is. Cost is a heavier install and no Bengali-by-default from the
same pipeline.

Held-out third attack: MMS-TTS is usable only if CELL 5 reports no Latin letters dropped.
Otherwise it silently deletes the English from every code-switched sentence, which would
make the held-out attack a monolingual Hindi test wearing a code-switched label.

Whatever ships, generation applies bottleneck() to every file. Measured band ratios above:
real {real_ratio:.2e}, raw fake {raw_ratio:.2e}, bottlenecked fake {fake_ratio:.2e}.

Licence note for the paper's data section: FreeVC is MIT, its VCTK-trained checkpoint
carries VCTK's CC BY 4.0 attribution requirement, and MMS-TTS is CC-BY-NC 4.0,
non-commercial, which academic use satisfies. Neither generator ships in the demo; the
deployed service runs the detector only.
""")
