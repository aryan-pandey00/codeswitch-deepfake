"""Decide whether XTTS kept the English in the code-switched transcripts.

XTTS fakes whose transcript contains a short Latin token run about 15% shorter than the real
file they pair with, while the rest run 10% longer. Either XTTS is dropping English words in
`hi` mode, in which case those fakes are monolingual Hindi wearing a code-switched label, or
utterances with short English words are simply shorter to begin with.

The two are separable before anyone listens. If English is being dropped the shortfall should
deepen as the Latin-script share rises; if the grouping is at fault it should be flat. The
audition cells then confirm by ear, which is the only way to tell a dropped word from a rushed
one. FreeVC runs through the same measurement as a reference, since it cannot alter content.

Requirements
    Kaggle notebook, CPU. Attached inputs: the real dataset and the generation output.
"""

# CELL 1: locate the two inputs and load the fake manifest
import os
import glob

import numpy as np
import pandas as pd
import soundfile as sf
import IPython.display as ipd

KAGGLE_USER = "aryankumarpandey"
ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"

# Same resolution as the generation notebook: the mount path carries the notebook slug, which
# is not worth typing by hand.
FAKES = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/fakes")) if os.path.isdir(p)),
    None)
if FAKES is None:
    raise SystemExit("generation output not attached, + Add Input, Your Work, Notebook Output")

FAKE_ROOT = os.path.dirname(FAKES)
FK = pd.read_csv(f"{FAKE_ROOT}/manifest_fake.csv")
print(f"{len(FK):,} rendered fakes under {FAKES}")


def play(path):
    """Decode rather than embed the file, so the browser never has to handle FLAC itself."""
    x, sr = sf.read(path, dtype="float32")
    return ipd.Audio(x, rate=sr)


# CELL 2: does the shortfall track the amount of English?
def latin_share(text):
    words = str(text).split()
    if not words:
        return 0.0
    return sum(w.isascii() and any(c.isalpha() for c in w) for w in words) / len(words)


FK["latin"] = FK.transcript.fillna("").apply(latin_share)
FK["ratio"] = FK.dur_fake / FK.dur

cs = FK[FK.set == "hindi_english"].copy()
cs["band"] = pd.cut(cs.latin, [-0.01, 0.0, 0.1, 0.2, 0.35, 1.0])

print(cs.groupby(["attack", "band"], observed=True)
        .agg(n=("ratio", "size"), median_ratio=("ratio", "median"))
        .round(3).to_string())

# A system that speaks the English shows a flat column down the table. One that drops it shows
# the ratio falling as the Latin share climbs.
print()
for atk, g in cs.groupby("attack"):
    none = g[g.latin == 0].ratio.median()
    lots = g[g.latin >= 0.2].ratio.median()
    print(f"{atk}: {none:.2f} with no English, {lots:.2f} at 20% or more, "
          f"difference {none - lots:+.2f}")


# CELL 3: the worst XTTS cases, each against the recording it was cloned from
N_CLIPS = 6

worst = (cs[(cs.attack == "xtts") & (cs.latin > 0)]
         .sort_values("ratio").head(N_CLIPS))

for r in worst.itertuples():
    print(f"\nspeaker {r.speaker}, real {r.dur:.1f}s, fake {r.dur_fake:.1f}s, "
          f"ratio {r.ratio:.2f}, English {r.latin:.0%}")
    print(r.transcript)
    print("real")
    ipd.display(play(f"{ROOT}/{r.path}"))
    print("fake")
    ipd.display(play(f"{FAKES}/{r.fake_fname}"))


# CELL 4: two references for comparison
# XTTS on a transcript with no English at all. If these sound fine, whatever is happening above
# is specific to the English and not a general problem with the model or the reference clips.
mono = (cs[(cs.attack == "xtts") & (cs.latin == 0)]
        .sort_values("dur", ascending=False).head(2))

for r in mono.itertuples():
    print(f"\nxtts, no English, ratio {r.ratio:.2f}")
    print(r.transcript)
    ipd.display(play(f"{FAKES}/{r.fake_fname}"))

# FreeVC on heavily code-switched transcripts. Content comes from the source recording, so the
# English is necessarily present; this is the standard the XTTS fakes are being judged against.
vc = (cs[(cs.attack == "freevc") & (cs.latin >= 0.2)]
      .sort_values("dur", ascending=False).head(2))

for r in vc.itertuples():
    print(f"\nfreevc, English {r.latin:.0%}, ratio {r.ratio:.2f}")
    print(r.transcript)
    ipd.display(play(f"{FAKES}/{r.fake_fname}"))


# CELL 5: what the answer decides
print("""
Listen to the CELL 3 pairs against their transcripts and settle three questions.

1. Are the English words there at all? If they are missing from the audio, the XTTS half of
   the Hinglish condition is mislabelled: it is monolingual Hindi being scored as
   code-switched, and every Hinglish number computed over it would be measuring the wrong
   thing. Those files come out and the condition runs on FreeVC alone.

2. If present, are they spoken as English or forced through Devanagari phonology? Accented
   but intact is acceptable and arguably realistic, since that is how the speakers say them.
   Reduced to an unrecognisable syllable is not.

3. Does the audio stop early? Generation warned that transcripts past 150 characters may be
   truncated for `hi`. Truncation would produce the same short durations as dropped English
   without being about English at all, and it is fixed by splitting long transcripts rather
   than by abandoning the attack.

CELL 2 should already point at one of these. A ratio that falls steadily as the English share
rises is question 1. A ratio that is flat across the bands, with the shortfall concentrated in
the longest transcripts instead, is question 3.
""")
