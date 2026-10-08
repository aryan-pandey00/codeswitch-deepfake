"""Count the dataset composition from the two frozen manifests.

Notebook 02 printed these counts for the real half and notebook 07 for the fakes; both
printouts died with their sessions. This reads the manifests and counts. Nothing is generated
or re-derived, no audio is touched.

It carries the three properties the design rests on rather than sizes alone, since this is
where a reader checks the design:

    the train/dev split is speaker-disjoint, asserted rather than described
    every fake is paired to a real utterance that is in the manifest
    the code-switching bins are populated well enough to be read as bins

Long format, with a `section` column marking which question each block answers.

Requirements
    Kaggle notebook, CPU.
    Attached inputs: codeswitch-deepfake-real and codeswitch-deepfake-fakes.
"""

# CELL 1: mounts

import os
import sys
import glob

import pandas as pd

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
OUT_CSV = "/kaggle/working/dataset_counts.csv"

# The fakes mount path depends on the notebook slug rather than the dataset name, so it is
# found by looking for the directory the generator wrote. Same rule as notebooks 07 and 09.
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


# CELL 2: the two manifests, and the design properties behind them

real = pd.read_csv(f"{REAL_ROOT}/manifest_real.csv")
fake = pd.read_csv(f"{FAKE_ROOT}/manifest_fake.csv")
print(f"{len(real):,} real rows, {len(fake):,} fake rows")

tr = set(real[(real.role == "train_source") & (real.split == "train")].speaker)
dv = set(real[(real.role == "train_source") & (real.split == "dev")].speaker)
assert not (tr & dv), "train/dev speaker leak"

# The fake manifest inherits `fname` from the real row it was generated from, so this is the
# pairing check: a fake that references an utterance outside the manifest would mean the two
# halves describe different datasets.
assert set(fake.fname) <= set(real.fname), "a fake references a real utterance not in the manifest"
print(f"speaker-disjoint train {len(tr)} / dev {len(dv)}, overlap 0")
print(f"pairing: {fake.fname.nunique():,} distinct reals carry a fake")


# CELL 3: the table

rows = []


def block(section, df, axis, spk_col="speaker", dur_col="dur", extra=None):
    """One block of the table. Hours come from the manifest's own durations, so they describe
    the audio as stored after the 8 kHz bottleneck, not the source corpora.

    `extra` adds columns for one section only, rather than overloading an existing column with
    a second unit."""
    for level, g in df.groupby(axis, dropna=False):
        row = dict(
            section=section,
            level=str(level),
            n_utterances=len(g),
            n_speakers=g[spk_col].nunique(),
            hours=round(g[dur_col].sum() / 3600, 2),
            secs_median=round(g[dur_col].median(), 2),
        )
        if extra:
            row.update(extra(g))
        rows.append(row)


# `role` and `split` together are the experimental design: one corpus trains, three test.
real["role_split"] = real.role + " / " + real.split.astype(str)
block("real: role and split", real, "role_split")

# Native sample rate is the reason for the bottleneck: monolingual sets arrive at 8 kHz and
# code-switched at 16 kHz, a split falling exactly on the train/test boundary.
block("real: corpus", real, "set", extra=lambda g: dict(native_sr=int(g.src_sr.median())))

# A bin with too few utterances cannot carry a per-bin number, so the counts belong here.
block("real: code-switching bin", real, "cs_bin")
block("real: utterance condition", real, "condition")

fake["dur_out"] = fake.dur_fake
block("fake: attack system", fake, "attack", spk_col="voiced_by", dur_col="dur_out")

rows.append(dict(section="total", level="real", n_utterances=len(real),
                 n_speakers=real.speaker.nunique(), hours=round(real.dur.sum() / 3600, 2),
                 secs_median=round(real.dur.median(), 2)))
rows.append(dict(section="total", level="fake", n_utterances=len(fake),
                 n_speakers=fake.voiced_by.nunique(), hours=round(fake.dur_fake.sum() / 3600, 2),
                 secs_median=round(fake.dur_fake.median(), 2)))

table = pd.DataFrame(rows)
table.to_csv(OUT_CSV, index=False)
print(table.to_string(index=False))
print(f"\nwrote {OUT_CSV}, {len(table)} rows")
