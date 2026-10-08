"""Measure how much target-corpus data it takes to close the channel gap.

Adding target data always works and is therefore not a contribution, but it is the ceiling the
augmentation result has to be read against, and it costs no GPU because the features are cached.

It answers how far rejection falls when target-corpus speakers enter training at all, how few
speakers are enough, and whether adapting on one code-switched corpus transfers to the other.

Speakers added to training are removed from evaluation throughout, so every number is measured
on speakers the model never heard. Adding a speaker and then scoring them would measure
memorisation.

Requirements
    Kaggle notebook, CPU. Attached input: codeswitch-deepfake-feats."""

# CELL 1: setup
import os
import sys
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

LAYER = 20
SEED = 1337
N_ADDED = [0, 1, 2, 4, 8, 16]
# five draws per point, because which speakers get added is itself a large effect
N_REPEATS = 5

FEAT_DIR = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/feats")) if os.path.isdir(p)),
    None)
if FEAT_DIR is None:
    sys.exit("attach codeswitch-deepfake-feats")

IDX = pd.read_csv(f"{FEAT_DIR}/index.csv")
FEATS = np.load(f"{FEAT_DIR}/pooled_meanstd_fp16.npy")
Y = (IDX.label == "fake").values.astype(int)
X = np.asarray(FEATS[:, LAYER], dtype=np.float32)

BASE_TRAIN = ((IDX.role == "train_source") & (IDX.split == "train")).values
DEV = ((IDX.role == "train_source") & (IDX.split == "dev")).values
CONTROL = (IDX.role == "control_indomain").values

SPK = {c: sorted(IDX[IDX.set == c].speaker.dropna().unique())
       for c in ("hindi_english", "bengali_english")}
print({c: len(v) for c, v in SPK.items()})


def train_eval(train_mask):
    """Fit on a mask, return a scorer and the dev-calibrated gap midpoint.

    The threshold comes from the in-domain dev set every time, because a deployment has no
    labelled target data to calibrate on. Calibrating on the target would answer a question
    nobody gets to ask in practice.
    """
    mu, sd = X[train_mask].mean(0), X[train_mask].std(0) + 1e-6
    # 1000 iterations rather than 3000: the features are standardised and the classes are close
    # to separable, so the extra iterations only sharpen an already-converged boundary, and this
    # routine is called about eighty times.
    clf = LogisticRegression(max_iter=1000).fit((X[train_mask] - mu) / sd, Y[train_mask])
    s = clf.decision_function((X - mu) / sd)
    d = s[DEV]
    thr = float((d[Y[DEV] == 0].max() + d[Y[DEV] == 1].min()) / 2)
    return s, thr


def eer_of(s, mask):
    y = Y[mask]
    if len(set(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s[mask])
    fnr = 1 - tpr
    return 100 * float((fpr + fnr)[int(np.nanargmin(np.abs(fnr - fpr)))] / 2)


def rejected(s, thr, mask):
    v = s[mask & (Y == 0)]
    return 100 * float((v > thr).mean()) if len(v) else np.nan


# CELL 2: the data-efficiency curve
rows = []
for corpus in ("hindi_english", "bengali_english"):
    other = "bengali_english" if corpus == "hindi_english" else "hindi_english"
    for n in N_ADDED:
        for rep in range(N_REPEATS if n else 1):
            rng = np.random.RandomState(SEED + 97 * rep)
            added = list(rng.choice(SPK[corpus], n, replace=False)) if n else []

            train = BASE_TRAIN | ((IDX.set == corpus) & IDX.speaker.isin(added)).values
            # Held out by speaker, not by utterance. An added speaker never appears in any
            # number reported below.
            held = ((IDX.set == corpus) & ~IDX.speaker.isin(added)).values
            held_other = (IDX.set == other).values

            s, thr = train_eval(train)
            rows.append(dict(
                corpus=corpus, n_added=n, rep=rep,
                n_train=int(train.sum()),
                dev_eer=eer_of(s, DEV),
                ctrl_rej=rejected(s, thr, CONTROL),
                held_eer=eer_of(s, held),
                held_rej=rejected(s, thr, held),
                other_eer=eer_of(s, held_other),
                other_rej=rejected(s, thr, held_other)))

CURVE = pd.DataFrame(rows)
agg = (CURVE.groupby(["corpus", "n_added"])
       .agg(held_rej_mean=("held_rej", "mean"), held_rej_std=("held_rej", "std"),
            held_eer_mean=("held_eer", "mean"),
            other_rej_mean=("other_rej", "mean"), other_eer_mean=("other_eer", "mean"),
            ctrl_rej_mean=("ctrl_rej", "mean"), dev_eer_mean=("dev_eer", "mean"))
       .round(2))
print(agg.to_string())
CURVE.to_csv("/kaggle/working/fix_data_curve.csv", index=False)


# CELL 3: read the three questions off the curve
for corpus in ("hindi_english", "bengali_english"):
    g = agg.loc[corpus]
    base = float(g.loc[0, "held_rej_mean"])
    print(f"\n{corpus}: {base:.1f}% of genuine speech rejected with no target data")
    for n in N_ADDED[1:]:
        if n in g.index:
            v = float(g.loc[n, "held_rej_mean"])
            print(f"+{n} speakers -> {v:.1f}% ({base - v:+.1f} points)")

    # Transfer is what separates a general fix from a corpus-specific patch. If adapting on one
    # code-switched corpus also helps the other, the model learned tolerance; if not, it
    # learned that one recording chain.
    o_base = float(g.loc[0, "other_rej_mean"])
    o_best = float(g.loc[max(N_ADDED), "other_rej_mean"])
    print(f"transfer to the other corpus: {o_base:.1f}% -> {o_best:.1f}% "
          f"({o_base - o_best:+.1f} points)")

    # A fix that quietly costs in-domain performance is not free, and the paper has to say so.
    print(f"in-domain cost: dev EER {float(g.loc[max(N_ADDED), 'dev_eer_mean']):.2f}%, "
          f"control rejection {float(g.loc[max(N_ADDED), 'ctrl_rej_mean']):.1f}%")


# CELL 4: pooled adaptation, both corpora at once
# The realistic deployment version: a developer with a little data from several places, rather
# than a lot from one. Held out by speaker in both corpora simultaneously.
rows = []
for n in N_ADDED:
    for rep in range(N_REPEATS if n else 1):
        rng = np.random.RandomState(SEED + 97 * rep)
        added = {c: list(rng.choice(SPK[c], n, replace=False)) if n else [] for c in SPK}
        mask_added = np.zeros(len(IDX), dtype=bool)
        for c, spks in added.items():
            mask_added |= ((IDX.set == c) & IDX.speaker.isin(spks)).values

        train = BASE_TRAIN | mask_added
        s, thr = train_eval(train)
        held = IDX.set.isin(list(SPK)).values & ~mask_added
        rows.append(dict(n_added_per_corpus=n, dev_eer=eer_of(s, DEV),
                         held_eer=eer_of(s, held), held_rej=rejected(s, thr, held),
                         ctrl_rej=rejected(s, thr, CONTROL)))

POOLED = (pd.DataFrame(rows).groupby("n_added_per_corpus").mean().round(2))
print(POOLED.to_string())
# The group key is the index here, so writing with index=False drops the number of speakers
# added and leaves a curve with no x-axis. It is the column Figure 5 is plotted against.
POOLED.reset_index().to_csv("/kaggle/working/fix_data_pooled.csv", index=False)

print("\nceiling for the augmentation experiment to be read against: "
      f"{float(POOLED.held_rej.iloc[-1]):.1f}% rejection with "
      f"{max(N_ADDED)} speakers per corpus, against "
      f"{float(POOLED.held_rej.iloc[0]):.1f}% with none")
