"""Check whether the channel grid's band-ratio gap tracks the label or only the corpus.

Notebook 23 found fakes sitting below reals on energy above 4 kHz in the clean condition, at a
single-scalar AUC of 0.86. That is a property of the frozen dataset rather than of the grid.

It matters for one reason. If genuine speech recorded outside the training corpus also sits low
on that scalar, then "low band ratio looks fake" predicts rejection of genuine out-of-corpus
speech, which is an alternative explanation for the main result. If it does not, the gap is a
discrimination shortcut that cannot touch false rejection.

Reads notebook 23's manifest. No audio, no GPU, seconds.

Requirements
    Kaggle notebook, CPU. Attached input: the output of notebook 23."""

# CELL 1: find the manifest

import glob
import sys

import numpy as np
import pandas as pd

OUT_CSV = "/kaggle/working/channel_confound.csv"

MANIFEST = next(
    (p for depth in ("*", "*/*", "*/*/*", "*/*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/manifest_channels.csv"))),
    None)
if MANIFEST is None:
    sys.exit("manifest_channels.csv not found among the mounted inputs, attach notebook 23's output")

MAN = pd.read_csv(MANIFEST)
ok = MAN[MAN.status == "ok"]
print(f"{MANIFEST}\n{len(ok):,} rows, {ok.channel.nunique()} channels")


# CELL 2: the two AUCs, per channel

def auc(y, s):
    """Rank AUC. Ties get average ranks, which is what the quantisation floor produces here."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    if y.all() or not y.any():
        return float("nan")
    r = pd.Series(s).rank().values
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


rows = []
for channel, g in ok.groupby("channel"):
    real = g[g.label == "real"]
    fake_like = -g.band_ratio.values
    # Fakes score low on this scalar, so the fake-like direction is the negated ratio and the
    # same direction is used for both questions.
    rows.append(dict(
        channel=channel,
        n=len(g),
        auc_label=auc((g.label == "fake").values, fake_like),
        auc_corpus=auc((real.condition_grid != "in-domain control").values,
                       -real.band_ratio.values),
        median_real=real.band_ratio.median(),
        median_fake=g[g.label == "fake"].band_ratio.median(),
        zeros_real=int((real.band_ratio == 0).sum()),
        zeros_fake=int((g[g.label == "fake"].band_ratio == 0).sum()),
    ))

T = pd.DataFrame(rows).sort_values("channel")
T.to_csv(OUT_CSV, index=False)

print("\nauc_label   fakes vs reals, fake-like direction")
print("auc_corpus  genuine out of corpus vs genuine in domain, same direction")
print(T[["channel", "n", "auc_label", "auc_corpus",
         "median_real", "median_fake", "zeros_real", "zeros_fake"]].round(4).to_string(index=False))

c = float(T.loc[T.channel == "clean", "auc_corpus"].iloc[0])
print(f"\nclean auc_corpus {c:.3f}")
if abs(c - 0.5) < 0.15:
    print("The scalar does not order genuine speech by corpus, so it cannot explain rejection.")
else:
    print("The scalar orders genuine speech by corpus. It is a live alternative explanation")
    print("for the false rejection and has to be stated as one.")


# CELL 3: the same cut by condition, since pooling could hide one corpus

print("\nband ratio by condition and label, clean channel")
C = ok[ok.channel == "clean"]
print(C.groupby(["condition_grid", "label"]).band_ratio
      .agg(n="size", median="median", mean="mean", zeros=lambda s: int((s == 0).sum()))
      .to_string())

print("\nper corpus against in-domain genuine, genuine only, clean channel")
base = C[(C.label == "real") & (C.condition_grid == "in-domain control")]
for corpus in ("Hindi-English", "Bengali-English"):
    g = C[(C.label == "real") & (C.condition_grid == corpus)]
    pair = pd.concat([base, g])
    y = (pair.condition_grid == corpus).values
    print(f"  {corpus:16s} AUC {auc(y, -pair.band_ratio.values):.3f}  n={len(g)}")

print(f"\nwrote {OUT_CSV}")
