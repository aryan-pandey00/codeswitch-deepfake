"""Train a head on vocoded fakes and ask whether it generalises to generators it never saw.

A head trained on XTTS-v2 and FreeVC returned an unknown generator's clip as genuine at full
margin, and both decode through HiFi-GAN, so what it held was one vocoder family. 21's premise
is that fakes supplied as audio through a resynthesis bottleneck teach the property every
generator shares instead of the fingerprint one of them leaves. Four questions: per-path catch
rates including the held-out one, whether a vocoded-trained head catches the two generators it
never saw, how badly the original head misses vocoded fakes, and whether any of it costs false
rejection of genuine speech.

Recipe copied from 13 so the numbers stack. Three departures forced by the data: the vocoded
set has no dev split, since 21 drew sources from `split=="train"` only, so one is carved by
speaker; classes are 4:1 so `class_weight` is balanced; and the two caches are stacked, which
22 verified is legitimate at a scale ratio of 0.98.

Requirements
    Kaggle notebook, CPU, no internet, a few minutes.
    Attached inputs: the output of notebook 22, and codeswitch-deepfake-feats.
"""

# CELL 1: mounts and the two caches
import os
import sys
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

LAYER = 20
SEED = 1337
WORK = "/kaggle/working"
OUT_CSV = f"{WORK}/vocoded_heads.csv"


def find(name):
    """Locate a mounted folder; mounts are namespaced and sometimes deeper still."""
    for depth in range(1, 7):
        hits = [p for p in sorted(glob.glob("/kaggle/input/" + "*/" * depth + name))
                if os.path.isdir(p)]
        if hits:
            return hits[0]
    return None


OLD_DIR = find("feats")
if OLD_DIR is None:
    sys.exit("attach codeswitch-deepfake-feats")

VOC_DIR = next((os.path.dirname(p) for depth in range(1, 7)
                for p in sorted(glob.glob("/kaggle/input/" + "*/" * depth + "voc_index.csv"))),
               None)
if VOC_DIR is None:
    sys.exit("attach the output of notebook 22")
print(f"old: {OLD_DIR}\nvoc: {VOC_DIR}")

# The old cache stores all 25 hidden states and is 3.5 GB, so only layer 20 is ever realised.
OI = pd.read_csv(f"{OLD_DIR}/index.csv")
XO = np.asarray(np.load(f"{OLD_DIR}/pooled_meanstd_fp16.npy", mmap_mode="r")[:, LAYER],
                dtype=np.float32)
YO = (OI.label == "fake").values.astype(int)

VI = pd.read_csv(f"{VOC_DIR}/voc_index.csv")
XV = np.asarray(np.load(f"{VOC_DIR}/voc_layer20_fp16.npy"), dtype=np.float32)
YV = VI.label.values.astype(int)

assert len(OI) == len(XO) and len(VI) == len(XV), "index and feature rows disagree"
print(f"old {XO.shape}, voc {XV.shape}")
print(VI.groupby(["vocoder", "role"]).size().to_string())

HELD_OUT = sorted(VI[VI.held_out == True].vocoder.unique())  # noqa: E712
assert len(HELD_OUT) == 1, f"expected one held-out path, found {HELD_OUT}"
HELD_OUT = HELD_OUT[0]
print(f"held out: {HELD_OUT}")


# CELL 2: the recipe, identical to 13 so the numbers stack with the rest of the project
def fit_head(X, Y, train_mask, dev_mask):
    """Fit on a mask, return a scorer over any matrix and the dev-calibrated threshold.

    The scorer carries its own standardisation, so scoring the other cache later cannot silently
    use statistics from the set being scored.
    """
    mu, sd = X[train_mask].mean(0), X[train_mask].std(0) + 1e-6
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit((X[train_mask] - mu) / sd, Y[train_mask])

    def score(M):
        return clf.decision_function((M - mu) / sd)

    d = score(X[dev_mask])
    yd = Y[dev_mask]
    # Gap midpoint on dev, falling back to the median if the classes overlap.
    thr = (float((d[yd == 0].max() + d[yd == 1].min()) / 2)
           if d[yd == 0].max() < d[yd == 1].min() else float(np.median(d)))
    return score, thr


def eer_of(s, y):
    if len(set(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    return 100 * float((fpr + fnr)[int(np.nanargmin(np.abs(fnr - fpr)))] / 2)


def caught(s, thr):
    """Percent of fakes scored above threshold. The detection rate."""
    return 100 * float((s > thr).mean()) if len(s) else np.nan


def rejected(s, thr):
    """Percent of genuine clips scored above threshold. False rejection."""
    return 100 * float((s > thr).mean()) if len(s) else np.nan


ROWS = []


def record(**kw):
    ROWS.append(kw)
    print(" " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in kw.items()))


# CELL 3: sweep which path is held out, and score every target from each variant
# Dev is carved by speaker, not by row: a source contributes five rows sharing a speaker.
V_SRC = (VI.role == "train_source").values
spk = sorted(VI.loc[V_SRC, "speaker"].dropna().astype(str).unique())
n_dev = max(1, round(0.2 * len(spk)))
DEV_SPK = set(np.random.default_rng(SEED).choice(spk, n_dev, replace=False).tolist())
print(f"\n{len(spk)} training speakers, {n_dev} held for dev: {sorted(DEV_SPK)}")

in_dev = VI.speaker.astype(str).isin(DEV_SPK).values
V_TEST = ~V_SRC
V_TEST_REAL = V_TEST & (YV == 0)

old_fake = (YO == 1)
old_real = (YO == 0)
ATTACK_COL = next((c for c in ("attack", "system", "generator") if c in OI.columns), None)
PATHS = sorted(VI[VI.label == 1].vocoder.unique())


def run_variant(held):
    """Fit with one path held out of training and of the threshold, then score every target.

    Twins stay in both, since the genuine class is not what is being held out.
    """
    train = V_SRC & ~in_dev & (VI.vocoder != held).values
    dev = V_SRC & in_dev & (VI.vocoder != held).values
    assert dev.sum() and set(YV[dev]) == {0, 1}, f"dev split degenerate for held={held}"

    score, thr = fit_head(XV, YV, train, dev)
    SV = score(XV)
    dev_eer = eer_of(SV[dev], YV[dev])
    print(f"\nheld out: {held}, {train.sum():,} train, dev EER {dev_eer:.2f}%")

    for voc in PATHS:
        m = V_TEST & (VI.vocoder == voc).values
        if not m.sum():
            continue
        record(question="per_path", held_out=held, head="vocoded", target=voc,
               unseen=bool(voc == held), n=int(m.sum()), dev_eer=dev_eer,
               detection=caught(SV[m], thr),
               eer=eer_of(np.concatenate([SV[m], SV[V_TEST_REAL]]),
                          np.concatenate([np.ones(m.sum()), np.zeros(V_TEST_REAL.sum())])))

    # The old cache holds XTTS-v2 and FreeVC. This head has seen no TTS at all.
    SO = score(XO)
    targets = ([(str(a), old_fake & (OI[ATTACK_COL] == a).values)
                for a in sorted(OI.loc[old_fake, ATTACK_COL].dropna().unique())]
               if ATTACK_COL else [("all_tts", old_fake)])
    for name, m in targets:
        record(question="forward", held_out=held, head="vocoded", target=name, unseen=True,
               n=int(m.sum()), dev_eer=dev_eer, detection=caught(SO[m], thr),
               eer=eer_of(np.concatenate([SO[m], SO[old_real]]),
                          np.concatenate([np.ones(m.sum()), np.zeros(old_real.sum())])))

    for name, m in (("in_domain_dev", dev & (YV == 0)), ("out_of_corpus", V_TEST_REAL)):
        record(question="rejection", held_out=held, head="vocoded", target=name, unseen=False,
               n=int(m.sum()), dev_eer=dev_eer, rejection=rejected(SV[m], thr))

    return SV, thr


# The first draw held out lowrank and it was undetectable at EER 52.8%: either unseen attacks are
# hard, or that path teaches nothing. Holding out each in turn tells them apart, for free.
SWEEP = {held: run_variant(held) for held in PATHS}
# The shipped choice, carried into the comparison below.
SV, v_thr = SWEEP[HELD_OUT]

print("\nRead the sweep by column, not by row. A path that is undetectable no matter which "
      "variant is trained is a path that teaches nothing, and should be dropped rather than "
      "reported as a hard unseen attack. Forward transfer that stays low in every variant is "
      "the experiment failing, not the held-out choice failing.")


# CELL 4: reverse transfer, the downloaded-clip anecdote made measurable
# Trained the way this project always has, then shown the vocoded fakes.
O_TRAIN = ((OI.role == "train_source") & (OI.split == "train")).values
O_DEV = ((OI.role == "train_source") & (OI.split == "dev")).values
print(f"\noriginal head: {O_TRAIN.sum():,} train, {O_DEV.sum():,} dev")
o_score, o_thr = fit_head(XO, YO, O_TRAIN, O_DEV)
print(f"dev EER {eer_of(o_score(XO[O_DEV]), YO[O_DEV]):.2f}%, threshold {o_thr:.3f}")

SV_from_o = o_score(XV)
print("\nreverse: original head on the vocoded fakes, none of which it has seen")
for voc in sorted(VI[VI.label == 1].vocoder.unique()):
    m = V_TEST & (VI.vocoder == voc).values
    if m.sum():
        record(question="reverse", head="original", target=voc, unseen=True,
               n=int(m.sum()), detection=caught(SV_from_o[m], o_thr), eer=np.nan)


# CELL 5: false rejection, which nothing here is allowed to regress

print("\nfalse rejection of genuine speech, measured on twins")
for name, m in (("in_domain_dev", O_DEV & (YO == 0)),
                ("out_of_corpus", (OI.role != "train_source").values & (YO == 0))):
    record(question="rejection", head="original", target=name, unseen=False,
           n=int(m.sum()), detection=np.nan, rejection=rejected(o_score(XO)[m], o_thr))

print("\nThe vocoded head is only interesting if this column holds. A detector that catches "
      "every generator by rejecting half of genuine speech is the failure this project is about.")


# CELL 6: the contract
os.makedirs(f"{WORK}/results", exist_ok=True)
R = pd.DataFrame(ROWS)
R.to_csv(OUT_CSV, index=False)
print(f"\n{len(R)} rows -> {OUT_CSV}")

# The sweep only reads as a grid: rows are the variant, columns what it was asked to catch.
for q in ("per_path", "forward"):
    sub = R[R.question == q]
    if len(sub) and "held_out" in sub.columns:
        print(f"\ndetection %, {q}, rows = path held out of training")
        print(sub.pivot_table(index="held_out", columns="target", values="detection")
              .round(1).to_string())

print()
print(R.to_string(index=False))
print("\nCopy to paper/results/vocoded_heads.csv and fill the Results Index row in "
      "paper/FINDINGS.md. A number that is not in that index does not exist.")