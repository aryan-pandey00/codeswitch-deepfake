"""Compute the full metric set for both detectors across every evaluation condition.

Produces accuracy, precision, recall, F1 and AUC once, so every later document quotes the same
numbers rather than recomputing them.

The positive class is fake, so the two error rates are named rather than left to the reader:
FAR is a fake accepted, FRR is genuine speech rejected. Rates carry Wilson intervals, because
several conditions have zero errors out of a few hundred trials and a bare 0.0% invites a
stronger claim than the sample supports.

Requirements
    Kaggle notebook, CPU. Attached input: codeswitch-deepfake-feats. Optionally the exported
    heads.json, which makes the numbers describe the shipped artifact rather than a copy."""

# CELL 1: configuration and the feature cache
import os
import sys
import glob
import json

import numpy as np
import pandas as pd

LAYER = 20
SEED = 1337
N_FIX = 8

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

CORPORA = ("hindi_english", "bengali_english")


def find(name):
    """Kaggle mounts inputs under the notebook slug, not the dataset name."""
    return next(
        (p for depth in ("*", "*/*", "*/*/*")
         for p in sorted(glob.glob(f"/kaggle/input/{depth}/{name}")) if os.path.isdir(p)),
        None)


FEAT_DIR = find("feats")
if FEAT_DIR is None:
    sys.exit("attach codeswitch-deepfake-feats")

IDX = pd.read_csv(f"{FEAT_DIR}/index.csv")
FEATS = np.load(f"{FEAT_DIR}/pooled_meanstd_fp16.npy", mmap_mode="r")
X = np.asarray(FEATS[:, LAYER], dtype=np.float32)
Y = (IDX.label == "fake").values.astype(int)

print(f"{len(IDX):,} rows, layer {LAYER}, dim {X.shape[1]}, {Y.sum():,} fake")


# CELL 2: the two heads, loaded if the export is reachable and retrained if not
from sklearn.linear_model import LogisticRegression

BASE_TRAIN = ((IDX.role == "train_source") & (IDX.split == "train")).values
DEV = ((IDX.role == "train_source") & (IDX.split == "dev")).values
CONTROL = (IDX.role == "control_indomain").values

SPK = {c: sorted(IDX[IDX.set == c].speaker.dropna().unique()) for c in CORPORA}
rng = np.random.RandomState(SEED)
ADDED = {c: sorted(rng.choice(SPK[c], N_FIX, replace=False)) for c in CORPORA}

mask_added = np.zeros(len(IDX), dtype=bool)
for c, spks in ADDED.items():
    mask_added |= ((IDX.set == c) & IDX.speaker.isin(spks)).values

# Speakers used to build the fixed head are excluded from every number below. Without this the
# fixed column would be scored partly on its own training data and the comparison would be void.
HELD = IDX.set.isin(list(CORPORA)).values & ~mask_added

TRAIN = {"baseline": BASE_TRAIN, "fixed": BASE_TRAIN | mask_added}

heads_path = next((p for p in sorted(glob.glob("/kaggle/input/**/heads.json", recursive=True))
                   + ["heads.json", "../models/heads.json"] if os.path.exists(p)), None)

SCORES = {}
if heads_path:
    spec = json.load(open(heads_path))
    for name, h in spec["heads"].items():
        mu, sd = np.array(h["mu"]), np.array(h["sd"])
        SCORES[name] = ((X - mu) / sd) @ np.array(h["w"]) + h["b"]
    source = f"exported weights, {heads_path}"
else:
    from sklearn.metrics import roc_curve
    for name, train in TRAIN.items():
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
        clf = LogisticRegression(max_iter=1000).fit((X[train] - mu) / sd, Y[train])
        raw = clf.decision_function((X - mu) / sd)
        d = raw[DEV]
        lo, hi = d[Y[DEV] == 1].min(), d[Y[DEV] == 0].max()
        if lo > hi:
            thr = float((lo + hi) / 2)
        else:
            fpr, tpr, cut = roc_curve(Y[DEV], d)
            thr = float(cut[int(np.nanargmin(np.abs((1 - tpr) - fpr)))])
        SCORES[name] = raw - thr
    source = f"retrained from the cache, seed {SEED}, {N_FIX} added speakers per corpus"

print(f"heads: {source}")
print("decision rule: score above zero means fake")


# CELL 3: the conditions
# Ordered from the training distribution outward, because that ordering is the result.
CS = IDX.cs_bin.fillna("").values

CONDITIONS = [
    ("in-domain dev, unseen speakers", DEV),
    ("in-domain control, official test partition", CONTROL),
    ("Hindi-English corpus, all", HELD & (IDX.set == "hindi_english").values),
    ("Hindi-English corpus, monolingual utterances only",
     HELD & (IDX.set == "hindi_english").values & (CS == "mono")),
    ("Hindi-English corpus, code-switched utterances only",
     HELD & (IDX.set == "hindi_english").values & (CS != "mono")),
    ("Bengali-English corpus, all", HELD & (IDX.set == "bengali_english").values),
    ("Bengali-English corpus, monolingual utterances only",
     HELD & (IDX.set == "bengali_english").values & (CS == "mono")),
    ("Bengali-English corpus, code-switched utterances only",
     HELD & (IDX.set == "bengali_english").values & (CS != "mono")),
    ("out of corpus, pooled", HELD),
]

for label, mask in CONDITIONS:
    if mask.sum() == 0:
        print(f"empty condition: {label}")


# CELL 4: metrics
from sklearn.metrics import roc_auc_score, roc_curve


def wilson(k, n, z=1.96):
    """95% interval for a rate. Needed because several cells here have zero errors."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def eer(y, s):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    return 100 * float((fpr + fnr)[int(np.nanargmin(np.abs(fnr - fpr)))] / 2)


def metrics(y, s):
    """Positive class is fake. FRR is the error this project is about."""
    pred = (s > 0).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    n_fake, n_real = tp + fn, fp + tn
    nan = float("nan")

    prec = tp / (tp + fp) if tp + fp else nan
    rec = tp / n_fake if n_fake else nan
    f1 = 2 * prec * rec / (prec + rec) if tp + fp and n_fake and prec + rec else nan
    far = fn / n_fake if n_fake else nan
    frr = fp / n_real if n_real else nan

    frr_lo, frr_hi = wilson(fp, n_real)
    far_lo, far_hi = wilson(fn, n_fake)

    both = n_fake > 0 and n_real > 0
    return dict(
        n=len(y), n_genuine=n_real, n_fake=n_fake,
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=100 * (tp + tn) / len(y) if len(y) else nan,
        balanced_accuracy=100 * ((rec + (1 - frr)) / 2) if both else nan,
        precision=100 * prec if prec == prec else nan,
        recall=100 * rec if rec == rec else nan,
        f1=100 * f1 if f1 == f1 else nan,
        far=100 * far if far == far else nan,
        far_lo=far_lo, far_hi=far_hi,
        frr=100 * frr if frr == frr else nan,
        frr_lo=frr_lo, frr_hi=frr_hi,
        auc=100 * roc_auc_score(y, s) if both else nan,
        eer=eer(y, s) if both else nan,
    )


# CELL 5: compute and write
rows = []
for model, score in SCORES.items():
    for label, mask in CONDITIONS:
        if mask.sum() == 0:
            continue
        rows.append(dict(model=model, condition=label, **metrics(Y[mask], score[mask])))

M = pd.DataFrame(rows)
M.to_csv(f"{OUT}/metrics_main.csv", index=False)
M[["model", "condition", "tp", "fp", "fn", "tn"]].to_csv(f"{OUT}/confusion_main.csv", index=False)

# Raw scores let every figure be drawn later from one run, so two documents cannot quote
# different numbers. Only non-overlapping conditions are labelled: "all" is the union of the
# monolingual and code-switched rows and "pooled" the union of the corpora, so labelling those
# would overwrite the finer label. Reconstruct them from set and cs_bin.
LEAF = [(label, mask) for label, mask in CONDITIONS
        if "corpus, all" not in label and "pooled" not in label]

cond_of = np.full(len(IDX), "", dtype=object)
for label, mask in LEAF:
    cond_of[mask] = label

keep = cond_of != ""
RAW = pd.DataFrame({
    "row": np.flatnonzero(keep),
    "set": IDX.set.values[keep],
    "speaker": IDX.speaker.values[keep],
    "cs_bin": CS[keep],
    "condition": cond_of[keep],
    "label": IDX.label.values[keep],
    "y": Y[keep],
})
for model, score in SCORES.items():
    RAW[f"score_{model}"] = score[keep]
RAW.to_csv(f"{OUT}/scores_raw.csv", index=False)

print(f"wrote metrics_main.csv, confusion_main.csv, scores_raw.csv ({len(RAW):,} rows) to {OUT}")


# CELL 6: the tables, formatted to paste
def fmt(v, dp=1):
    return "-" if v != v else f"{v:.{dp}f}"


print("\nTable 2. Detection metrics by condition. Positive class is fake; FRR is genuine "
      "speech rejected.\n")
print("| Model | Condition | n | Acc | Prec | Rec | F1 | AUC | EER | FAR | FRR (95% CI) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in M.iterrows():
    ci = f"{fmt(r.frr)} ({fmt(r.frr_lo)}-{fmt(r.frr_hi)})"
    print(f"| {r.model} | {r.condition} | {r.n:,} | {fmt(r.accuracy)} | {fmt(r.precision)} | "
          f"{fmt(r.recall)} | {fmt(r.f1)} | {fmt(r.auc, 2)} | {fmt(r.eer, 2)} | "
          f"{fmt(r.far, 2)} | {ci} |")

print("\nThe shape of the result, stated as the paper will state it:\n")
for model in SCORES:
    g = M[M.model == model].set_index("condition")
    dev = g.loc["in-domain dev, unseen speakers"]
    pooled = g.loc["out of corpus, pooled"]
    print(f"{model}: in-domain EER {fmt(dev.eer, 2)}%, out of corpus it misses "
          f"{fmt(pooled.far, 2)}% of fakes and rejects {fmt(pooled.frr)}% of genuine speech")


# CELL 7: per speaker, because the failure is not uniform
rows = []
for model, score in SCORES.items():
    for corpus in CORPORA:
        m = HELD & (IDX.set == corpus).values & (Y == 0)
        if m.sum() == 0:
            continue
        df = pd.DataFrame({"speaker": IDX.speaker.values[m], "s": score[m]})
        for spk, g in df.groupby("speaker"):
            rows.append(dict(model=model, corpus=corpus, speaker=spk, n=len(g),
                             frr=100 * float((g.s > 0).mean()),
                             median_score=float(g.s.median())))

BY_SPK = pd.DataFrame(rows)
BY_SPK.to_csv(f"{OUT}/metrics_by_speaker.csv", index=False)

print("\nper-speaker false rejection on genuine speech, held-out speakers only\n")
for model in SCORES:
    for corpus in CORPORA:
        g = BY_SPK[(BY_SPK.model == model) & (BY_SPK.corpus == corpus)]
        if not len(g):
            continue
        print(f"{model} {corpus}: {len(g)} speakers, median {g.frr.median():.1f}%, "
              f"{int((g.frr > 50).sum())} above 50%, {int((g.frr < 5).sum())} below 5%")
