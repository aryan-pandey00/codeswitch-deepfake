"""Attribute the Hinglish degradation to a cause before acting on it.

Notebook 10 measured a Hindi-trained detector going from 0% EER on monolingual Hindi to 6.69% on
Hindi-English, rejecting 32% of genuine code-switched speech. The magnitude is not in question,
the cause is.

The control was already in the dataset. Both code-switched corpora contain genuine monolingual
utterances, `cs_bin == "mono"`, from the same speakers in the same sessions, differing in
recording conditions but not in language mixing. A flat profile down to `mono` leaves nothing
for switching to explain. Everything is computed at three layers, since a conclusion holding at
one depth and not another is a fact about depth.

Requirements
    Kaggle notebook, CPU. Attached input: codeswitch-deepfake-feats."""

# CELL 1: baseline, refit per layer
import os
import sys
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

# filterbank, the tie-break pick, the best transfer
LAYERS = [0, 7, 20]

FEAT_DIR = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/feats")) if os.path.isdir(p)),
    None)
if FEAT_DIR is None:
    sys.exit("attach codeswitch-deepfake-feats")

IDX = pd.read_csv(f"{FEAT_DIR}/index.csv")
FEATS = np.load(f"{FEAT_DIR}/pooled_meanstd_fp16.npy")
Y = (IDX.label == "fake").values.astype(int)

TRAIN = ((IDX.role == "train_source") & (IDX.split == "train")).values
DEV = ((IDX.role == "train_source") & (IDX.split == "dev")).values
CONTROL = (IDX.role == "control_indomain").values


def scores_for(layer):
    X = np.asarray(FEATS[:, layer], dtype=np.float32)
    mu, sd = X[TRAIN].mean(0), X[TRAIN].std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000).fit((X[TRAIN] - mu) / sd, Y[TRAIN])
    return clf.decision_function((X - mu) / sd)


SCORES = {L: scores_for(L) for L in LAYERS}


def eer_of(score, mask):
    y, s = Y[mask], score[mask]
    if len(set(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    return 100 * float((fpr + fnr)[int(np.nanargmin(np.abs(fnr - fpr)))] / 2)


def operating_points(score):
    """Dev has no errors of either kind, so every threshold inside the gap is equally
    supported by the data. The edge of the real distribution is the least robust choice in
    that interval, and it is what the EER point returns."""
    s = score[DEV]
    real, fake = s[Y[DEV] == 0], s[Y[DEV] == 1]
    fpr, tpr, thr = roc_curve(Y[DEV], s)
    return {
        "eer_point": float(thr[int(np.nanargmin(np.abs((1 - tpr) - fpr)))]),
        "gap_mid": float((real.max() + fake.min()) / 2),
        "fa1pct": float(np.quantile(real, 0.99)),
    }


POINTS = {L: operating_points(SCORES[L]) for L in LAYERS}
for L in LAYERS:
    s = SCORES[L][DEV]
    real, fake = s[Y[DEV] == 0], s[Y[DEV] == 1]
    print(f"layer {L}: dev real max {real.max():.2f}, dev fake min {fake.min():.2f}, "
          f"gap {fake.min()-real.max():.2f}, thresholds "
          + ", ".join(f"{k} {v:.2f}" for k, v in POINTS[L].items()))


def rejected(score, thr, mask):
    s = score[mask & (Y == 0)]
    return 100 * float((s > thr).mean()) if len(s) else np.nan


# CELL 2: the control: same corpus, same speakers, no switching
ORDER = ["mono", "cs_low", "cs_mid", "cs_high", "cs_very_high"]

for corpus in ("hindi_english", "bengali_english"):
    rows = []
    for b in ORDER:
        m = ((IDX.set == corpus) & (IDX.cs_bin == b)).values
        if m.sum() == 0:
            continue
        real = m & (Y == 0)
        r = dict(bin=b, n_real=int(real.sum()),
                 med_eng=round(float(IDX[real].eng_ratio.median()), 2),
                 med_cmi=round(float(IDX[real].cmi.median()), 1))
        for L in LAYERS:
            r[f"eer_L{L}"] = round(eer_of(SCORES[L], m), 2)
            r[f"rej_L{L}"] = round(rejected(SCORES[L], POINTS[L]["gap_mid"], m), 1)
        rows.append(r)
    print(f"\n{corpus}, genuine speech by intensity, monolingual bin included")
    print(pd.DataFrame(rows).to_string(index=False))

print("\nreference conditions (rejection at the dev gap midpoint)")
for name, m in [("dev, mono Hindi, unseen speakers", DEV),
                ("control, mono Hindi test partition", CONTROL)]:
    cells = ", ".join(f"L{L} eer {eer_of(SCORES[L], m):.2f} "
                      f"rej {rejected(SCORES[L], POINTS[L]['gap_mid'], m):.1f}%" for L in LAYERS)
    print(f"{name}: {cells}")


# CELL 3: the comparison the paper would have to make
summary = []
for name, m in [
    ("mono Hindi, train corpus, unseen speakers", DEV),
    ("mono Hindi, official test partition", CONTROL),
    ("mono Hindi, inside the Hinglish corpus",
     ((IDX.set == "hindi_english") & (IDX.cs_bin == "mono")).values),
    ("code-switched Hindi-English",
     ((IDX.set == "hindi_english") & (IDX.cs_bin != "mono")).values),
    ("mono Bengali, inside the Bengali corpus",
     ((IDX.set == "bengali_english") & (IDX.cs_bin == "mono")).values),
    ("code-switched Bengali-English",
     ((IDX.set == "bengali_english") & (IDX.cs_bin != "mono")).values),
]:
    if m.sum() == 0:
        continue
    r = dict(condition=name, n_real=int((m & (Y == 0)).sum()))
    for L in LAYERS:
        r[f"eer_L{L}"] = round(eer_of(SCORES[L], m), 2)
        r[f"rej_L{L}"] = round(rejected(SCORES[L], POINTS[L]["gap_mid"], m), 1)
    summary.append(r)

SUM = pd.DataFrame(summary)
print(SUM.to_string(index=False))
SUM.to_csv("/kaggle/working/collapse_decomposition.csv", index=False)

for L in LAYERS:
    a = SUM[SUM.condition == "mono Hindi, inside the Hinglish corpus"][f"rej_L{L}"]
    b = SUM[SUM.condition == "code-switched Hindi-English"][f"rej_L{L}"]
    if len(a) and len(b):
        d = float(b.iloc[0]) - float(a.iloc[0])
        print(f"layer {L}: same corpus without switching {float(a.iloc[0]):.1f}%, "
              f"with switching {float(b.iloc[0]):.1f}%, switching adds {d:+.1f} points")


# CELL 4: corpus effect, or a few recording sessions?
for corpus in ("hindi_english", "bengali_english"):
    real = ((IDX.set == corpus) & (Y == 0)).values
    sub = IDX[real].copy()
    for L in LAYERS:
        sub[f"L{L}"] = SCORES[L][real] > POINTS[L]["gap_mid"]
    by_spk = (sub.groupby("speaker")
              .agg(n=("label", "size"), **{f"rej_L{L}": (f"L{L}", lambda s: 100 * s.mean())
                                           for L in LAYERS})
              .sort_values(f"rej_L{LAYERS[-1]}", ascending=False).round(1))
    print(f"\n{corpus}: rejection by speaker, {len(by_spk)} speakers")
    print(by_spk.to_string())
    for L in LAYERS:
        share = by_spk[f"rej_L{L}"]
        print(f"layer {L}: above 50% rejected {int((share > 50).sum())}/{len(share)} speakers, "
              f"below 5% {int((share < 5).sum())}")


# CELL 5: does switching matter once the corpus is held constant?
# Within one corpus only, so recording chain and speaker pool are constant by construction and
# switch rate is the thing that varies. Genuine utterances only: the claim is about real speech
# being rejected, not about fakes being missed.
import statsmodels.formula.api as smf

for corpus in ("hindi_english", "bengali_english"):
    d = IDX[((IDX.set == corpus) & (Y == 0)).values].copy()
    d["switch_rate"] = d.switches / d.n_words.clip(lower=1)
    for L in LAYERS:
        d["score"] = SCORES[L][((IDX.set == corpus) & (Y == 0)).values]
        fit = smf.ols("score ~ switch_rate + eng_ratio + n_words", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.speaker})
        print(f"\n{corpus}, layer {L}: switch_rate {fit.params['switch_rate']:+.3f} "
              f"(p={fit.pvalues['switch_rate']:.3f}), "
              f"eng_ratio {fit.params['eng_ratio']:+.3f} (p={fit.pvalues['eng_ratio']:.3f}), "
              f"n={len(d)}")
