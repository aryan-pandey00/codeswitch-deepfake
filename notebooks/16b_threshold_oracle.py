"""Test whether the collapse is a calibration failure that a better threshold would fix.

The metrics carry an apparent contradiction. The frozen XLS-R probe reaches 99.7 AUC and 2.4%
equal error rate on the Hindi-English corpus, which is near-perfect ranking, and at the same
time rejects half of all genuine speech. Both are true: the genuine score distribution slides up
until its median sits on the decision boundary, so the ordering survives and the verdicts do
not.

That invites the obvious objection, and it has to be answered before a reviewer raises it. If
the scores merely shifted, the fix would be a threshold rather than a training set, and the
contribution would be far smaller than claimed.

The comparison is deliberately unfair in the objection's favour. The oracle threshold is fitted
on the entire target corpus with its labels, which no deployment has and which is far more
target information than the eight speakers the fixed head was trained on. If the retrained head
still wins, the objection is closed.

Running it across every architecture also separates two different failures that look identical
in a verdict. A model whose ranking survives can in principle be rescued by a threshold; a model
whose equal error rate has itself collapsed cannot be rescued by any threshold, because the
equal error rate is the floor a threshold can reach.

Runs on the laptop against the CSVs the training notebooks already produced. Seconds, no GPU.

Requirements
    pip install numpy pandas
    paper/results/scores_raw.csv, and any of scores_cnn_bilstm.csv, scores_lora.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "paper" / "results"
OUT = RESULTS / "threshold_oracle.csv"

SOURCES = ["scores_raw.csv", "scores_cnn_bilstm.csv", "scores_lora.csv"]
CORPORA = ("hindi_english", "bengali_english")
SHIPPED_THRESHOLD = 0.0

frames = []
for name in SOURCES:
    path = RESULTS / name
    if not path.exists():
        print(f"skipping {name}, not present")
        continue
    d = pd.read_csv(path)
    cols = ["row", "set", "y"] + [c for c in d.columns if c.startswith("score_")]
    frames.append(d[cols].set_index(["row", "set", "y"]))

D = pd.concat(frames, axis=1).reset_index()
D = D[D.set.isin(CORPORA)]
MODELS = [c[len("score_"):] for c in D.columns if c.startswith("score_")]
print(f"{len(D):,} held-out rows, models: {', '.join(MODELS)}")


def rates(score, y, thr):
    """FRR is genuine speech rejected, FAR is a fake accepted. Percentages."""
    return (100 * float((score[y == 0] > thr).mean()),
            100 * float((score[y == 1] <= thr).mean()))


def eer_threshold(score, y):
    """The threshold where the two error rates meet, fitted on these very labels."""
    cand = np.unique(np.round(score, 3))
    gap = [abs(np.subtract(*rates(score, y, t))) for t in cand]
    return float(cand[int(np.argmin(gap))])


rows = []
for model in MODELS:
    col = f"score_{model}"
    oracle = {}
    for corpus in CORPORA:
        g = D[D.set == corpus].dropna(subset=[col])
        if g.empty:
            continue
        s, y = g[col].values, g.y.values

        frr, far = rates(s, y, SHIPPED_THRESHOLD)
        rows.append(dict(model=model, corpus=corpus, threshold_fitted_on="none, shipped",
                         threshold=SHIPPED_THRESHOLD, frr=frr, far=far,
                         genuine_median=float(np.median(s[y == 0])),
                         fake_median=float(np.median(s[y == 1]))))

        thr = eer_threshold(s, y)
        oracle[corpus] = thr
        frr, far = rates(s, y, thr)
        rows.append(dict(model=model, corpus=corpus,
                         threshold_fitted_on=f"oracle, {corpus}", threshold=thr,
                         frr=frr, far=far,
                         genuine_median=float(np.median(s[y == 0])),
                         fake_median=float(np.median(s[y == 1]))))

    # A threshold is only worth anything if it survives the next corpus. This is where the
    # comparison is decided, rather than on the corpus each was fitted to.
    for src, thr in oracle.items():
        for dst in CORPORA:
            if src == dst or dst not in oracle:
                continue
            g = D[D.set == dst].dropna(subset=[col])
            s, y = g[col].values, g.y.values
            frr, far = rates(s, y, thr)
            rows.append(dict(model=model, corpus=dst, threshold_fitted_on=f"oracle, {src}",
                             threshold=thr, frr=frr, far=far,
                             genuine_median=float(np.median(s[y == 0])),
                             fake_median=float(np.median(s[y == 1]))))

T = pd.DataFrame(rows)
T.to_csv(OUT, index=False)
print(T.round(2).to_string(index=False))

print("\nthe best any threshold can do, against what retraining does\n")
for model in MODELS:
    if model.endswith("fixed"):
        continue
    fixed = model.replace("baseline", "fixed")
    for corpus in CORPORA:
        o = T[(T.model == model) & (T.corpus == corpus)
              & (T.threshold_fitted_on == f"oracle, {corpus}")]
        f = T[(T.model == fixed) & (T.corpus == corpus)
              & (T.threshold_fitted_on == "none, shipped")]
        if o.empty or f.empty:
            continue
        o, f = o.iloc[0], f.iloc[0]
        print(f"{model} on {corpus}: oracle threshold reaches FRR {o.frr:.2f}% at FAR "
              f"{o.far:.2f}%, retrained reaches FRR {f.frr:.2f}% at FAR {f.far:.2f}%")

print("\nSliding a threshold trades one error for the other along a fixed score axis, so the "
      "best it can reach is bounded by the equal error rate. Where that rate has itself "
      "collapsed, no threshold recovers the model and only the representation can.")
