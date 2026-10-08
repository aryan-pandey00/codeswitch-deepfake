"""Merge every architecture into one comparison table.

Four detectors were measured in four notebooks, each writing its own scores. This puts them
under one metric definition so the headline table is computed once rather than assembled by
hand. Nothing is retrained or rescored.

A protocol difference is declared rather than smoothed over: the frozen probe was scored on
whole-utterance pooled features, CNN-BiLSTM and LoRA on a centred five-second span, AASIST on
the first 64,600 samples, which is its released protocol. Held-out utterances average 4.7 s and
notebook 14 measured windowed against whole-clip scoring at 100% verdict agreement.

Requirements
    pip install numpy pandas. Runs locally in seconds against whichever score CSVs exist in
    the results directory. Missing files are skipped with a note."""

# CELL 1: load every score file that exists and merge them
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "paper" / "results"

SOURCES = ["scores_raw.csv", "scores_cnn_bilstm.csv", "scores_lora.csv", "scores_aasist.csv"]

# Display name, baseline column, fixed column, and the scoring protocol each was measured under.
FAMILIES = [
    ("XLS-R + linear probe", "baseline", "fixed", "whole utterance, pooled"),
    ("CNN-BiLSTM", "cnn_bilstm_baseline", "cnn_bilstm_fixed", "centred 5 s span"),
    ("XLS-R + LoRA", "lora_baseline", "lora_fixed", "centred 5 s span"),
    ("AASIST, zero-shot", "aasist_zeroshot", None, "first 64,600 samples, as released"),
    ("AASIST, level-normalised control", "aasist_zeroshot_levelnorm", None,
     "first 64,600 samples, peak normalised"),
]

frames = []
for name in SOURCES:
    path = RESULTS / name
    if not path.exists():
        print(f"skipping {name}, not present")
        continue
    d = pd.read_csv(path)
    keys = ["row", "set", "speaker", "cs_bin", "condition", "y"]
    keep = [c for c in keys if c in d.columns] + [c for c in d.columns if c.startswith("score_")]
    frames.append(d[keep].set_index([c for c in keys if c in d.columns]))

D = pd.concat(frames, axis=1).reset_index()
PRESENT = {c[len("score_"):] for c in D.columns if c.startswith("score_")}
FAMILIES = [f for f in FAMILIES if f[1] in PRESENT]

print(f"{len(D):,} rows, {len(PRESENT)} score columns")
for label, base, fixed, protocol in FAMILIES:
    got = "baseline + fixed" if fixed in PRESENT else "baseline only"
    print(f"{label}: {got}, scored on {protocol}")


# CELL 2: conditions and metrics, defined once for every model
CS = D.cs_bin.fillna("").values
SET = D.set.values

CONDITIONS = [
    ("in-domain dev, unseen speakers", D.condition == "in-domain dev, unseen speakers"),
    ("in-domain control, official test partition",
     D.condition == "in-domain control, official test partition"),
    ("Hindi-English corpus, all", SET == "hindi_english"),
    ("Hindi-English corpus, monolingual utterances only",
     (SET == "hindi_english") & (CS == "mono")),
    ("Hindi-English corpus, code-switched utterances only",
     (SET == "hindi_english") & (CS != "mono")),
    ("Bengali-English corpus, all", SET == "bengali_english"),
    ("Bengali-English corpus, monolingual utterances only",
     (SET == "bengali_english") & (CS == "mono")),
    ("Bengali-English corpus, code-switched utterances only",
     (SET == "bengali_english") & (CS != "mono")),
    ("out of corpus, pooled", np.isin(SET, ["hindi_english", "bengali_english"])),
]
CONDITIONS = [(label, np.asarray(mask)) for label, mask in CONDITIONS]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


# The two curve functions are written out rather than imported, so this file runs on a laptop
# with numpy and pandas alone. The Kaggle notebooks that produced its inputs use scikit-learn.
def roc(y, s):
    order = np.argsort(-np.asarray(s, dtype=float))
    y = np.asarray(y)[order]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tpr = np.concatenate([[0.0], np.cumsum(y == 1) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(y == 0) / n_neg])
    return fpr, tpr


def auc_of(y, s):
    """Rank-based, which is exact under ties and needs no integration."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ranks = pd.Series(s).rank().values
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def eer(y, s):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr = roc(y, s)
    fnr = 1 - tpr
    return 100 * float((fpr + fnr)[int(np.nanargmin(np.abs(fnr - fpr)))] / 2)


def metrics(y, s):
    """Positive class is fake. FAR is a fake accepted, FRR is genuine speech rejected."""
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
        n=len(y), n_genuine=n_real, n_fake=n_fake, tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=100 * (tp + tn) / len(y) if len(y) else nan,
        balanced_accuracy=100 * ((rec + (1 - frr)) / 2) if both else nan,
        precision=100 * prec if prec == prec else nan,
        recall=100 * rec if rec == rec else nan,
        f1=100 * f1 if f1 == f1 else nan,
        far=100 * far if far == far else nan, far_lo=far_lo, far_hi=far_hi,
        frr=100 * frr if frr == frr else nan, frr_lo=frr_lo, frr_hi=frr_hi,
        auc=100 * auc_of(y, s) if both else nan,
        eer=eer(y, s) if both else nan,
    )


rows = []
for label, base, fixed, protocol in FAMILIES:
    for variant, col in (("baseline", base), ("fixed", fixed)):
        if col is None or col not in PRESENT:
            continue
        s_all = D[f"score_{col}"].values
        for cname, mask in CONDITIONS:
            m = mask & ~np.isnan(s_all)
            if not m.sum():
                continue
            rows.append(dict(family=label, variant=variant, column=col, protocol=protocol,
                             condition=cname, **metrics(D.y.values[m], s_all[m])))

A = pd.DataFrame(rows)
A.to_csv(RESULTS / "arch_comparison.csv", index=False)
print(f"\nwrote arch_comparison.csv, {len(A)} rows")


# CELL 3: Table 3, the headline
def fmt(v, dp=1):
    return "-" if v != v else f"{v:.{dp}f}"


OUT_OF_CORPUS = "out of corpus, pooled"
IN_DOMAIN = "in-domain dev, unseen speakers"

print("\nTable 3. Every architecture, in domain and out of it. Positive class is fake; FRR is "
      "genuine speech rejected.\n")
print("| Architecture | Variant | In-domain EER | Out-of-corpus Acc | AUC | EER | FAR | FRR (95% CI) |")
print("|---|---|---|---|---|---|---|---|")
for label, base, fixed, protocol in FAMILIES:
    for variant in ("baseline", "fixed"):
        g = A[(A.family == label) & (A.variant == variant)]
        if g.empty:
            continue
        ind = g[g.condition == IN_DOMAIN]
        out = g[g.condition == OUT_OF_CORPUS]
        if ind.empty or out.empty:
            continue
        ind, out = ind.iloc[0], out.iloc[0]
        print(f"| {label} | {variant} | {fmt(ind.eer, 2)} | {fmt(out.accuracy)} | "
              f"{fmt(out.auc, 2)} | {fmt(out.eer, 2)} | {fmt(out.far, 2)} | "
              f"{fmt(out.frr)} ({fmt(out.frr_lo)}-{fmt(out.frr_hi)}) |")


# CELL 4: what the fix does to each architecture
print("\nTable 4. The same eight speakers per corpus, added to each architecture's training set.\n")
print("| Architecture | FRR before | FRR after | FAR before | FAR after | In-domain EER after |")
print("|---|---|---|---|---|---|")
for label, base, fixed, protocol in FAMILIES:
    if fixed is None or fixed not in PRESENT:
        continue
    b = A[(A.family == label) & (A.variant == "baseline") & (A.condition == OUT_OF_CORPUS)]
    f = A[(A.family == label) & (A.variant == "fixed") & (A.condition == OUT_OF_CORPUS)]
    fi = A[(A.family == label) & (A.variant == "fixed") & (A.condition == IN_DOMAIN)]
    if b.empty or f.empty:
        continue
    b, f, fi = b.iloc[0], f.iloc[0], fi.iloc[0]
    print(f"| {label} | {fmt(b.frr)} | {fmt(f.frr)} | {fmt(b.far, 2)} | {fmt(f.far, 2)} | "
          f"{fmt(fi.eer, 2)} |")


# CELL 5: the ordering, which is the claim
# Three trained points is not a regression and is not presented as one. It is an ordering, and
# the ordering is what the discussion argues from.
print("\nRejection against how much capacity was allowed to fit the training corpus.\n")
CAPACITY = {
    "XLS-R + linear probe": "none, frontend frozen, 2,048 weights above it",
    "CNN-BiLSTM": "504,321 parameters trained from scratch on it",
    "XLS-R + LoRA": "657,409 adapters steering a 265M encoder",
    "AASIST, zero-shot": "none, never trained here at all",
}
# A model with no discriminative signal has to be kept out of this ordering. Its rejection rate
# is low only because it accepts the fakes too, so ranking it beside models that catch every fake
# compares two different things and flatters the broken one.
MIN_AUC = 60.0

ordering, excluded = [], []
for label, base, fixed, protocol in FAMILIES:
    g = A[(A.family == label) & (A.variant == "baseline") & (A.condition == OUT_OF_CORPUS)]
    if g.empty or label not in CAPACITY:
        continue
    g = g.iloc[0]
    (ordering if g.auc >= MIN_AUC else excluded).append((g.frr, label, g.far, g.eer, g.auc))

for frr, label, far, e, auc in sorted(ordering):
    print(f"{fmt(frr)}% rejected, {fmt(far, 2)}% missed, EER {fmt(e, 2)}%, "
          f"- {label}: {CAPACITY[label]}")

for frr, label, far, e, auc in excluded:
    print(f"\n excluded, AUC {fmt(auc, 2)}: {label} separates nothing here. Its {fmt(frr)}% "
          f"rejection sits beside {fmt(far, 2)}% of fakes accepted, which describes a model "
          "without a decision rather than a cautious one.")


# CELL 6: per speaker, because the corpus means hide two different shapes
rows = []
for label, base, fixed, protocol in FAMILIES:
    for variant, col in (("baseline", base), ("fixed", fixed)):
        if col is None or col not in PRESENT:
            continue
        s = D[f"score_{col}"].values
        for corpus in ("hindi_english", "bengali_english"):
            m = (SET == corpus) & (D.y.values == 0) & ~np.isnan(s)
            if not m.sum():
                continue
            df = pd.DataFrame({"speaker": D.speaker.values[m], "s": s[m]})
            for spk, g in df.groupby("speaker"):
                rows.append(dict(family=label, variant=variant, corpus=corpus, speaker=spk,
                                 n=len(g), frr=100 * float((g.s > 0).mean())))

BY_SPK = pd.DataFrame(rows)
BY_SPK.to_csv(RESULTS / "speakers_by_architecture.csv", index=False)

print("\nPer-speaker false rejection on genuine speech, held-out speakers only.\n")
print("| Architecture | Variant | Corpus | Speakers | Median | Above 50% | Below 5% |")
print("|---|---|---|---|---|---|---|")
for (label, variant, corpus), g in BY_SPK.groupby(["family", "variant", "corpus"], sort=False):
    print(f"| {label} | {variant} | {corpus} | {len(g)} | {fmt(g.frr.median())} | "
          f"{int((g.frr > 50).sum())} | {int((g.frr < 5).sum())} |")

print(f"\nwrote speakers_by_architecture.csv, {len(BY_SPK)} rows")
print("\nProtocol, to be stated in the paper rather than buried: the models were scored under "
      "the three protocols listed at the top. Utterances average 4.7 s against a 5 s span, and "
      "notebook 14 measured windowed against whole-clip scoring at 100% verdict agreement on "
      "stitched clips of 19-49 s, so the difference is bounded and does not carry the result.")
