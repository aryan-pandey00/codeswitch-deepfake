"""Train the monolingual Hindi baseline and run the code-switching go/no-go.

The question is whether a detector trained only on monolingual Hindi transfers to Hinglish. The
answer is the gap between EER on held-out monolingual Hindi and EER on code-switched speech from
unseen speakers, plus the false-positive rate on genuine code-switched speech at a threshold
calibrated the way a deployment would calibrate it.

A control runs first, because the in-domain baseline reached 0% EER. Hand-computed scalars
(band ratio, level, duration, spectral shape) are fitted to the same task; if those alone
separate the classes, the classifier is reading the channel.

Requirements
    Kaggle notebook, CPU. Attached inputs: codeswitch-deepfake-feats, plus the real and fake
    audio datasets for the control in cell 3."""

# CELL 1: load the cache
import os
import sys
import glob

import numpy as np
import pandas as pd

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"
SEED = 1337
RUN_AUDIO_CONTROL = True

FEAT_DIR = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/feats")) if os.path.isdir(p)),
    None)
FAKE_DIR = next(
    (p for depth in ("*", "*/*", "*/*/*")
     for p in sorted(glob.glob(f"/kaggle/input/{depth}/fakes")) if os.path.isdir(p)),
    None)
if FEAT_DIR is None:
    sys.exit("no `feats` directory among the mounted inputs, attach codeswitch-deepfake-feats")

IDX = pd.read_csv(f"{FEAT_DIR}/index.csv")
# Held in RAM rather than memory-mapped: the layer sweep slices the same array 25 times along
# an axis that is strided on disk, and 3 GiB is well inside a CPU session.
FEATS = np.load(f"{FEAT_DIR}/pooled_meanstd_fp16.npy")
N, N_LAYERS, DIM = FEATS.shape

# The stored paths were absolute at extraction time. Rebuilding them from the basename keeps
# this notebook working if the datasets mount elsewhere.
base = IDX.file.map(os.path.basename)
IDX["file"] = np.where(IDX.label == "real", f"{REAL_ROOT}/audio/" + base,
                       f"{FAKE_DIR}/" + base if FAKE_DIR else "")

assert len(IDX) == N
print(f"{N:,} utterances, {N_LAYERS} layers, {DIM} dims")

Y = (IDX.label == "fake").values.astype(int)
TRAIN = ((IDX.role == "train_source") & (IDX.split == "train")).values
DEV = ((IDX.role == "train_source") & (IDX.split == "dev")).values
CONTROL = (IDX.role == "control_indomain").values
HINGLISH = (IDX.set == "hindi_english").values
BENGALI = (IDX.set == "bengali_english").values

for name, m in [("train", TRAIN), ("dev", DEV), ("control in-domain", CONTROL),
                ("hindi_english", HINGLISH), ("bengali_english", BENGALI)]:
    print(f"{name}: {int(m.sum()):,} utts, {int(Y[m].sum()):,} fake, "
          f"{IDX[m].speaker.nunique()} speakers")

# Training speakers appearing anywhere in an evaluation set would make every number below an
# in-domain number wearing an out-of-domain label.
train_spk = set(IDX[TRAIN].speaker.dropna())
for name, m in [("dev", DEV), ("control", CONTROL), ("hinglish", HINGLISH), ("bengali", BENGALI)]:
    assert not (train_spk & set(IDX[m].speaker.dropna())), f"train speakers leak into {name}"
print("speaker-disjoint against every evaluation set")


# CELL 2: metrics and the fitting routine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, log_loss


def eer_and_threshold(y, score):
    """EER, and the score at which false accepts and false rejects are equal."""
    fpr, tpr, thr = roc_curve(y, score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return (fpr[i] + fnr[i]) / 2, float(thr[i])


def false_alarm(y, score, threshold):
    """Share of genuine utterances scored above the threshold, i.e. called fake.

    This is the number the demo lives or dies by, and it is not symmetric with EER: a model can
    hold a respectable EER while rejecting most real speech, if the fakes move with it.
    """
    real = y == 0
    return float((score[real] > threshold).mean())


def fit(layer, C=1.0, train_mask=TRAIN):
    """Standardise on train, fit logistic regression, return a scorer over any mask."""
    X = np.asarray(FEATS[:, layer], dtype=np.float32)
    mu = X[train_mask].mean(0)
    sd = X[train_mask].std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=C).fit((X[train_mask] - mu) / sd, Y[train_mask])

    def score(mask):
        return clf.decision_function((X[mask] - mu) / sd)

    return clf, score


# Dev EER saturates at zero, so it cannot rank layers on its own. Log-loss still moves, and a
# layer that separates with more margin is the one to carry forward.
rows = []
for layer in range(N_LAYERS):
    _, score = fit(layer)
    s_dev = score(DEV)
    e, _ = eer_and_threshold(Y[DEV], s_dev)
    ll = log_loss(Y[DEV], 1 / (1 + np.exp(-s_dev)), labels=[0, 1])
    s_cs = score(HINGLISH)
    e_cs, _ = eer_and_threshold(Y[HINGLISH], s_cs)
    rows.append(dict(layer=layer, dev_eer=100 * e, dev_logloss=ll, hinglish_eer=100 * e_cs))

SWEEP = pd.DataFrame(rows)
print(SWEEP.round(4).to_string(index=False))

BEST = int(SWEEP.sort_values(["dev_eer", "dev_logloss"]).layer.iloc[0])
print(f"\nselected layer {BEST} on dev, before looking at any code-switched number")

# Whether the cue is low-level or linguistic shows up in the shape of this curve. Layer 0 is
# the convolutional output, roughly a learned filterbank; if it separates as well as the
# transformer layers, the model is not using anything the later layers add.
print(f"layer 0 dev EER {SWEEP.dev_eer.iloc[0]:.2f}%, "
      f"layer {BEST} dev EER {SWEEP.dev_eer.iloc[BEST]:.2f}%, "
      f"final layer dev EER {SWEEP.dev_eer.iloc[-1]:.2f}%")


# CELL 3: the control: can hand-computed channel statistics do the same job?
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor


def channel_stats(path):
    """Scalars a detector could exploit without knowing anything about synthesis."""
    x, sr = sf.read(path, dtype="float32")
    n = min(len(x), sr * 4)
    x = x[:n]
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    hi = spec[freqs >= 4000].sum()
    lo = spec[freqs < 4000].sum() + 1e-12
    centroid = float((freqs * spec).sum() / (spec.sum() + 1e-12))
    flatness = float(np.exp(np.mean(np.log(spec + 1e-20))) / (spec.mean() + 1e-20))
    return [np.log10(hi / lo + 1e-20), np.log10(np.sqrt((x ** 2).mean()) + 1e-12),
            len(x) / sr, centroid, flatness]


if RUN_AUDIO_CONTROL and FAKE_DIR:
    want = TRAIN | DEV | HINGLISH
    paths = IDX.file.values[want]
    with ThreadPoolExecutor(max_workers=8) as ex:
        S = np.array(list(ex.map(channel_stats, paths)), dtype=np.float32)

    full = np.zeros((N, S.shape[1]), dtype=np.float32)
    full[want] = S
    mu, sd = full[TRAIN].mean(0), full[TRAIN].std(0) + 1e-6
    ctrl = LogisticRegression(max_iter=3000).fit((full[TRAIN] - mu) / sd, Y[TRAIN])

    e_dev, _ = eer_and_threshold(Y[DEV], ctrl.decision_function((full[DEV] - mu) / sd))
    e_cs, _ = eer_and_threshold(Y[HINGLISH], ctrl.decision_function((full[HINGLISH] - mu) / sd))
    print(f"\nchannel-statistics control: dev EER {100*e_dev:.2f}%, "
          f"hinglish EER {100*e_cs:.2f}%")
    names = ["log band ratio >4k", "log rms", "duration", "centroid", "flatness"]
    print(pd.Series(ctrl.coef_[0], index=names).round(3).to_string())

    # The control is expected to carry *some* signal; generators do leave level and spectral
    # traces. It becomes a confound only if it approaches the frontend's own separation, in
    # which case the frontend has no work left to do and the result is about the channel.
    verdict = ("CONFOUND. The channel alone separates the classes, treat every number below "
               "as bandwidth" if 100 * e_dev < 5 else
               "the frontend is doing work the channel statistics cannot")
    print(verdict)
else:
    print("audio control skipped")


# CELL 4: the go/no-go
clf, score = fit(BEST)

s_dev = score(DEV)
dev_eer, THRESH = eer_and_threshold(Y[DEV], s_dev)

results = []
for name, m in [("dev (mono Hindi, unseen speakers)", DEV),
                ("control (mono Hindi test partition)", CONTROL),
                ("hindi_english (code-switched)", HINGLISH),
                ("bengali_english (code-switched)", BENGALI)]:
    s = score(m)
    e, _ = eer_and_threshold(Y[m], s)
    results.append(dict(condition=name, n=int(m.sum()), eer=100 * e,
                        false_alarm=100 * false_alarm(Y[m], s, THRESH),
                        miss=100 * float((s[Y[m] == 1] <= THRESH).mean())))

RES = pd.DataFrame(results)
print(f"threshold calibrated on dev at its EER point: {THRESH:.3f}")
print(RES.round(2).to_string(index=False))

mono = RES.eer.iloc[0]
cs = RES.eer.iloc[2]
fa = RES.false_alarm.iloc[2]
print(f"\nmonolingual {mono:.2f}% EER -> code-switched {cs:.2f}% EER")
print(f"genuine code-switched speech called fake: {fa:.1f}%")
print("GO. The collapse is real" if cs > 5 * max(mono, 0.5) or fa > 10 else
      "NO-GO, Hindi training transfers; pivot to the channel axis")


# CELL 5: does the failure track switching, or just English?
# The settled primary analysis: switch rate is the linguistic measure of code-switching, and
# English ratio and length are what it has to be separated from. Genuine utterances only,
# because the claim is about real speech being rejected.
import statsmodels.formula.api as smf

cs_real = IDX[HINGLISH & (Y == 0)].copy()
cs_real["score"] = score(HINGLISH & (Y == 0))
cs_real["switch_rate"] = cs_real.switches / cs_real.n_words.clip(lower=1)

model = smf.ols("score ~ switch_rate + eng_ratio + n_words", data=cs_real).fit(
    cov_type="cluster", cov_kwds={"groups": cs_real.speaker})
print(model.summary().tables[1])

# The figure version of the same thing. CMI peaks near balance while eng_ratio keeps climbing,
# so the two axes separate at the top bin and the bins are worth reporting against both.
bins = (cs_real.groupby("cs_bin")
        .agg(n=("score", "size"), median_score=("score", "median"),
             called_fake=("score", lambda s: 100 * (s > THRESH).mean()),
             med_eng=("eng_ratio", "median"), med_cmi=("cmi", "median"))
        .reindex(["cs_low", "cs_mid", "cs_high", "cs_very_high"]).dropna(how="all"))
print("\ngenuine code-switched speech by intensity bin")
print(bins.round(2).to_string())


# CELL 6: per-attack breakdown, and the summary to record
for name, m in [("dev", DEV), ("hindi_english", HINGLISH)]:
    sub = IDX[m].copy()
    sub["score"] = score(m)
    sub["y"] = Y[m]
    out = []
    for atk, g in sub[sub.attack != "none"].groupby("attack"):
        # Each attack is scored against the same real pool, so the two EERs are comparable.
        real = sub[sub.y == 0]
        y = np.r_[np.zeros(len(real)), np.ones(len(g))]
        s = np.r_[real.score.values, g.score.values]
        e, _ = eer_and_threshold(y, s)
        out.append(dict(attack=atk, n=len(g), eer=100 * e))
    print(f"\n{name} by attack")
    print(pd.DataFrame(out).round(2).to_string(index=False))

SWEEP.to_csv("/kaggle/working/layer_sweep.csv", index=False)
RES.to_csv("/kaggle/working/gonogo_results.csv", index=False)
print("\nwrote layer_sweep.csv and gonogo_results.csv")
