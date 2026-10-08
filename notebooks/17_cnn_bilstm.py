"""Retrain an MFCC CNN-BiLSTM on these splits, to test whether the collapse is architectural.

The finding so far rests on one architecture, a frozen self-supervised transformer with a linear
head. CNN-BiLSTM over MFCCs shares no component with XLS-R and learns its own filters rather
than inheriting a representation from 436,000 hours.

Two choices avoid shortcuts rather than chase accuracy. Inputs are a fixed five-second span at a
ten-millisecond hop, the window the service scores. Short clips are extended by wrapping rather
than zero-padding, since generated clips run systematically shorter and the model could
otherwise separate classes on its own padding.

Requirements
    Kaggle notebook, GPU T4, 30-40 minutes.
    Attached inputs: codeswitch-deepfake-feats (index.csv only), -real, -fakes."""

# CELL 1: configuration, mounts, and the split definition
import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
SEED = 1337
N_FIX = 8

N_MFCC = 40
N_FFT = 400
HOP = 160
FRAMES = 500

EPOCHS = 30
BATCH = 64
PATIENCE = 5

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
CORPORA = ("hindi_english", "bengali_english")

np.random.seed(SEED)


def find(name):
    """Kaggle mounts inputs under the notebook slug, not the dataset name."""
    return next(
        (p for depth in ("*", "*/*", "*/*/*")
         for p in sorted(glob.glob(f"/kaggle/input/{depth}/{name}")) if os.path.isdir(p)),
        None)


FEAT_DIR = find("feats")
FAKE_DIR = find("fakes")

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"

if FEAT_DIR is None or FAKE_DIR is None or not os.path.isdir(REAL_ROOT):
    sys.exit("attach codeswitch-deepfake-feats, -real and -fakes")

# index.csv rather than the manifests, so the rows, labels and splits are byte-identical to the
# ones the linear probe was measured on and the two architectures cannot silently diverge.
IDX = pd.read_csv(f"{FEAT_DIR}/index.csv")
Y = (IDX.label == "fake").values.astype(int)

IDX["path"] = [
    f"{FAKE_DIR}/{os.path.basename(p)}" if lab == "fake"
    else f"{REAL_ROOT}/audio/{os.path.basename(p)}"
    for p, lab in zip(IDX.file, IDX.label)
]

missing = [p for p in IDX.path.sample(200, random_state=SEED) if not os.path.exists(p)]
if missing:
    sys.exit(f"{len(missing)} of 200 sampled paths do not resolve, first {missing[0]}")

BASE_TRAIN = ((IDX.role == "train_source") & (IDX.split == "train")).values
DEV = ((IDX.role == "train_source") & (IDX.split == "dev")).values
CONTROL = (IDX.role == "control_indomain").values

SPK = {c: sorted(IDX[IDX.set == c].speaker.dropna().unique()) for c in CORPORA}
rng = np.random.RandomState(SEED)
ADDED = {c: sorted(rng.choice(SPK[c], N_FIX, replace=False)) for c in CORPORA}

mask_added = np.zeros(len(IDX), dtype=bool)
for c, spks in ADDED.items():
    mask_added |= ((IDX.set == c) & IDX.speaker.isin(spks)).values

HELD = IDX.set.isin(list(CORPORA)).values & ~mask_added
TRAIN = {"baseline": BASE_TRAIN, "fixed": BASE_TRAIN | mask_added}

print(f"{len(IDX):,} utterances, {Y.sum():,} fake")
print(f"baseline trains on {int(BASE_TRAIN.sum()):,}, fixed on {int(TRAIN['fixed'].sum()):,}")
print(f"held-out target corpora: {int(HELD.sum()):,} rows, "
      f"{int(mask_added.sum()):,} rows excluded as fixed-head training data")


# CELL 2: the model, defined before anything expensive runs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"tensorflow {tf.__version__}, GPUs {tf.config.list_physical_devices('GPU')}")


def build():
    """Convolutional front end over the MFCC map, then a recurrent pass over time.

    Pooling is heavier on the coefficient axis than on time, because the recurrent layer exists
    to model time and collapsing that axis early would leave it nothing to do.

    Capacity is set to land near half a million parameters, the same order as the published
    CNN-BiLSTM detectors this stands in for. An undersized version of the architecture would
    turn an architecture comparison into a capacity comparison.
    """
    inp = keras.Input(shape=(N_MFCC, FRAMES, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((5, 2))(x)

    # Time has to lead before the axes are merged. Reshaping straight from (freq, time, channel)
    # interleaves frequency into the time axis and hands the recurrent layer a scrambled
    # sequence, which trains to a plausible-looking number and means nothing.
    x = layers.Permute((2, 1, 3))(x)
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out)


build().summary()


# CELL 3: MFCC extraction
import librosa
from joblib import Parallel, delayed

SPAN = FRAMES * HOP


def load_span(path):
    """A fixed five-second span, wrapped rather than zero-padded when the clip is shorter.

    Generated clips in this dataset run about nine percent shorter than the reals they were
    made from. Zero padding would therefore differ in quantity by class, and a convolutional
    model can separate classes on the length of a silent tail without ever looking at speech.
    Wrapping keeps the padded region statistically the same material as the clip itself.
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if len(wav) < SPAN:
        wav = np.pad(wav, (0, SPAN - len(wav)), mode="wrap")
    else:
        start = (len(wav) - SPAN) // 2
        wav = wav[start: start + SPAN]
    m = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    return m[:, :FRAMES].astype(np.float16)


t0 = time.time()
X = np.stack(Parallel(n_jobs=4, batch_size=64)(
    delayed(load_span)(p) for p in IDX.path.values))
print(f"MFCCs {X.shape} {X.nbytes/2**30:.2f} GiB in {time.time()-t0:.0f}s")

# Normalisation statistics come from the baseline's training rows only. Computing them over
# everything would leak the target corpora's channel into every model through the scaler, which
# is the exact variable under study.
MU = X[BASE_TRAIN].astype(np.float32).mean((0, 2), keepdims=True)
SD = X[BASE_TRAIN].astype(np.float32).std((0, 2), keepdims=True) + 1e-6
print(f"normalised on {int(BASE_TRAIN.sum()):,} training rows")


# CELL 4: train both heads
def batches(rows, shuffle):
    """Normalise and add the channel axis lazily, so the fp16 array stays the only large copy."""
    def gen():
        order = np.random.permutation(rows) if shuffle else rows
        for i in range(0, len(order), BATCH):
            sel = order[i: i + BATCH]
            xb = (X[sel].astype(np.float32) - MU) / SD
            yield xb[..., None], Y[sel].astype(np.float32)

    return tf.data.Dataset.from_generator(
        gen, output_signature=(
            tf.TensorSpec((None, N_MFCC, FRAMES, 1), tf.float32),
            tf.TensorSpec((None,), tf.float32))).prefetch(tf.data.AUTOTUNE)


MODELS, CURVES = {}, []
dev_rows = np.flatnonzero(DEV)

for name, train_mask in TRAIN.items():
    keras.utils.set_random_seed(SEED)
    train_rows = np.flatnonzero(train_mask)

    model = build()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    hist = model.fit(
        batches(train_rows, True), validation_data=batches(dev_rows, False),
        epochs=EPOCHS, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                 restore_best_weights=True)])
    print(f"{name} trained in {time.time()-t0:.0f}s over {len(hist.history['loss'])} epochs")

    MODELS[name] = model
    model.save(f"{OUT}/cnn_bilstm_{name}.keras")
    for e in range(len(hist.history["loss"])):
        CURVES.append(dict(model=f"cnn_bilstm_{name}", epoch=e + 1,
                           **{k: float(v[e]) for k, v in hist.history.items()}))

pd.DataFrame(CURVES).to_csv(f"{OUT}/curves_cnn_bilstm.csv", index=False)
print(f"wrote curves_cnn_bilstm.csv, {len(CURVES)} epochs total")


# CELL 5: score everything, on the threshold the linear probe used
# The decision rule has to be the one the other architecture was judged by, or the comparison
# measures the thresholds rather than the models. There it was fitted on in-domain dev and
# nothing else, so the same is done here, without ever looking at a target corpus.
def predict_all(model):
    out = np.empty(len(IDX), dtype=np.float32)
    for i in range(0, len(IDX), 256):
        xb = (X[i: i + 256].astype(np.float32) - MU) / SD
        out[i: i + 256] = model.predict(xb[..., None], verbose=0).ravel()
    return out


from sklearn.metrics import roc_curve

SCORES = {}
for name, model in MODELS.items():
    p = predict_all(model)
    d = p[DEV]
    lo, hi = d[Y[DEV] == 1].min(), d[Y[DEV] == 0].max()
    if lo > hi:
        thr = float((lo + hi) / 2)
        sep = "separable"
    else:
        fpr, tpr, cut = roc_curve(Y[DEV], d)
        thr = float(cut[int(np.nanargmin(np.abs((1 - tpr) - fpr)))])
        sep = "overlapping"
    # Shifted so that above zero means fake, matching the convention of every other result.
    SCORES[f"cnn_bilstm_{name}"] = p - thr
    print(f"{name}: dev {sep}, threshold {thr:.4f}")


# CELL 6: metrics, in the same shape as notebook 16 so the tables concatenate
from sklearn.metrics import roc_auc_score

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


def wilson(k, n, z=1.96):
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
        auc=100 * roc_auc_score(y, s) if both else nan,
        eer=eer(y, s) if both else nan,
    )


rows = []
for model, score in SCORES.items():
    for label, mask in CONDITIONS:
        if mask.sum():
            rows.append(dict(model=model, condition=label, **metrics(Y[mask], score[mask])))

M = pd.DataFrame(rows)
M.to_csv(f"{OUT}/metrics_cnn_bilstm.csv", index=False)

LEAF = [(l, m) for l, m in CONDITIONS if "corpus, all" not in l and "pooled" not in l]
cond_of = np.full(len(IDX), "", dtype=object)
for label, mask in LEAF:
    cond_of[mask] = label
keep = cond_of != ""

RAW = pd.DataFrame({"row": np.flatnonzero(keep), "set": IDX.set.values[keep],
                    "speaker": IDX.speaker.values[keep], "cs_bin": CS[keep],
                    "condition": cond_of[keep], "label": IDX.label.values[keep], "y": Y[keep]})
for model, score in SCORES.items():
    RAW[f"score_{model}"] = score[keep]
RAW.to_csv(f"{OUT}/scores_cnn_bilstm.csv", index=False)


def fmt(v, dp=1):
    return "-" if v != v else f"{v:.{dp}f}"


print("\n| Model | Condition | n | Acc | AUC | EER | FAR | FRR (95% CI) |")
print("|---|---|---|---|---|---|---|---|")
for _, r in M.iterrows():
    print(f"| {r.model} | {r.condition} | {r.n:,} | {fmt(r.accuracy)} | {fmt(r.auc, 2)} | "
          f"{fmt(r.eer, 2)} | {fmt(r.far, 2)} | {fmt(r.frr)} "
          f"({fmt(r.frr_lo)}-{fmt(r.frr_hi)}) |")

print("\nthe comparison this notebook exists to make\n")
for model in SCORES:
    g = M[M.model == model].set_index("condition")
    dev = g.loc["in-domain dev, unseen speakers"]
    pooled = g.loc["out of corpus, pooled"]
    print(f"{model}: in-domain EER {fmt(dev.eer, 2)}%, out of corpus accuracy "
          f"{fmt(pooled.accuracy)}%, misses {fmt(pooled.far, 2)}% of fakes and rejects "
          f"{fmt(pooled.frr)}% of genuine speech")

print("\nFor reference, the frozen XLS-R linear probe on the same rows reached in-domain EER "
      "0.00%, and out of corpus rejected 29.2% of genuine speech as the baseline and 0.11% "
      "once eight speakers per corpus were added. A CNN-BiLSTM showing the same shape means "
      "the effect is not a property of self-supervised pretraining.")
