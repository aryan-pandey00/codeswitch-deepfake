"""Measure whether the vocoded fakes are separable by processing alone, and close it if they are.

Cell 6 of 21 found every vocoder three to eight times below the real side on energy above
4 kHz, in the same direction for all four. A shared offset across unlike paths is the pipeline,
not the vocoders: the real corpus passed the 8 kHz bottleneck once, a vocoded file passes it
again after resynthesis, and that band holds resampler leakage and nothing else.

Ten clips cannot tell that from noise, so this asks what medians cannot: is one scalar enough
to separate the classes. Measured 21 Sep, band ratio 0.61 pooled, energy and centroid at
chance. 0.61 is weak and cells 5 and 6 close it anyway, since a weak scalar correlating with
the label is still one a model may use. The fakes are not altered; the genuine class is given
the same processing history instead, so resynthesis is the only difference left.

Requirements
    Kaggle notebook, CPU, no internet, roughly fifteen minutes. Resumable.
    Attached inputs: codeswitch-deepfake-real, codeswitch-deepfake-vocoded.
    Cells 1-4 read only. Cells 5-6 write passthrough/*.flac and manifest_passthrough.csv.
"""

# CELL 1: mounts
import os
import glob
import time

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from sklearn.metrics import roc_auc_score

KAGGLE_USER = "aryankumarpandey"
REAL_ROOT = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"

# Saved dataset or previous version's output mount at different paths; whichever is present wins.
CANDIDATES = [
    f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-vocoded",
    "/kaggle/input/codeswitch-deepfake-vocoded",
] + sorted(glob.glob("/kaggle/input/*"))

VOC_ROOT = next((c for c in CANDIDATES if os.path.exists(f"{c}/manifest_vocoded.csv")), None)
assert VOC_ROOT, f"no manifest_vocoded.csv under any of {CANDIDATES}"
print(f"vocoded: {VOC_ROOT}")

WORK = "/kaggle/working"
OUT = f"{WORK}/passthrough"
MANIFEST = f"{WORK}/manifest_passthrough.csv"

TARGET_SR = 16000
BOTTLENECK_HZ = 8000
SEED = 1337
# Sources, not files: each contributes one real and one row per vocoder.
N_SOURCES = 300

os.makedirs(OUT, exist_ok=True)

VOC = pd.read_csv(f"{VOC_ROOT}/manifest_vocoded.csv")
AUDIO = f"{VOC_ROOT}/vocoded" if os.path.isdir(f"{VOC_ROOT}/vocoded") else VOC_ROOT
print(f"{len(VOC):,} vocoded rows, {VOC.vocoder.nunique()} paths: {sorted(VOC.vocoder.unique())}")


def find(name):
    """Locate a mounted folder. Mounts are namespaced (input/datasets/<user>/<slug>/,
    input/notebooks/<user>/<slug>/), sometimes deeper, so depth is searched not hardcoded."""
    for depth in range(1, 7):
        hits = [p for p in sorted(glob.glob("/kaggle/input/" + "*/" * depth + name))
                if os.path.isdir(p)]
        if hits:
            return hits[0]
    return None


# CELL 2: the scalars
# One number per clip, which is the point: a feature this cheap is one any model picks up free.
def scalars(y, sr=TARGET_SR):
    n = min(len(y), sr * 4)
    if n < sr // 2:
        return None
    w = y[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    hi = freqs >= BOTTLENECK_HZ // 2

    return {
        "band_ratio": float(spec[hi].sum() / (spec[~hi].sum() + 1e-12)),
        "rms": float(np.sqrt(np.mean(y[:n] ** 2))),
        "centroid": float((spec * freqs).sum() / (spec.sum() + 1e-12)),
    }


def measure(genuine_files, genuine_dir, sources, feature_set):
    """Scalars for a genuine set and the vocoded versions of the same sources.

    Called twice, against the frozen corpus and then against the twins, so the before and after
    are the same measurement rather than two that merely resemble each other.
    """
    by_source = {s: g for s, g in VOC[VOC.source_path.isin(set(sources))].groupby("source_path")}
    rows, missing = [], 0

    for src, gen_path in genuine_files.items():
        try:
            y, sr = sf.read(os.path.join(genuine_dir, gen_path), dtype="float32")
            g = scalars(y, sr)
        except Exception:
            missing += 1
            continue
        if g is None or src not in by_source:
            continue

        versions = []
        for r in by_source[src].itertuples():
            try:
                y, sr = sf.read(os.path.join(AUDIO, r.fname), dtype="float32")
            except Exception:
                missing += 1
                continue
            s = scalars(y, sr)
            if s:
                versions.append({**s, "vocoder": r.vocoder, "label": 1, "source_path": src})

        if versions:
            rows.append({**g, "vocoder": "genuine", "label": 0, "source_path": src})
            rows.extend(versions)

    return pd.DataFrame(rows), missing


def auc_table(D, features):
    """AUC per path and pooled, direction removed since a backwards feature separates just as well.

    Per path matters: one bad vocoder is a dropped path, all of them is the pipeline.
    """
    gen = D[D.label == 0]
    out = []
    for voc in sorted(D[D.label == 1].vocoder.unique()):
        pair = pd.concat([gen, D[D.vocoder == voc]])
        row = {"vocoder": voc}
        for f in features:
            a = roc_auc_score(pair.label, pair[f])
            row[f] = round(max(a, 1 - a), 3)
        out.append(row)

    pooled = {"vocoder": "ALL"}
    for f in features:
        a = roc_auc_score(D.label, D[f])
        pooled[f] = round(max(a, 1 - a), 3)
    out.append(pooled)
    return pd.DataFrame(out).set_index("vocoder")


FEATURES = ["band_ratio", "rms", "centroid"]


# CELL 3: read matched sets
# Sampling by source keeps every fake's genuine twin in the comparison.
SOURCES = (VOC.source_path.drop_duplicates()
           .sample(min(N_SOURCES, VOC.source_path.nunique()), random_state=SEED)
           .tolist())

D, missing = measure({s: s for s in SOURCES}, REAL_ROOT, SOURCES, FEATURES)
print(f"{len(D):,} clips read across {D.source_path.nunique():,} sources, {missing} unreadable\n")
print(D.groupby("vocoder")[FEATURES].median().to_string())


# CELL 4: separability, against the frozen corpus
BEFORE = auc_table(D, FEATURES)
print("\nAUC of one scalar, real against each path. 0.50 is indistinguishable.")
print(BEFORE.to_string())

worst = BEFORE.loc["ALL"].max()
NEEDS_FIX = worst >= 0.60
print(f"\nworst pooled AUC across the three scalars: {worst:.3f}")
if not NEEDS_FIX:
    print("No usable confound. Generation stands as it is; skip to 22.")
else:
    print("Separable on processing alone. Cells 5 and 6 build the matched genuine class.")

# The size of the gap the twins have to remove, as a ratio since the absolute values are leakage.
if NEEDS_FIX:
    med = D.groupby("vocoder").band_ratio.median()
    print("\nband ratio against real, per path")
    for voc in sorted(v for v in med.index if v != "genuine"):
        print(f" {voc}: {med[voc] / med['genuine']:.2f}x")
    print("All on the same side of 1.0 is the pipeline. Spread across it is the vocoders.")


# CELL 5: build the matched genuine class
# The transform is copied from 21, not imported, because a divergence between the two would be
# invisible. The peak limiter is part of it: the fakes were scaled when resampling overshot, and
# genuine files skipping that step would replace one confound with another.
def bottleneck(x, sr):
    """The same band every file in this project was forced through."""
    if sr != BOTTLENECK_HZ:
        x = soxr.resample(x, sr, BOTTLENECK_HZ)
    return soxr.resample(x, BOTTLENECK_HZ, TARGET_SR)


# Sources come from the vocoded manifest, not a fresh sample: one that failed during generation
# has no fakes, so twinning it adds a genuine file with nothing to pair against. Metadata travels
# with it, so role, split and speaker stay as the fakes recorded them.
CARRY = [c for c in ("set", "role", "split", "speaker", "transcript") if c in VOC.columns]
SRC = VOC.drop_duplicates("source_path")[["source_path"] + CARRY].reset_index(drop=True)
RESUME = find("passthrough")
if RESUME:
    print(f"resuming against {RESUME}")

if not NEEDS_FIX:
    print("skipped, cell 4 found nothing to close")
    PASS = pd.DataFrame()
else:
    print(f"{len(VOC):,} vocoded files from {len(SRC):,} sources")
    rows, failed = [], []
    t_start = time.time()

    for i, r in enumerate(SRC.itertuples()):
        fname = f"pass__{os.path.splitext(os.path.basename(r.source_path))[0]}.flac"
        dst = os.path.join(OUT, fname)
        carried = os.path.join(RESUME, fname) if RESUME else None

        if not os.path.exists(dst) and carried and os.path.exists(carried):
            import shutil
            shutil.copyfile(carried, dst)

        if not os.path.exists(dst):
            try:
                x, sr = sf.read(f"{REAL_ROOT}/{r.source_path}", dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                y = bottleneck(np.asarray(x, dtype=np.float32), sr)
                peak = float(np.abs(y).max()) or 1.0
                if peak > 0.99:
                    y = y * (0.99 / peak)
                tmp = dst + ".part"
                sf.write(tmp, y, TARGET_SR, format="FLAC", subtype="PCM_16")
                os.replace(tmp, dst)
            except Exception as exc:
                failed.append({"path": r.source_path, "err": str(exc)[:200]})
                continue

        rows.append({
            "fname": fname,
            "source_path": r.source_path,
            "attack": "none",
            "vocoder": "passthrough",
            "held_out": False,
            "label": 0,
            **{c: getattr(r, c, None) for c in CARRY},
            "dur": sf.info(dst).duration,
        })

        if i and i % 1000 == 0:
            print(f"{i:,}/{len(SRC):,}, {(time.time()-t_start)/60:.1f} min", flush=True)

    PASS = pd.DataFrame(rows)
    PASS.to_csv(MANIFEST, index=False)
    print(f"\n{len(PASS):,} twins written in {(time.time()-t_start)/60:.1f} min")
    if failed:
        pd.DataFrame(failed).to_csv(f"{WORK}/passthrough_failures.csv", index=False)
        print(f"{len(failed)} failures, see passthrough_failures.csv")

    # Every fake must have its twin present, since that pairing is what the training set rests on.
    print(f"fakes whose genuine twin is missing: {len(set(VOC.source_path) - set(PASS.source_path))}")


# CELL 6: the same measurement again, against the twins
# Near 0.50 means the extra resample was the whole story. Otherwise 22 should wait.
if NEEDS_FIX and len(PASS):
    twinned = PASS[PASS.source_path.isin(SOURCES)].set_index("source_path").fname.to_dict()
    A, _ = measure(twinned, OUT, list(twinned), FEATURES)
    AFTER = auc_table(A, FEATURES)

    print("\nAUC against the twins, same scalars, same sources.")
    print(AFTER.to_string())

    print("\npooled, before and after")
    for f in FEATURES:
        print(f" {f}: {BEFORE.loc['ALL', f]:.3f} -> {AFTER.loc['ALL', f]:.3f}")

    pooled = AFTER.loc["ALL"].max()
    print(f"\nworst pooled AUC: {worst:.3f} -> {pooled:.3f}")
    print("The extra resample was the whole of it. 22 uses manifest_passthrough.csv as genuine."
          if pooled < 0.56 else
          "Still separable, so something beyond the resample differs. Do not train on this yet.")

    # A number that lives only in a cell output does not exist.
    CONFOUND = f"{WORK}/vocoder_confound.csv"
    (BEFORE.stack().rename("auc").reset_index().assign(genuine_class="frozen_corpus")
     .pipe(lambda b: pd.concat([
         b, AFTER.stack().rename("auc").reset_index().assign(genuine_class="passthrough_twins")],
         ignore_index=True))
     .rename(columns={"level_1": "scalar"})
     .assign(n_sources=len(SOURCES))
     .to_csv(CONFOUND, index=False))
    print(f"\nwrote {CONFOUND}, copy to paper/results/vocoder_confound.csv")
