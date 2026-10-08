"""Build an Indian English control condition, so switching can be told apart from language.

The dose-response design varies switching through the English word ratio and the two rise
together. A monolingual Indian English condition separates them: accurate on Indian English but
degraded on Hinglish implicates switching, degraded on both implicates out-of-language content.

English comes from the MUCS train partitions, so it arrives through the same channel, protocol
and speakers, and each English utterance is paired with a code-switched one from the same
speaker at the nearest word count.

Requirements
    Kaggle notebook, CPU, internet on. Attached input: codeswitch-deepfake-real.
    Runtime dominated by the ~15 GB download."""

# CELL 1: configuration and environment
import os
import re
import sys
import glob
import time
import shutil
import tarfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

try:
    import soxr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soxr"], check=True)
    import soxr

import numpy as np
import pandas as pd
import soundfile as sf

# configuration
KAGGLE_USER = "aryankumarpandey"
BASE_DATASET = f"/kaggle/input/datasets/{KAGGLE_USER}/codeswitch-deepfake-real"

SCRATCH = "/kaggle/temp/mucs104"
OUT_DIR = "/kaggle/working/audio"
MIRROR = "https://openslr.trmal.net/resources/104"

TRAIN_TARS = {
    "hindi_english_train": "Hindi-English_train.tar.gz",
    "bengali_english_train": "Bengali-English_train.tar.gz",
}
LANG = {"hindi_english_train": "hi", "bengali_english_train": "bn"}

SR_OUT = 16000
# everything goes through 8 kHz first, so bandwidth is equal across corpora and
# cannot explain a later difference in detector error
SR_BOTTLENECK = 8000
MIN_DUR, MAX_DUR = 1.0, 20.0
# below four words, pairing on word count stops meaning anything
MIN_WORDS = 4
MIN_PER_ROLE = 150

os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

for p in ("/kaggle/working", "/kaggle/temp"):
    _, _, free = shutil.disk_usage(p)
    print(f"{p} free {free / 2**30:.1f} GB")

print(f"\nsoxr {soxr.__version__} | soundfile {sf.__version__} | pandas {pd.__version__}")

if not os.path.exists(f"{BASE_DATASET}/manifest_real.csv"):
    raise SystemExit(f"base dataset not mounted at {BASE_DATASET}")

BASE = pd.read_csv(f"{BASE_DATASET}/manifest_real.csv")
print(f"base dataset: {len(BASE):,} utterances")


# CELL 2: download the SLR104 train partitions
for name, fn in TRAIN_TARS.items():
    tgz = os.path.join(SCRATCH, fn)
    if os.path.exists(tgz) and os.path.getsize(tgz) > 1_000_000:
        print(f"[skip] {fn}")
        continue
    t0 = time.time()
    print(f"[get] {fn}", flush=True)
    rc = os.system(f'wget -q --tries=3 -O "{tgz}" "{MIRROR}/{fn}"')
    size = os.path.getsize(tgz) if os.path.exists(tgz) else 0
    print(f"rc={rc} {size / 2**30:.2f} GB in {time.time() - t0:.0f}s")
    if rc != 0 or size < 1_000_000:
        raise SystemExit(f"download failed: {fn}")


# CELL 3: extract
# Each SLR104 tarball contains a top-level `test/` directory, so extracting
# into a shared folder would merge and overwrite them. One namespace each.
for name, fn in TRAIN_TARS.items():
    dest = os.path.join(SCRATCH, name)
    if os.path.isdir(dest) and any(os.scandir(dest)):
        print(f"[skip] {name}")
        continue
    os.makedirs(dest, exist_ok=True)
    t0 = time.time()
    print(f"[tar] {fn} -> {name}/", flush=True)
    with tarfile.open(os.path.join(SCRATCH, fn)) as tf:
        tf.extractall(dest, filter="data")
    print(f"extracted in {time.time() - t0:.0f}s")
    os.remove(os.path.join(SCRATCH, fn))

_, _, free = shutil.disk_usage("/kaggle/temp")
print(f"\n/kaggle/temp free: {free / 2**30:.1f} GB")


# CELL 4: parse Kaldi metadata and score code-switching
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
BENGALI = re.compile(r"[ঀ-৿]")
LATIN = re.compile(r"[A-Za-z]")


def find(root, *names):
    """First file under `root` matching any of `names`."""
    for dirpath, _, filenames in os.walk(root):
        for n in names:
            if n in filenames:
                return os.path.join(dirpath, n)
    return None


def read_kv(path):
    """Kaldi key-value file -> dict."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            parts = ln.strip().split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def build(name):
    """Parse one Kaldi-format corpus into a dataframe of utterances."""
    root = os.path.join(SCRATCH, name)
    text = read_kv(find(root, "text", "transcription.txt"))
    segments = find(root, "segments")
    wavs = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
    stem2path = {os.path.splitext(os.path.basename(w))[0]: w for w in wavs}

    # wav.scp maps recording ids to files; entries may be pipelines, so take
    # the last .wav token rather than assuming field position.
    wavscp = {}
    scp = find(root, "wav.scp")
    if scp:
        with open(scp, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                parts = ln.split()
                hits = [t for t in parts[1:] if t.lower().endswith(".wav")]
                if len(parts) >= 2 and hits:
                    wavscp[parts[0]] = hits[-1]

    utt2spk = read_kv(find(root, "utt2spk"))
    if not utt2spk:
        spk2utt = find(root, "spk2utt")
        if spk2utt:
            with open(spk2utt, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    parts = ln.split()
                    for u in parts[1:]:
                        utt2spk[u] = parts[0]

    rows = []
    with open(segments, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            parts = ln.split()
            if len(parts) != 4:
                continue
            utt, rec = parts[0], parts[1]
            wav = wavscp.get(rec) or stem2path.get(rec)
            if wav and not os.path.isabs(wav):
                wav = stem2path.get(os.path.splitext(os.path.basename(wav))[0])
            rows.append(dict(utt_id=utt, wav=wav, start=float(parts[2]),
                             end=float(parts[3]), speaker=utt2spk.get(utt)))

    df = pd.DataFrame(rows)
    df = df[df.wav.notna()].copy()
    df["set"] = name
    df["lang"] = LANG[name]
    df["kind"] = "cs"
    df["transcript"] = df.utt_id.map(text)
    df["seg_dur"] = df.end - df.start
    return df


def tag_word(word, indic_re):
    """Label a token by script: 'i' indic, 'e' english, None if neither."""
    has_i, has_e = bool(indic_re.search(word)), bool(LATIN.search(word))
    if has_i and not has_e:
        return "i"
    if has_e and not has_i:
        return "e"
    # a token carrying both scripts counts as indic
    if has_i and has_e:
        return "i"
    # digits or punctuation only
    return None


def cs_metrics(text, lang):
    """Per-utterance code-switching statistics.

    cmi is the Code-Mixing Index: 0 when monolingual, 50 when perfectly
    balanced. Note it is symmetric, so it falls again once English dominates.
    eng_ratio is monotonic in English content; the two diverge at high ratios.
    """
    empty = dict(n_words=0, n_eng=0, eng_ratio=0.0, cmi=0.0, switches=0)
    if not isinstance(text, str) or not text.strip():
        return pd.Series(empty)
    indic_re = BENGALI if lang == "bn" else DEVANAGARI
    tags = [t for t in (tag_word(w, indic_re) for w in text.split()) if t]
    n = len(tags)
    if n == 0:
        return pd.Series(empty)
    n_e, n_i = tags.count("e"), tags.count("i")
    cmi = 0.0 if (n_e == 0 or n_i == 0) else 100 * (1 - max(n_e, n_i) / n)
    switches = sum(1 for a, b in zip(tags, tags[1:]) if a != b)
    return pd.Series(dict(n_words=n, n_eng=n_e, eng_ratio=n_e / n,
                          cmi=cmi, switches=switches))


def condition(row):
    if row.n_words == 0:
        return "empty"
    if row.eng_ratio >= 0.999:
        return "english_only"
    if row.eng_ratio <= 0.001:
        return "indic_only"
    return "code_switched"


TRAIN = pd.concat([build(n) for n in TRAIN_TARS], ignore_index=True)
TRAIN = pd.concat(
    [TRAIN, TRAIN.apply(lambda r: cs_metrics(r.transcript, r.lang), axis=1)],
    axis=1,
)
TRAIN["condition"] = TRAIN.apply(condition, axis=1)

print("=" * 76)
print(f"{'set':<24}{'utts':>9}{'spk':>7}{'hours':>9}")
for name, g in TRAIN.groupby("set"):
    print(f"{name:<24}{len(g):>9,}{g.speaker.nunique():>7}"
          f"{g.seg_dur.sum() / 3600:>9.1f}")

print("\nCONDITION BREAKDOWN")
print(pd.crosstab(TRAIN.set, TRAIN.condition).to_string())

ENGLISH = TRAIN[TRAIN.condition == "english_only"]
print(f"\nmonolingual English utterances: {len(ENGLISH):,} "
      f"across {ENGLISH.speaker.nunique()} speakers")
print(f"{'sufficient' if len(ENGLISH) >= MIN_PER_ROLE else 'BELOW THRESHOLD'} "
      f"(threshold {MIN_PER_ROLE})")
for t in ENGLISH.transcript.dropna().head(4):
    print(f"{t[:95]}")


# CELL 5: speaker relationship to the evaluation set
# Determines how the control is described. Shared speakers give the strongest form. Disjoint
# speakers still share corpus, channel and protocol, and the paired design below controls
# speaker within the comparison that carries the argument.
eval_speakers = set(BASE[BASE.role == "test_source"].speaker.dropna().astype(str))
train_speakers = set(TRAIN.speaker.dropna().astype(str))
shared = eval_speakers & train_speakers

print(f"evaluation-set speakers: {len(eval_speakers)}")
print(f"train-partition speakers: {len(train_speakers)}")
print(f"shared: {len(shared)}")
print("\n" + ("speakers overlap: the control shares speakers with the evaluation "
              "set. No leakage, since none of this is training data."
              if shared else
              "speaker-disjoint from the evaluation set; corpus and channel "
              "remain matched."))


# CELL 6: paired selection
# For each English utterance, take the code-switched utterance from the same
# speaker whose word count is closest, without replacement. This holds speaker
# and length constant so that switching is the only difference.
english = TRAIN[(TRAIN.condition == "english_only")
                & (TRAIN.n_words >= MIN_WORDS)].copy()
switched = TRAIN[(TRAIN.condition == "code_switched")
                 & (TRAIN.n_words >= MIN_WORDS)].copy()

pairs = []
for speaker, group in english.groupby("speaker"):
    pool = switched[switched.speaker == speaker]
    if pool.empty:
        continue
    used = set()
    for utt in group.itertuples():
        available = pool[~pool.utt_id.isin(used)]
        if available.empty:
            break
        nearest = (available.n_words - utt.n_words).abs().idxmin()
        used.add(available.loc[nearest, "utt_id"])
        pairs.append((utt.utt_id, available.loc[nearest, "utt_id"]))

english_ids = {a for a, _ in pairs}
switched_ids = {b for _, b in pairs}

SELECTED = pd.concat([
    english[english.utt_id.isin(english_ids)]
        .assign(role="english_indian", cs_bin="english"),
    switched[switched.utt_id.isin(switched_ids)]
        .assign(role="cs_paired", cs_bin="cs_paired"),
], ignore_index=True)
SELECTED["split"] = "test"

print(f"matched pairs: {len(pairs):,} speakers: {SELECTED.speaker.nunique()}")
print("\nword-count balance across roles (should be close)")
print(SELECTED.groupby("role").n_words.describe()[["count", "mean", "50%"]]
      .round(1).to_string())
print("\nmedian eng_ratio by role (english_indian must be 1.0)")
print(SELECTED.groupby("role").eng_ratio.median().round(3).to_string())

SELECTED["fname"] = (
    SELECTED.set.astype(str) + "__"
    + SELECTED.utt_id.astype(str).str.replace(r"[^A-Za-z0-9._-]", "_", regex=True)
)
collisions = SELECTED.fname.duplicated(keep=False)
if collisions.any():
    SELECTED.loc[collisions, "fname"] += "__" + SELECTED.index[collisions].astype(str)
assert SELECTED.fname.is_unique


# CELL 7: segment, resample, encode
def render(rec):
    """Cut one segment, apply the bandwidth bottleneck, write FLAC.

    Returns (fname, duration, source_rate, status). Existing outputs are
    reused so an interrupted run can resume.
    """
    dst = os.path.join(OUT_DIR, rec.fname + ".flac")
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        try:
            info = sf.info(dst)
            return rec.fname, info.duration, info.samplerate, "cached"
        except Exception:
            pass

    try:
        info = sf.info(rec.wav)
        sr = info.samplerate
        start = max(0, int(float(rec.start) * sr))
        stop = min(info.frames, int(float(rec.end) * sr))
        if stop <= start:
            return rec.fname, 0.0, sr, "badseg"
        x, _ = sf.read(rec.wav, start=start, stop=stop, dtype="float32")
    except Exception as e:
        return rec.fname, 0.0, 0, f"err:{type(e).__name__}"

    if x.ndim > 1:
        x = x.mean(axis=1)

    dur = len(x) / sr
    if dur < MIN_DUR:
        return rec.fname, dur, sr, "short"
    if dur > MAX_DUR:
        return rec.fname, dur, sr, "long"

    if sr != SR_BOTTLENECK:
        x = soxr.resample(x, sr, SR_BOTTLENECK, quality="VHQ")
    x = soxr.resample(x, SR_BOTTLENECK, SR_OUT, quality="VHQ")
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    sf.write(dst, x, SR_OUT, format="FLAC", subtype="PCM_16")
    return rec.fname, len(x) / SR_OUT, sr, "ok"


records = list(SELECTED.itertuples(index=False))
t0, done, results = time.time(), 0, []
with ThreadPoolExecutor(max_workers=8) as pool:
    for result in pool.map(render, records):
        results.append(result)
        done += 1
        if done % 500 == 0 or done == len(records):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"{done:,}/{len(records):,} {rate:.1f} utt/s", flush=True)

RENDERED = pd.DataFrame(results, columns=["fname", "dur", "src_sr", "status"])
print("\n" + RENDERED.status.str.split(":").str[0].value_counts().to_string())
print("\nsource sample rates (SLR104 is natively 16 kHz)")
print(RENDERED.src_sr.value_counts().to_string())


# CELL 8: retain intact pairs and write the manifest
# A pair is only usable if both halves rendered; dropping one half would
# break the within-speaker balance the design depends on.
SELECTED = SELECTED.merge(RENDERED, on="fname", how="left")
KEEP = SELECTED[SELECTED.status.isin(["ok", "cached"])].copy()

rendered_ids = set(KEEP.utt_id)
intact = [(a, b) for a, b in pairs if a in rendered_ids and b in rendered_ids]
keep_ids = {i for pair in intact for i in pair}
KEEP = KEEP[KEEP.utt_id.isin(keep_ids)].copy()

KEEP["path"] = "audio/" + KEEP.fname + ".flac"
KEEP["label"] = "real"

print(f"pairs selected: {len(pairs):,} intact after rendering: {len(intact):,}")
print("\n" + "=" * 76)
print(KEEP.groupby(["role", "set"]).agg(
    utts=("fname", "size"),
    spk=("speaker", "nunique"),
    hours=("dur", lambda s: s.sum() / 3600),
    med_words=("n_words", "median"),
    med_eng=("eng_ratio", "median"),
).round(2).to_string())

for role in ("english_indian", "cs_paired"):
    n = int((KEEP.role == role).sum())
    print(f"{role} {n} utts, "
          f"{'ok' if n >= MIN_PER_ROLE else 'below threshold'}")

COLUMNS = ["fname", "path", "utt_id", "set", "lang", "kind", "speaker",
           "transcript", "n_words", "n_eng", "eng_ratio", "cmi", "switches",
           "condition", "cs_bin", "role", "split", "label", "dur", "src_sr"]
KEEP[COLUMNS].to_csv("/kaggle/working/manifest_english_control.csv", index=False)
pd.DataFrame(intact, columns=["english_utt", "cs_utt"]).to_csv(
    "/kaggle/working/pairs.csv", index=False)

total = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
print(f"\naudio/ {len(os.listdir(OUT_DIR)):,} files, {total / 2**30:.2f} GB")


# CELL 9: perceptual spot-check and dataset README
# Automated checks cannot confirm that the English is Indian-accented or that
# the bandwidth reduction preserved intelligibility.
import IPython.display as ipd

for role, n in (("english_indian", 3), ("cs_paired", 2)):
    print(f"\n--- {role} ---")
    for r in KEEP[KEEP.role == role].sample(n, random_state=5).itertuples():
        print(f"[{r.speaker} | {r.dur:.1f}s] {str(r.transcript)[:100]}")
        ipd.display(ipd.Audio(os.path.join("/kaggle/working", r.path)))

with open("/kaggle/working/README.md", "w", encoding="utf-8") as fh:
    fh.write(
        "# Indian English control condition\n\n"
        "Companion to `codeswitch-deepfake-real`; mount both.\n\n"
        "Monolingual Indian English utterances from the MUCS/OpenSLR 104 train\n"
        "partitions, each paired with a code-switched utterance from the same\n"
        "speaker at the nearest word count. Separates degradation caused by\n"
        "code-switching from degradation caused by English content alone.\n\n"
        "Preprocessing is identical to the main dataset: native rate -> 8 kHz ->\n"
        "16 kHz mono FLAC, equalising bandwidth across corpora.\n\n"
        "Roles: `english_indian`, `cs_paired`. `pairs.csv` holds the mapping.\n\n"
        "Source data is CC BY-SA 4.0.\n"
    )

print("\nPublish as a separate dataset rather than a new version of the base "
      "dataset; a new version would replace it and require re-rendering all "
      "existing audio.")
