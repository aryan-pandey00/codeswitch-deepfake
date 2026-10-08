"""Build the real-audio dataset once and publish it as a reusable Kaggle Dataset.

/kaggle/working is wiped between sessions, so starting from the raw corpora meant
re-downloading 7 GB every time. This does it once and writes a private Dataset that mounts
read-only. It replaces the five acquisition scripts in superseded/ and must run top to bottom
in a fresh container.

Resampling is native -> 8 kHz -> 16 kHz deliberately. The monolingual sets are natively 8 kHz
and the code-switched 16 kHz, a split falling exactly on the train/test boundary, so without
the bottleneck any later result would be attributable to bandwidth.

Requirements
    Kaggle notebook, no accelerator, internet on. 45-75 min, mostly download.
    Produces ~14,000 FLAC and manifest_real.csv.
"""


# CELL 1: environment, disk headroom, dependencies
import os, sys, shutil, subprocess, time

print(sys.version.split()[0])
for p in ("/kaggle/working", "/kaggle/temp"):
    os.makedirs(p, exist_ok=True)
    total, used, free = shutil.disk_usage(p)
    print(f"{p} free {free/2**30:.1f} GB of {total/2**30:.0f} GB")

try:
    import soxr
except ImportError:
    print("[pip] installing soxr ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soxr"], check=True)
    import soxr

import numpy as np
import pandas as pd
import soundfile as sf

print(f"\nsoxr {soxr.__version__} | soundfile {sf.__version__} | "
      f"pandas {pd.__version__} | numpy {np.__version__}")

# Roughly 12 GB of scratch is needed for the raw corpora, plus ~2 GB for the output.
_, _, free_temp = shutil.disk_usage("/kaggle/temp")
if free_temp / 2**30 < 14:
    print("\n!! less than 14 GB free in /kaggle/temp, tarballs are deleted right "
          "after extraction below, but watch this.")


# CELL 2: resolve the real filenames on OpenSLR, then download
# Tarball names are not hardcoded. The mirror's directory listing is parsed and matched by
# keyword, so an OpenSLR rename surfaces as the printed listing rather than a silent 404.
import urllib.request, re

DATA = "/kaggle/temp/mucs"
os.makedirs(DATA, exist_ok=True)
MIRROR = "https://openslr.trmal.net/resources"


def remote_list(res):
    html = urllib.request.urlopen(f"{MIRROR}/{res}/", timeout=60).read()
    html = html.decode("utf-8", "replace")
    return sorted({os.path.basename(h) for h in
                   re.findall(r'href="([^"?#]+\.tar\.gz)"', html)})


INDEX = {r: remote_list(r) for r in (103, 104)}
for r, files in INDEX.items():
    print(f"SLR{r} offers:")
    for f in files:
        print(f"{f}")
    print()


def pick(res, *musts):
    for f in INDEX[res]:
        low = f.lower()
        if all(m.lower() in low for m in musts):
            return f
    raise SystemExit(f"no SLR{res} tarball matching {musts}, see listing above")


# name -> (openslr resource, tarball). The official test partitions of the code-switched
# sets are the right choice here: the train partitions run ~90 h / 30 GB, and the test
# partitions already supply enough utterances per intensity bin.
# hindi_mono_train is the training source; hindi_mono is the in-domain control
WANT = {
    "hindi_mono_train": (103, pick(103, "hindi", "train")),
    "hindi_mono": (103, pick(103, "hindi", "test")),
    "hindi_english": (104, pick(104, "hindi-english", "test")),
    "bengali_english": (104, pick(104, "bengali-english", "test")),
}
print("RESOLVED:")
for k, (r, f) in WANT.items():
    print(f"{k} SLR{r}/{f}")

print()
for name, (res, fn) in WANT.items():
    tgz = os.path.join(DATA, fn)
    if os.path.exists(tgz) and os.path.getsize(tgz) > 1_000_000:
        print(f"[skip] {fn} already here ({os.path.getsize(tgz)/2**30:.2f} GB)")
        continue
    t0 = time.time()
    print(f"[get] {fn} ...", flush=True)
    rc = os.system(f'wget -q --tries=3 -O "{tgz}" "{MIRROR}/{res}/{fn}"')
    size = os.path.getsize(tgz) if os.path.exists(tgz) else 0
    print(f"rc={rc} {size/2**30:.2f} GB in {time.time()-t0:.0f}s")
    if rc != 0 or size < 1_000_000:
        raise SystemExit(f"download failed for {fn}")


# CELL 3: extract each tarball into its own namespace
# Every SLR104 tarball contains a top-level `test/` directory, so extracting them into a
# shared folder silently merges and clobbers them. One directory per corpus, always.
import tarfile

for name, (res, fn) in WANT.items():
    dest = os.path.join(DATA, name)
    if os.path.isdir(dest) and any(os.scandir(dest)):
        print(f"[skip] {name} already extracted")
        continue
    os.makedirs(dest, exist_ok=True)
    t0 = time.time()
    print(f"[tar] {fn} -> {name}/ ...", flush=True)
    with tarfile.open(os.path.join(DATA, fn)) as tf:
        tf.extractall(dest, filter="data")
    print(f"extracted in {time.time()-t0:.0f}s")

# tarballs are dead weight now; reclaim the disk before the audio pass
for fn in {f for _, f in WANT.values()}:
    p = os.path.join(DATA, fn)
    if os.path.exists(p):
        os.remove(p)
        print(f"[rm] {fn}")

_, _, free = shutil.disk_usage("/kaggle/temp")
print(f"\n/kaggle/temp free: {free/2**30:.1f} GB")


# CELL 4: build the manifest (corrected parser)
# Two layouts to handle:
# Kaldi segmented (the code-switched sets): wav.scp, segments and utt2spk, several
# utterances cut from one long recording. Flat (the monolingual Hindi sets): one wav per
# utterance, id <prompt>_<speaker>, so the speaker is the suffix. Reading it as the prefix
# reports 386 speakers, which is the prompt count against a real count of 19.
import glob

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
BENGALI = re.compile(r"[ঀ-৿]")
LATIN = re.compile(r"[A-Za-z]")

CFG = {
    "hindi_mono_train": dict(lang="hi", kind="mono"),
    "hindi_mono": dict(lang="hi", kind="mono"),
    "hindi_english": dict(lang="hi", kind="cs"),
    "bengali_english": dict(lang="bn", kind="cs"),
}


def find(root, *names):
    for dirpath, _, filenames in os.walk(root):
        for n in names:
            if n in filenames:
                return os.path.join(dirpath, n)
    return None


def read_kv(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            p = ln.strip().split(None, 1)
            if len(p) == 2:
                out[p[0]] = p[1]
    return out


def build(name, cfg):
    root = os.path.join(DATA, name)
    if not os.path.isdir(root):
        print(f"[--] {name}: missing")
        return pd.DataFrame()

    # hindi_mono ships `transcription.txt`; the Kaldi sets ship `text`
    text = read_kv(find(root, "text", "transcription.txt"))
    segs = find(root, "segments")
    wavs = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
    stem2path = {os.path.splitext(os.path.basename(w))[0]: w for w in wavs}

    rows = []
    if segs:
        wavscp = {}
        scp = find(root, "wav.scp")
        if scp:
            with open(scp, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    p = ln.split()
                    hit = [t for t in p[1:] if t.lower().endswith(".wav")]
                    if len(p) >= 2 and hit:
                        wavscp[p[0]] = hit[-1]
        utt2spk = read_kv(find(root, "utt2spk"))
        if not utt2spk:
            s2u = find(root, "spk2utt")
            if s2u:
                with open(s2u, encoding="utf-8", errors="replace") as fh:
                    for ln in fh:
                        p = ln.split()
                        for u in p[1:]:
                            utt2spk[u] = p[0]
        with open(segs, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                p = ln.split()
                if len(p) != 4:
                    continue
                utt, rec = p[0], p[1]
                wav = wavscp.get(rec) or stem2path.get(rec)
                if wav and not os.path.isabs(wav):
                    wav = stem2path.get(os.path.splitext(os.path.basename(wav))[0])
                rows.append(dict(utt_id=utt, wav=wav, start=float(p[2]),
                                 end=float(p[3]), speaker=utt2spk.get(utt)))
    else:
        for stem, wav in stem2path.items():
            parts = stem.split("_")
            rows.append(dict(utt_id=stem, wav=wav, start=np.nan, end=np.nan,
                             speaker=parts[-1] if len(parts) > 1 else None))

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[--] {name}: nothing parsed")
        return df
    df["set"] = name
    df["lang"] = cfg["lang"]
    df["kind"] = cfg["kind"]
    df["transcript"] = df.utt_id.map(text)
    df["seg_dur"] = df.end - df.start
    return df


MAN = pd.concat([build(n, c) for n, c in CFG.items()], ignore_index=True)
MAN = MAN[MAN.wav.notna()].reset_index(drop=True)

print("=" * 76)
print(f"{'set':<20}{'utts':>9}{'spk':>7}{'trans':>9}{'hours':>9}")
print("-" * 76)
for name, g in MAN.groupby("set"):
    hrs = g.seg_dur.sum() / 3600 if g.seg_dur.notna().any() else float("nan")
    print(f"{name:<20}{len(g):>9,}{g.speaker.nunique():>7}"
          f"{g.transcript.notna().sum():>9,}{hrs:>9.1f}")
print("-" * 76)
print(f"{'TOTAL':<20}{len(MAN):>9,}")

# guardrails, fail loudly rather than build a broken dataset
assert MAN.transcript.notna().mean() > 0.95, "too many missing transcripts"
for a in CFG:
    for b in CFG:
        if a < b:
            ov = set(MAN[MAN.set == a].speaker.dropna()) & \
                 set(MAN[MAN.set == b].speaker.dropna())
            assert not ov, f"speaker overlap {a}/{b}: {sorted(ov)[:5]}"
print("\nOK: transcripts present, zero speaker overlap between sets.")


# CELL 5: code-switching metrics, condition, intensity bins
def tag_word(w, indic_re):
    has_i, has_e = bool(indic_re.search(w)), bool(LATIN.search(w))
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


MAN = pd.concat([MAN, MAN.apply(lambda r: cs_metrics(r.transcript, r.lang),
                                axis=1)], axis=1)


def condition(r):
    if r.n_words == 0:
        return "empty"
    # all-English is a separate condition, not a high-intensity switch
    if r.eng_ratio >= 0.999:
        return "english_only"
    if r.eng_ratio <= 0.001:
        return "indic_only"
    return "code_switched"


MAN["condition"] = MAN.apply(condition, axis=1)

BINS = [0.0, 0.10, 0.25, 0.50, 0.999]
LABELS = ["cs_low", "cs_mid", "cs_high", "cs_very_high"]
ORDER = ["mono", "cs_low", "cs_mid", "cs_high", "cs_very_high"]

MAN["cs_bin"] = pd.Series(pd.NA, index=MAN.index, dtype="object")
m = MAN.condition == "code_switched"
MAN.loc[m, "cs_bin"] = pd.cut(MAN.loc[m, "eng_ratio"], bins=BINS,
                              labels=LABELS, include_lowest=True).astype(object)
MAN.loc[MAN.condition == "indic_only", "cs_bin"] = "mono"

print("=" * 76)
print("CONDITION BREAKDOWN")
print("=" * 76)
print(pd.crosstab(MAN.set, MAN.condition).to_string())

print("\nTEST BINS (the dose-response axis), before audio filtering")
for name in ("hindi_english", "bengali_english"):
    g = MAN[(MAN.set == name) & MAN.cs_bin.notna()]
    tab = (g.groupby("cs_bin").agg(utts=("utt_id", "size"),
                                   spk=("speaker", "nunique"),
                                   med_eng=("eng_ratio", "median"))
             .reindex(ORDER).dropna(how="all"))
    print(f"\n{name}")
    print(tab.to_string())


# CELL 6: freeze the splits (speaker-disjoint, seeded)
# this seed is quoted in the paper; changing it invalidates the reported splits
RNG = np.random.RandomState(1337)
# per-speaker cap on the train source
CAP = 120

train_pool = MAN[MAN.set == "hindi_mono_train"].copy()
speakers = sorted(train_pool.speaker.dropna().unique())
RNG.shuffle(speakers)
n_dev = max(8, int(0.20 * len(speakers)))
dev_spk = set(speakers[:n_dev])

train_pool["split"] = np.where(train_pool.speaker.isin(dev_spk), "dev", "train")
capped = (train_pool.sample(frac=1.0, random_state=7)
                    .groupby("speaker", sort=False)
                    .head(CAP)
                    .copy())

# in-domain control: the official Hindi test partition, also capped
ctrl = (MAN[MAN.set == "hindi_mono"].sample(frac=1.0, random_state=7)
          .groupby("speaker", sort=False).head(100).copy())
ctrl["split"] = "test"

test = MAN[MAN.set.isin(["hindi_english", "bengali_english"])
           & MAN.cs_bin.notna()].copy()
test["split"] = "test"

FINAL = pd.concat([
    capped.assign(role="train_source"),
    ctrl.assign(role="control_indomain", cs_bin="mono"),
    test.assign(role="test_source"),
], ignore_index=True)

# filesystem-safe, unique filename per utterance
FINAL["fname"] = (FINAL.set.astype(str) + "__" +
                  FINAL.utt_id.astype(str).str.replace(r"[^A-Za-z0-9._-]", "_",
                                                       regex=True))
dup = FINAL.fname.duplicated(keep=False)
if dup.any():
    FINAL.loc[dup, "fname"] += "__" + FINAL.index[dup].astype(str)
assert FINAL.fname.is_unique

print("=" * 76)
print(f"train/dev speakers: train={len(speakers)-n_dev} dev={n_dev} overlap=0")
print(FINAL.groupby(["role", "split", "set"]).size().rename("utts").to_string())
print(f"\nTOTAL utterances to render: {len(FINAL):,}")


# CELL 7: cut, bottleneck-resample, write FLAC
# Sample-rate confound: the monolingual sets are 8 kHz and the code-switched 16 kHz, so any
# collapse measured across them could just be bandwidth. Every file goes native -> 8 kHz ->
# 16 kHz; the 8 kHz stage is a real lowpass and 16 kHz is what wav2vec2 expects.
from concurrent.futures import ThreadPoolExecutor

SR_OUT, MIN_DUR, MAX_DUR = 16000, 1.0, 20.0
OUT = "/kaggle/working/audio"
os.makedirs(OUT, exist_ok=True)


def render(rec):
    dst = os.path.join(OUT, rec.fname + ".flac")
    # resume-safe
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        try:
            return rec.fname, sf.info(dst).duration, sf.info(dst).samplerate, "cached"
        except Exception:
            pass
    try:
        info = sf.info(rec.wav)
    except Exception as e:
        return rec.fname, 0.0, 0, f"err:{type(e).__name__}"
    sr = info.samplerate
    try:
        # x == x is False only for NaN, so this skips rows with no segment bounds
        if rec.start == rec.start and rec.end == rec.end:
            a = max(0, int(float(rec.start) * sr))
            b = min(info.frames, int(float(rec.end) * sr))
            if b <= a:
                return rec.fname, 0.0, sr, "badseg"
            x, _ = sf.read(rec.wav, start=a, stop=b, dtype="float32")
        else:
            x, _ = sf.read(rec.wav, dtype="float32")
    except Exception as e:
        return rec.fname, 0.0, sr, f"err:{type(e).__name__}"

    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr
    if dur < MIN_DUR:
        return rec.fname, dur, sr, "short"
    if dur > MAX_DUR:
        return rec.fname, dur, sr, "long"

    # the bottleneck every file passes through, whatever its native rate
    if sr != 8000:
        x = soxr.resample(x, sr, 8000, quality="VHQ")
    x = soxr.resample(x, 8000, SR_OUT, quality="VHQ")
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    sf.write(dst, x, SR_OUT, format="FLAC", subtype="PCM_16")
    return rec.fname, len(x) / SR_OUT, sr, "ok"


records = list(FINAL.itertuples(index=False))
t0, done, results = time.time(), 0, []
with ThreadPoolExecutor(max_workers=8) as ex:
    for out in ex.map(render, records):
        results.append(out)
        done += 1
        if done % 1000 == 0 or done == len(records):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"{done:,}/{len(records):,} {rate:.1f} utt/s, "
                  f"eta {(len(records)-done)/max(rate,1e-9)/60:.1f} min", flush=True)

REND = pd.DataFrame(results, columns=["fname", "dur", "src_sr", "status"])
print("\nRENDER STATUS")
print(REND.status.str.split(":").str[0].value_counts().to_string())
print("\nsource sample rates seen, as evidence the confound was removed")
print(pd.merge(FINAL[["fname", "set"]], REND, on="fname")
        .pipe(lambda d: pd.crosstab(d.set, d.src_sr)).to_string())


# CELL 8: keep only what rendered, verify, write final manifest
FINAL = FINAL.merge(REND, on="fname", how="left")
KEEP = FINAL[FINAL.status.isin(["ok", "cached"])].copy()
KEEP["path"] = "audio/" + KEEP.fname + ".flac"
KEEP["label"] = "real"

dropped = len(FINAL) - len(KEEP)
print(f"dropped {dropped:,} of {len(FINAL):,} "
      f"({100*dropped/len(FINAL):.1f}%), too short, too long, or unreadable")

print("\n" + "=" * 76)
print("FINAL TEST BINS. AFTER audio filtering (this is what the paper reports)")
print("=" * 76)
for name in ("hindi_english", "bengali_english"):
    g = KEEP[KEEP.set == name]
    tab = (g.groupby("cs_bin").agg(utts=("fname", "size"),
                                   spk=("speaker", "nunique"),
                                   hours=("dur", lambda s: s.sum() / 3600),
                                   med_eng=("eng_ratio", "median"),
                                   med_cmi=("cmi", "median"))
             .reindex(ORDER).dropna(how="all").round(2))
    print(f"\n{name}")
    print(tab.to_string())
    ok = int((tab.utts >= 150).sum())
    print(f"bins with >=150 utts: {ok}/{len(tab)} "
          f"{'-> VIABLE' if ok >= 3 else '-> TOO SPARSE, REVISIT'}")

print("\n" + "=" * 76)
print("TRAIN / DEV / CONTROL")
print("=" * 76)
print(KEEP.groupby(["role", "split"]).agg(
        utts=("fname", "size"), spk=("speaker", "nunique"),
        hours=("dur", lambda s: s.sum() / 3600)).round(2).to_string())

tr = set(KEEP[(KEEP.role == "train_source") & (KEEP.split == "train")].speaker)
dv = set(KEEP[(KEEP.role == "train_source") & (KEEP.split == "dev")].speaker)
assert not (tr & dv), "train/dev speaker leak"
print(f"\nspeaker-disjoint check: train {len(tr)} / dev {len(dv)} / overlap 0 OK")

cols = ["fname", "path", "utt_id", "set", "lang", "kind", "speaker", "transcript",
        "n_words", "n_eng", "eng_ratio", "cmi", "switches", "condition",
        "cs_bin", "role", "split", "label", "dur", "src_sr"]
KEEP[cols].to_csv("/kaggle/working/manifest_real.csv", index=False)

size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
print(f"\naudio/ {len(os.listdir(OUT)):,} files, {size/2**30:.2f} GB")
print("manifest_real.csv written")


# CELL 9: listen check + README, then upload
# A pipeline nobody has listened to is not verified. This confirms the clips are speech,
# that the transcript matches, and that the 8 kHz bottleneck left them intelligible.
import IPython.display as ipd

demo = (KEEP.groupby("set", group_keys=False)
            .apply(lambda g: g.sample(min(2, len(g)), random_state=3)))
for r in demo.itertuples():
    print(f"\n[{r.set} | {r.cs_bin} | eng_ratio={r.eng_ratio:.2f} | {r.dur:.1f}s]")
    print(f"{str(r.transcript)[:110]}")
    ipd.display(ipd.Audio(os.path.join("/kaggle/working", r.path)))

with open("/kaggle/working/README.md", "w", encoding="utf-8") as fh:
    fh.write("""# CodeSwitch-Deepfake, real speech, analysis-ready

Built from MUCS / OpenSLR [103](https://openslr.org/103/) and
[104](https://openslr.org/104/), both CC BY-SA 4.0. This derived dataset is
redistributed under the same licence, with attribution to those corpora.

**Every file has been resampled native -> 8 kHz -> 16 kHz.** The monolingual
corpora are natively 8 kHz and the code-switched ones 16 kHz; forcing all four
through the same 8 kHz bottleneck removes bandwidth as a confound, so a
difference in detector error across bins cannot be explained by sample rate.

* `audio/`, {len(os.listdir(OUT)):,} mono FLAC, 16 kHz, PCM_16
* `manifest_real.csv`, one row per utterance

Key columns: `path`, `speaker`, `transcript`, `eng_ratio`, `cmi`, `switches`,
`condition` (indic_only / code_switched / english_only), `cs_bin`
(mono < cs_low < cs_mid < cs_high < cs_very_high), `role`, `split`.

Roles: `train_source` (monolingual Hindi, speaker-disjoint train/dev,
{CAP}/speaker cap, seed 1337) · `control_indomain` (official Hindi test
partition) · `test_source` (Hindi-English and Bengali-English, binned by
code-switching intensity).

All rows are `label=real`. Synthetic counterparts live in a separate dataset.
""")

print("""
Done. Publishing this output as a Dataset is what makes it reusable:
 1. Save Version -> Save & Run All (Commit). It re-runs end to end in the background.
 2. On completion, open the notebook, Output tab, New Dataset.
 3. Title codeswitch-deepfake-real, visibility private.
 4. Later notebooks reach it through + Add Input -> Datasets.
    It mounts in about 10 seconds.

From then on, a session starts with:

    import pandas as pd
    ROOT = "/kaggle/input/codeswitch-deepfake-real"
    MAN = pd.read_csv(f"{ROOT}/manifest_real.csv")
    MAN["path"] = ROOT + "/" + MAN.path

No download, no extraction, no rebuild.
""")
