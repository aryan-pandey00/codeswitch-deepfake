"""Detection API for the code-switched deepfake demo.

Scores one upload with two detectors and returns both. The baseline is trained on a single
corpus and the fixed model adds a handful of speakers from other recording conditions. Showing
them side by side is the point of the demo: on genuine speech recorded outside the training
corpus the baseline says fake and the fixed model says real, which is the paper's finding made
visible without a slide.

Both detectors share one frozen frontend, so a request costs one forward pass and two dot
products. The heads are logistic regressions over pooled features, stored as plain arrays.

Preprocessing is not negotiable. Every training file passed through native to 8 kHz to 16 kHz,
so the models have never seen energy above 4 kHz. An upload from a phone arrives at 44.1 kHz
full band, and feeding that in unchanged puts the input out of distribution in exactly the way
this project measured. The same bottleneck is applied here.

Requirements
    pip install modal, then `modal setup` once.
    Deploy with: modal deploy service.py
    Model artifacts in MODEL_DIR. Absent, the service answers with a stub and says so, which
    keeps the frontend demonstrable while the export is still being produced.
"""

import modal

APP_NAME = "codeswitch-deepfake-api"
MAX_UPLOAD_MB = 10

# Generous, because compute is bounded by the window cap below rather than by clip length.
# What this guards is decoding and resampling, which are cheap but not free.
MAX_SECONDS = 120
BOTTLENECK_HZ = 8000
TARGET_SR = 16000
MODEL_DIR = "/models"

# Features are a mean and a standard deviation over frames, and the training utterances average
# under five seconds. A sixty second upload pooled whole is a different statistic, not a longer
# one, so the clip is cut into training-length spans and the spans are scored.
WINDOW_S = 5.0
HOP_S = 4.0

# A ceiling on work per request, so latency cannot grow with upload length.
MAX_WINDOWS = 8

# Measured in notebook 14: about 430 ms of compute per second of audio, per core.
CORES = 4

ALLOWED_ORIGINS = ["*"]

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "fastapi[standard]",
        "python-multipart",
        "soundfile",
        "soxr",
        "numpy",
        "onnxruntime",
    )
)

models = modal.Volume.from_name("codeswitch-deepfake-models", create_if_missing=True)

with image.imports():
    import io
    import json
    import os
    import subprocess
    import tempfile
    import time

    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    import soxr


def decode(raw):
    """Return mono float32 audio and its sample rate.

    Browsers record WebM or MP4 rather than WAV, and libsndfile reads neither, so anything it
    rejects is handed to ffmpeg before giving up.
    """
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".bin") as src:
            src.write(raw)
            src.flush()
            out = subprocess.run(
                ["ffmpeg", "-v", "quiet", "-i", src.name, "-f", "wav", "-ac", "1",
                 "-ar", str(TARGET_SR), "pipe:1"],
                capture_output=True,
            )
        if out.returncode != 0 or not out.stdout:
            raise ValueError("unsupported audio format")
        audio, sr = sf.read(io.BytesIO(out.stdout), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def bottleneck(x, sr):
    """The sample rate path every training file took."""
    return soxr.resample(soxr.resample(x, sr, BOTTLENECK_HZ), BOTTLENECK_HZ, TARGET_SR)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def windows(audio):
    """Cut into spans of the length the model was trained on.

    Anything shorter than one window is returned whole, which is the old behaviour and still
    correct. The final window is anchored to the end of the clip rather than started on the
    grid, so the tail is never dropped.
    """
    n, hop = int(WINDOW_S * TARGET_SR), int(HOP_S * TARGET_SR)
    if len(audio) <= n:
        return [audio]

    starts = list(range(0, len(audio) - n + 1, hop))
    if starts[-1] + n < len(audio):
        starts.append(len(audio) - n)
    spans = [audio[s: s + n] for s in starts]

    # Each span is normalised to unit variance on its own, so a span of silence between two
    # sentences would be amplified into noise and scored as if it were speech.
    ref = rms(audio)
    loud = [s for s in spans if rms(s) >= 0.1 * ref] or spans

    if len(loud) > MAX_WINDOWS:
        # Spread rather than truncated, so a long clip is still sampled across its whole length.
        step = (len(loud) - 1) / (MAX_WINDOWS - 1)
        loud = [loud[int(round(i * step))] for i in range(MAX_WINDOWS)]
    return loud


class Detector:
    """One frozen frontend, several logistic heads over pooled features."""

    def __init__(self, model_dir):
        self.ready = False
        graph = os.path.join(model_dir, "frontend_int8.onnx")
        heads = os.path.join(model_dir, "heads.json")
        if not (os.path.exists(graph) and os.path.exists(heads)):
            return

        # Matched to the cores the container reserves. Fewer wastes them, more thrashes.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = CORES
        self.session = ort.InferenceSession(graph, opts, providers=["CPUExecutionProvider"])

        with open(heads) as fh:
            spec = json.load(fh)
        self.heads = {
            name: {k: np.asarray(v, dtype=np.float32) if isinstance(v, list) else v
                   for k, v in head.items()}
            for name, head in spec["heads"].items()
        }
        self.meta = spec.get("meta", {})
        self.ready = True

    def features(self, audio):
        """Mean and standard deviation over frames, matching how the heads were trained."""
        x = audio.astype(np.float32)
        # Written the way the training feature extractor wrote it, epsilon inside the root.
        # The difference is tiny and the point of matching it is that nobody has to wonder.
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        hidden = self.session.run(None, {"input_values": x[None, :]})[0][0]
        return np.concatenate([hidden.mean(0), hidden.std(0)])

    def score(self, audio):
        feats = [self.features(w) for w in windows(audio)]
        out = {}
        for name, head in self.heads.items():
            z = np.array([float(((f - head["mu"]) / head["sd"] * head["w"]).sum() + head["b"])
                          for f in feats])
            # Averaged before the sigmoid, not after: one saturated window would otherwise
            # pull a mean of already-squashed numbers further than its evidence deserves.
            # The margin travels alongside it because sigmoid(-6) already rounds to 100%, so
            # the score alone cannot tell a comfortable call from an overwhelming one, and out
            # of corpus the calibration is wrong anyway.
            out[name] = {"score": sigmoid(z.mean()),
                         "margin": float(z.mean()),
                         "windows": [sigmoid(v) for v in z]}
        return out


def sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-float(z))))


def stub_scores(audio):
    """Used only when no model artifacts are mounted.

    Values derive from the signal rather than being constant, so a score that varies between
    clips shows the audio reached this point instead of decoding having failed upstream.
    """
    per = [float(np.clip(np.abs(w).mean() * 4, 0, 1)) for w in windows(audio)]
    mean = sum(per) / len(per)
    return {"baseline": {"score": mean, "windows": per},
            "fixed": {"score": mean * 0.3, "windows": [p * 0.3 for p in per]}}


# Not held warm. Four cores kept alive around the clock costs several times the monthly credit,
# while four cores used only while a request is running costs cents for a demo session. The
# frontend calls /health on page load, which starts the container while the page is still being
# read, and the scaledown window keeps it up for the rest of a sitting.
@app.function(image=image, volumes={MODEL_DIR: models}, cpu=CORES,
              min_containers=0, scaledown_window=900)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    detector = Detector(MODEL_DIR)

    web = FastAPI(title=APP_NAME)
    web.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Descriptions travel with the scores so the interface does not hardcode claims about how
    # each model was trained.
    LABELS = {
        # The notes carry the measured out-of-corpus rates rather than the in-domain ones. A
        # baseline card reading "0% error in domain" is true and reads as an endorsement of the
        # model the rest of the page argues against.
        "baseline": {
            "title": "Baseline",
            "trained_on": "one corpus, monolingual Hindi",
            "note": "rejects 29.2% of genuine speech recorded elsewhere",
        },
        "fixed": {
            "title": "Ours",
            "trained_on": "same, plus eight speakers per corpus from other recording conditions",
            "note": "0.11% rejected, and no fakes missed",
        },
    }

    @web.get("/health")
    def health():
        return {
            "status": "ok",
            "mode": "onnx" if detector.ready else "stub",
            "models": list(detector.heads) if detector.ready else list(LABELS),
        }

    @web.post("/predict")
    async def predict(file: UploadFile = File(...)):
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"file exceeds {MAX_UPLOAD_MB} MB")

        t0 = time.time()
        try:
            audio, sr = decode(raw)
        except Exception as exc:
            raise HTTPException(415, f"could not decode audio: {exc}")

        if len(audio) < sr:
            raise HTTPException(422, "clip is shorter than one second")

        original_s = len(audio) / sr
        if original_s > MAX_SECONDS:
            audio = audio[: int(MAX_SECONDS * sr)]

        prepared = bottleneck(audio, sr)
        scores = detector.score(prepared) if detector.ready else stub_scores(prepared)
        n_windows = len(next(iter(scores.values()))["windows"])

        return {
            "duration_s": round(original_s, 2),
            "scored_s": round(len(prepared) / TARGET_SR, 2),
            "sample_rate": sr,
            "n_windows": n_windows,
            "window_s": WINDOW_S,
            "hop_s": HOP_S,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "mode": "onnx" if detector.ready else "stub",
            "verdicts": [
                {
                    "id": name,
                    "score": round(v["score"], 4),
                    "margin": round(v["margin"], 3),
                    "label": "fake" if v["score"] > 0.5 else "real",
                    # Per window, so the interface can show where in a clip the detector reacted
                    # rather than only what it concluded.
                    "windows": [round(w, 4) for w in v["windows"]],
                    **LABELS.get(name, {}),
                }
                for name, v in scores.items()
            ],
        }

    return web
