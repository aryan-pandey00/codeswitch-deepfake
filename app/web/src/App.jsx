import { useEffect, useRef, useState } from "react";
import Recorder from "./Recorder.jsx";
import Verdict from "./Verdict.jsx";
import Waveform from "./Waveform.jsx";
import Spectrogram from "./Spectrogram.jsx";
import Player from "./Player.jsx";
import { checkHealth, predict, OFFLINE_EXAMPLE } from "./api.js";

// Every figure here is measured and traceable to a notebook. Nothing on this page is a
// placeholder, which is the reason the mock's dashboard and history sections were dropped
// rather than restyled.
const STATS = [
  { value: "0.11%", label: "genuine speech wrongly rejected by the shipped model" },
  { value: "29.2%", label: "rejected by the same model trained on one corpus" },
  { value: "31,922", label: "clips in the evaluation, speaker-disjoint" },
  { value: "3", label: "detector architectures compared" },
];

const PIPELINE = [
  {
    title: "Audio in",
    body: "A file or a live microphone recording. Anything ffmpeg can decode, up to two minutes.",
  },
  {
    title: "Bandwidth match",
    body: "Mono at 16 kHz, through the same 8 kHz bottleneck the models were trained behind. Skipping it would put the clip out of distribution.",
  },
  {
    title: "XLS-R features",
    body: "Twenty transformer blocks of a frozen wav2vec2-XLS-R, pooled to a mean and a standard deviation over frames.",
  },
  {
    title: "Two heads, one input",
    body: "The same features scored by two logistic heads. One trained on a single corpus, one with eight extra speakers per corpus.",
  },
  {
    title: "Windowed verdict",
    body: "Five-second spans at a four-second hop, averaged as logits. The threshold was calibrated in domain, never on the clip being scored.",
  },
];

// Mirrors MAX_UPLOAD_MB and MAX_WINDOWS in the API. Duplicated rather than fetched because
// the check has to happen before any request is made.
const MAX_UPLOAD_MB = 10;
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
const MAX_WINDOWS = 8;

const FINDINGS = [
  { arch: "XLS-R, frozen + linear head", capacity: "nothing fitted here", before: "29.2%", after: "0.11%" },
  { arch: "CNN-BiLSTM over MFCCs", capacity: "504k trained from scratch", before: "65.7%", after: "0.27%" },
  { arch: "XLS-R + LoRA", capacity: "657k adapters on a 265M encoder", before: "84.7%", after: "0.27%" },
];

export default function App() {
  const [status, setStatus] = useState("connecting");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [clipUrl, setClipUrl] = useState(null);
  const [picked, setPicked] = useState(null);
  const [tab, setTab] = useState("upload");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef(null);

  // Also warms a cold container, so the first real request is not the one that pays for it.
  useEffect(() => {
    checkHealth()
      .then((h) => setStatus(h.mode === "stub" ? "stub" : "ready"))
      .catch(() => setStatus("offline"));
  }, []);

  useEffect(() => {
    return () => clipUrl && URL.revokeObjectURL(clipUrl);
  }, [clipUrl]);

  useEffect(() => {
    const nav = document.querySelector("nav.topbar");
    const onScroll = () => nav?.classList.toggle("scrolled", window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  async function submit(blob, filename) {
    setError(null);

    // Checked before the upload starts, not after. The server enforces the same ceiling, but
    // reaching it means the whole file has already crossed the network, so an oversized clip
    // spends a minute uploading only to be refused on arrival.
    if (blob.size > MAX_UPLOAD_BYTES) {
      setPicked(filename);
      setResult(null);
      setError(
        `${(blob.size / 1024 / 1024).toFixed(1)} MB is over the ${MAX_UPLOAD_MB} MB limit. ` +
          `Only ${MAX_WINDOWS} spans spread across the clip are scored, so a trimmed 30 seconds ` +
          `gives the same verdict as the whole file.`
      );
      return;
    }

    setBusy(true);
    setResult(null);
    setPicked(filename);

    if (clipUrl) URL.revokeObjectURL(clipUrl);
    setClipUrl(URL.createObjectURL(blob));

    try {
      setResult(await predict(blob, filename));
    } catch (err) {
      if (status === "offline") {
        setResult(OFFLINE_EXAMPLE);
      } else {
        setError(err.message);
      }
    } finally {
      setBusy(false);
    }
  }

  function onFile(e) {
    const file = e.target.files?.[0];
    if (file) submit(file, file.name);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && !busy) submit(file, file.name);
  }

  // The contrast is the demo. Highlighting the pair only when they disagree keeps that from
  // reading as decoration on clips where both models agree.
  const verdicts = result?.verdicts ?? [];
  const disagree = new Set(verdicts.map((v) => v.label)).size > 1;

  return (
    <>
      <nav className="topbar">
        <div className="nav-inner">
          <a className="brand" href="#top">
            <Logo />
            VaaniRaksha
          </a>
          <div className="nav-links">
            <a href="#demo">Analyze</a>
            <a href="#pipeline">How it works</a>
            <a href="#findings">What we found</a>
          </div>
          <StatusPill status={status} />
        </div>
      </nav>

      <header className="hero" id="top">
        <div className="hero-inner">
          <div className="hero-text">
            <h1>Know if a voice is real before you trust it.</h1>
            <p className="hero-sub">
              Two detectors score the same recording. One was trained on a single corpus, the way
              almost every published detector is. The other saw eight extra speakers recorded
              somewhere else. On genuine speech from an unfamiliar microphone they disagree, and
              that disagreement is the finding.
            </p>
            <div className="hero-actions">
              <a href="#demo" className="btn btn-primary">
                Analyze a recording
              </a>
              <a href="#findings" className="btn btn-ghost">
                See the results
              </a>
            </div>
          </div>
          <Spectrogram />
        </div>

        <Waveform />

        <div className="stat-row">
          {STATS.map((s) => (
            <div className="stat" key={s.label}>
              <span className="stat-value">{s.value}</span>
              <span className="stat-label">{s.label}</span>
            </div>
          ))}
        </div>
      </header>

      <main>
        <section className="demo" id="demo">
          <div className="section-head">
            <h2>Run a recording through both models</h2>
            <p>
              Upload a clip or record one. Hindi, English or a mix of the two is what these
              models were built for. Nothing is stored. The audio is scored and discarded.
            </p>
          </div>

          <div className="panel">
            <div className="input-tabs" role="tablist">
              <button
                className={`tab ${tab === "upload" ? "active" : ""}`}
                role="tab"
                aria-selected={tab === "upload"}
                onClick={() => setTab("upload")}
              >
                Upload a file
              </button>
              <button
                className={`tab ${tab === "record" ? "active" : ""}`}
                role="tab"
                aria-selected={tab === "record"}
                onClick={() => setTab("record")}
              >
                Record now
              </button>
            </div>

            {tab === "upload" ? (
              <label
                className={`upload-zone ${dragging ? "drag" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <input
                  ref={fileInput}
                  type="file"
                  accept="audio/*"
                  onChange={onFile}
                  disabled={busy}
                  hidden
                />
                <UploadIcon />
                <span className="upload-title">Drop an audio file here</span>
                <span className={`upload-sub ${picked ? "picked" : ""}`}>
                  {picked ?? "or click to browse, up to 2 minutes, 10 MB"}
                </span>
              </label>
            ) : (
              <Recorder onClip={submit} disabled={busy} />
            )}

            {clipUrl && <Player src={clipUrl} name={picked ?? "clip"} />}

            {busy && (
              <div className="scan">
                <div className="scan-bar">
                  <span />
                </div>
                <p className="hint">Scoring both models</p>
              </div>
            )}
            {error && <p className="hint error">{error}</p>}

            {result && (
              <>
                {result.mode !== "onnx" && <NotLive mode={result.mode} />}
                {disagree && <Contrast />}
                <div className="verdicts">
                  {verdicts.map((v) => (
                    <Verdict
                      key={v.id}
                      verdict={v}
                      highlight={disagree && v.id === "fixed"}
                      windowSeconds={result.window_s}
                    />
                  ))}
                </div>
                {/* The example's duration and sample rate describe a clip nobody uploaded, and
                    showing them beside the player reading a different length invites the reader
                    to trust one of two numbers that disagree. Stub mode measured a real file, so
                    it keeps its row. */}
                {result.mode !== "example" && <Meta result={result} />}
              </>
            )}
          </div>
        </section>

        <section id="pipeline">
          <div className="section-head">
            <h2>What happens after you hit analyze</h2>
            <p>
              The frontend is frozen and shared. The only difference between the two verdicts is
              which speakers the classifier on top of it was trained on.
            </p>
          </div>
          <ol className="pipeline-steps">
            {PIPELINE.map((s, i) => (
              <li key={s.title}>
                <span className="step-index">{String(i + 1).padStart(2, "0")}</span>
                <h3>{s.title}</h3>
                <p>{s.body}</p>
              </li>
            ))}
          </ol>
        </section>

        <section id="findings">
          <div className="section-head">
            <h2>A detector learns its microphone, not the forgery</h2>
            <p>
              Train a detector on one corpus and it reaches zero error there. Move it to speech
              recorded somewhere else and it still catches every fake, while rejecting a large
              share of real people. Three architectures, the same result, and the damage grows
              with how much of the model was fitted to the training corpus.
            </p>
          </div>

          <div className="findings-table">
            <table className="results">
              <thead>
                <tr>
                  <th>Architecture</th>
                  <th>Fitted to the training corpus</th>
                  <th>Real speech rejected</th>
                  <th>After the fix</th>
                </tr>
              </thead>
              <tbody>
                {FINDINGS.map((f) => (
                  <tr key={f.arch} className={f.arch.startsWith("XLS-R, frozen") ? "ours" : ""}>
                    <td>{f.arch}</td>
                    <td>{f.capacity}</td>
                    <td className="num">{f.before}</td>
                    <td className="num">{f.after}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="findings-note">
            Every one of them missed almost no fakes. The error is entirely in the other
            direction, calling real people synthetic. The fix is not a better architecture: it is
            eight speakers per corpus added to the training set, roughly a 7% increase in data.
            The highlighted row is the model this page serves.
          </p>
        </section>
      </main>

      <footer className="site-footer">
        <div className="footer-top">
          <div className="footer-brand">
            <a className="brand" href="#top">
              <Logo />
              VaaniRaksha
            </a>
            <p>
              A research prototype. Not a forensic instrument. A single verdict is evidence of
              nothing on its own, and the models are known to be sensitive to recording
              conditions they have not seen. They detect synthesis in a file; they do not
              detect a deepfake replayed through a speaker, which arrives carrying a real
              microphone and reads as genuine.
            </p>
          </div>
          <div className="footer-col">
            <h5>Built on</h5>
            <ul>
              <li>wav2vec2-XLS-R, frozen to 20 blocks</li>
              <li>MUCS / OpenSLR 103 and 104</li>
              <li>XTTS-v2 and FreeVC for the synthetic half</li>
              <li>int8 ONNX, scored on CPU</li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <div className="footer-bottom-inner">
            <span className="mono">Audio is scored in memory and never stored.</span>
            <span className="mono">Hindi · Bengali · Indian English</span>
          </div>
        </div>
      </footer>
    </>
  );
}

function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">
      <rect className="logo-frame" x="2" y="2" width="28" height="28" rx="8" strokeWidth="1.6" />
      <path
        className="logo-wave"
        d="M7 19 L11 19 L13.5 12 L17 24 L19.5 8 L22 19 L25 19"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M12 16V4m0 0L8 8m4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" />
    </svg>
  );
}

function StatusPill({ status }) {
  const text = {
    connecting: "Connecting",
    ready: "Service live",
    stub: "No model loaded",
    offline: "Service unreachable",
  }[status];

  return (
    <span className={`status-pill ${status}`} title={text}>
      <i />
      {text}
    </span>
  );
}

// Two cards contradicting each other reads as a malfunction unless something says otherwise
// at the moment it happens. It cannot claim which one is right, because an arbitrary upload
// has no ground truth here, only explain why they can differ.
function Contrast() {
  return (
    <p className="contrast">
      <strong>These are two different models, and they disagree.</strong> The baseline was
      trained on a single corpus and tends to call unfamiliar recordings fake even when they are
      genuine speech. The second one saw a handful of speakers from other recording conditions.
      Same audio, same frontend, eight speakers of difference.
    </p>
  );
}

// Loud on purpose. A cold container or a dropped network otherwise puts confident numbers on
// screen that no model produced, and one grey line of status text is not enough to stop
// someone reading them as a result.
function NotLive({ mode }) {
  const why =
    mode === "stub"
      ? "The service is running but has no model loaded, so these scores are placeholders."
      : "The service could not be reached, so this is a stored example of what it returns.";

  return (
    <aside className="notlive" role="alert">
      <strong>Example, not a live result</strong>
      <span>{why}</span>
    </aside>
  );
}

function Meta({ result }) {
  return (
    <dl className="meta">
      <div>
        <dt>Clip</dt>
        <dd>{result.duration_s}s</dd>
      </div>
      <div>
        <dt>Uploaded at</dt>
        <dd>{(result.sample_rate / 1000).toFixed(1)} kHz</dd>
      </div>
      <div>
        <dt>Scored</dt>
        <dd>{result.scored_s}s at 16 kHz</dd>
      </div>
      <div>
        <dt>Windows</dt>
        <dd>{result.n_windows ?? 1}</dd>
      </div>
      <div>
        <dt>Latency</dt>
        <dd>{result.latency_ms} ms</dd>
      </div>
    </dl>
  );
}
