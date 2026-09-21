const BASE = import.meta.env.VITE_API_URL ?? "";

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`, { method: "GET" });
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  return res.json();
}

export async function predict(blob, filename = "clip.wav") {
  const body = new FormData();
  body.append("file", blob, filename);

  const res = await fetch(`${BASE}/predict`, { method: "POST", body });
  if (!res.ok) {
    // FastAPI puts the reason in `detail`, which is worth surfacing since the common
    // failures here are things the user can act on, like a clip under one second.
    let detail = `request failed: ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// Shown when the API cannot be reached. A portfolio link that renders a worked example still
// demonstrates the project; a dead link does not.
export const OFFLINE_EXAMPLE = {
  duration_s: 22.4,
  scored_s: 22.4,
  sample_rate: 44100,
  n_windows: 6,
  window_s: 5,
  hop_s: 4,
  latency_ms: 0,
  mode: "example",
  verdicts: [
    {
      id: "baseline",
      score: 0.91,
      label: "fake",
      title: "Baseline",
      trained_on: "one corpus, monolingual Hindi",
      note: "0% error in domain",
      windows: [0.71, 0.94, 0.83, 0.97, 0.62, 0.95],
    },
    {
      id: "fixed",
      score: 0.04,
      label: "real",
      title: "Ours",
      trained_on: "same, plus a few speakers from other recording conditions",
      note: "the fix this project measured",
      windows: [0.05, 0.02, 0.11, 0.03, 0.18, 0.04],
    },
  ],
};
