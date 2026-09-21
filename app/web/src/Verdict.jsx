// Distance from the decision boundary, not a probability.
//
// The head behind these verdicts separates its training corpus at 0.00% equal error rate, so
// |margin| runs past 6 on most clips, and sigmoid(-6) already rounds to "100% confident". Three
// quarters of a 45-clip acceptance test would print exactly 100, which tells a viewer nothing
// and claims more than the model knows. Out of corpus the calibration is measurably wrong while
// the ranking survives, which is the central result of this project, so a percentage would be
// the one number the work itself says not to trust.
const BANDS = [
  { at: 6, label: "well clear of the boundary" },
  { at: 3, label: "clear of the boundary" },
  { at: 1.5, label: "modest margin" },
  { at: 0, label: "borderline, close to the boundary" },
];

function band(margin) {
  const m = Math.abs(margin);
  return BANDS.find((b) => m >= b.at).label;
}

// Saturates rather than scaling linearly, so the bar still moves in the range that matters.
function fill(margin) {
  return Math.min(100, Math.round((Math.abs(margin) / 8) * 100));
}

export default function Verdict({ verdict, highlight, windowSeconds }) {
  const { label, score, margin, title, trained_on, note, windows } = verdict;

  // Older responses carry no margin. Recovering it from the score is lossy once the sigmoid
  // has saturated, but it keeps the card readable against an API that has not been redeployed.
  const m = margin ?? Math.log(Math.min(Math.max(score, 1e-6), 1 - 1e-6) / (1 - Math.min(Math.max(score, 1e-6), 1 - 1e-6)));

  return (
    <article className={`verdict ${label} ${highlight ? "highlight" : ""}`}>
      <header>
        <h3>{title ?? verdict.id}</h3>
        <p className="trained">{trained_on}</p>
      </header>

      <p className="call">{label === "fake" ? "Deepfake" : "Genuine"}</p>

      <div className="meter" role="img" aria-label={`margin ${m.toFixed(1)}, ${band(m)}`}>
        <span style={{ width: `${fill(m)}%` }} />
      </div>
      <p className="confidence">
        margin {m > 0 ? "+" : ""}{m.toFixed(1)}, {band(m)}
      </p>

      {windows?.length > 1 && <Timeline windows={windows} seconds={windowSeconds} />}

      {note && <p className="note">{note}</p>}
    </article>
  );
}

// The clip is scored in spans rather than whole, so the verdict is an average of several
// independent reads. Showing them turns one opaque number into something a viewer can argue
// with, and makes a clip that changes partway through visible instead of averaged away.
function Timeline({ windows, seconds }) {
  return (
    <figure className="timeline">
      <div className="bars">
        {windows.map((w, i) => (
          <span
            key={i}
            className={w > 0.5 ? "fake" : "real"}
            style={{ height: `${20 + Math.abs(w - 0.5) * 160}%` }}
            title={`window ${i + 1}: ${w > 0.5 ? "fake" : "real"} side`}
          />
        ))}
      </div>
      <figcaption>
        {windows.length} windows{seconds ? ` of ${seconds}s` : ""}, averaged
      </figcaption>
    </figure>
  );
}
