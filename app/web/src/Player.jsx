import { useEffect, useRef, useState } from "react";

// The native audio element renders as a white pill on this ground and breaks the page in the
// one place a viewer is already deciding whether to trust what they see. This is the same
// element with its own controls: still one <audio>, still the browser doing the decoding.
export default function Player({ src, name }) {
  const audio = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [at, setAt] = useState(0);
  const [length, setLength] = useState(0);

  useEffect(() => {
    setPlaying(false);
    setAt(0);
    setLength(0);
  }, [src]);

  function toggle() {
    const el = audio.current;
    if (!el) return;
    if (el.paused) {
      el.play();
    } else {
      el.pause();
    }
  }

  function seek(e) {
    const el = audio.current;
    if (!el || !length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    el.currentTime = ((e.clientX - rect.left) / rect.width) * length;
  }

  const pct = length ? (at / length) * 100 : 0;

  return (
    <div className="player">
      <audio
        ref={audio}
        src={src}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => setAt(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => {
          const d = e.currentTarget.duration;
          // A blob from MediaRecorder reports Infinity until it has been seeked once.
          setLength(Number.isFinite(d) ? d : 0);
        }}
      />
      <button
        type="button"
        className="player-btn"
        onClick={toggle}
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? (
          <svg width="12" height="14" viewBox="0 0 12 14" aria-hidden="true">
            <rect x="0" y="0" width="4" height="14" fill="currentColor" />
            <rect x="8" y="0" width="4" height="14" fill="currentColor" />
          </svg>
        ) : (
          <svg width="12" height="14" viewBox="0 0 12 14" aria-hidden="true">
            <path d="M0 0 L12 7 L0 14 Z" fill="currentColor" />
          </svg>
        )}
      </button>

      <div className="player-body">
        <span className="player-name mono">{name}</span>
        <div className="player-track" onClick={seek} role="presentation">
          <span style={{ width: `${pct}%` }} />
        </div>
      </div>

      <span className="player-time mono">{clock(at)}</span>
    </div>
  );
}

function clock(s) {
  const t = Math.max(0, Math.floor(s || 0));
  return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
}
