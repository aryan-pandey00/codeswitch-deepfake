import { useEffect, useRef } from "react";

// The hero panel. It shows the one constraint the whole project turns on: every file these
// models have ever seen was bandlimited to 4 kHz, so the upper half of the spectrum is empty by
// construction. A generic audio visual would have said "this page is about sound"; this says
// what kind of sound, and why the detector is fragile.
//
// The energy is synthesised rather than measured. It is labelled as a reference readout and
// makes no claim to be a live signal.
const NYQUIST = 8000;
const CUTOFF = 4000;
const BINS = 48;
const COLUMN_W = 5;

export default function Spectrogram() {
  const canvas = useRef(null);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;

    const ctx = el.getContext("2d");
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let t = 0;
    const columns = [];

    function size() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const { width, height } = el.getBoundingClientRect();
      el.width = Math.max(1, Math.floor(width * dpr));
      el.height = Math.max(1, Math.floor(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { width, height };
    }

    let box = size();
    const onResize = () => {
      box = size();
      if (still) render();
    };
    window.addEventListener("resize", onResize);

    // Speech-like: energy concentrated low, a couple of drifting formants, nothing above the
    // cutoff bin. Voiced and unvoiced stretches alternate so it does not read as a test tone.
    function column(phase) {
      const voiced = Math.sin(phase * 0.6) > -0.35;
      const f1 = 6 + Math.sin(phase * 0.9) * 3;
      const f2 = 15 + Math.sin(phase * 0.5 + 1.2) * 5;
      const cutoffBin = Math.floor((CUTOFF / NYQUIST) * BINS);

      return Array.from({ length: BINS }, (_, i) => {
        if (i >= cutoffBin) return 0;
        const tilt = Math.exp(-i / 14);
        const peaks =
          Math.exp(-((i - f1) ** 2) / 8) * 0.9 + Math.exp(-((i - f2) ** 2) / 14) * 0.55;
        const noise = 0.16 + Math.sin(i * 2.3 + phase * 4) * 0.06;
        return Math.max(0, Math.min(1, (voiced ? tilt * 0.5 + peaks : tilt * 0.3) + noise * 0.4));
      });
    }

    function render() {
      const { width, height } = box;
      ctx.clearRect(0, 0, width, height);

      const cols = Math.ceil(width / COLUMN_W);
      // Backfilled so the panel is already a plausible readout on the first frame rather than
      // filling in from empty while someone reads the headline.
      while (columns.length < cols) columns.unshift(column(t - columns.length * 0.12));
      while (columns.length > cols) columns.shift();

      const binH = height / BINS;
      columns.forEach((bins, x) => {
        bins.forEach((v, i) => {
          if (v <= 0.02) return;
          // Low energy sits dim and violet-leaning, peaks push toward the signal red, so the
          // formants read without a second hue entering the palette.
          const a = Math.pow(v, 1.5);
          const r = 120 + a * 135;
          const g = 30 + a * 29;
          const b = 58 + a * 20;
          ctx.fillStyle = `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${(0.12 + a * 0.85).toFixed(3)})`;
          ctx.fillRect(x * COLUMN_W, height - (i + 1) * binH, COLUMN_W - 0.5, binH - 0.5);
        });
      });

      const y = height - (CUTOFF / NYQUIST) * height;
      ctx.strokeStyle = "rgba(243, 241, 236, 0.32)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "rgba(243, 241, 236, 0.55)";
      ctx.font = "500 10px 'IBM Plex Mono', monospace";
      ctx.fillText("4 kHz ceiling", 10, y - 7);
    }

    if (still) {
      render();
    } else {
      const loop = () => {
        t += 0.12;
        columns.push(column(t));
        columns.shift();
        render();
        raf = requestAnimationFrame(loop);
      };
      loop();
    }

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return (
    <figure className="spectro">
      <canvas ref={canvas} aria-hidden="true" />
      <figcaption>
        <span className="mono">reference readout</span>
        Everything these models have heard was bandlimited to 4 kHz. The band above it is empty
        by construction, and a clip that arrives full of it is out of distribution.
      </figcaption>
    </figure>
  );
}
