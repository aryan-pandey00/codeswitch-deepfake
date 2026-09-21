import { useEffect, useRef } from "react";

// The strip under the hero. Decorative, but it is the one thing on the page that says "audio"
// before any text is read, so it earns its place. It draws a travelling waveform rather than a
// real signal and makes no claim to be one.
export default function Waveform() {
  const canvas = useRef(null);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;

    const ctx = el.getContext("2d");
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let t = 0;

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
      if (still) draw();
    };
    window.addEventListener("resize", onResize);

    // Three detuned components, so the envelope drifts instead of repeating on a visible beat.
    function amplitude(x, phase) {
      return (
        Math.sin(x * 0.021 + phase) * 0.55 +
        Math.sin(x * 0.047 - phase * 1.7) * 0.28 +
        Math.sin(x * 0.011 + phase * 0.4) * 0.17
      );
    }

    function draw() {
      const { width, height } = box;
      const mid = height / 2;
      ctx.clearRect(0, 0, width, height);

      const step = 3;
      for (let x = 0; x < width; x += step) {
        // Taper at both ends so the strip fades into the page rather than being cut off.
        const edge = Math.min(1, Math.min(x, width - x) / 90);
        const a = amplitude(x, t) * edge;
        const h = Math.abs(a) * (height * 0.36) + 1;
        const alpha = 0.22 + Math.abs(a) * 0.5;
        ctx.fillStyle = `rgba(255, 59, 78, ${alpha.toFixed(3)})`;
        ctx.fillRect(x, mid - h, 1.5, h * 2);
      }

      ctx.strokeStyle = "rgba(243, 241, 236, 0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(width, mid);
      ctx.stroke();
    }

    if (still) {
      draw();
    } else {
      const loop = () => {
        t += 0.012;
        draw();
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
    <div className="signal-strip" aria-hidden="true">
      <canvas ref={canvas} />
    </div>
  );
}
