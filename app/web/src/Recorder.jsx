import { useEffect, useRef, useState } from "react";

const MAX_SECONDS = 30;

export default function Recorder({ onClip, disabled }) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState(null);
  const recorder = useRef(null);
  const chunks = useRef([]);
  const timer = useRef(null);
  const canvas = useRef(null);
  const raf = useRef(0);
  const audioCtx = useRef(null);

  // A recorder left running holds the microphone open after the component goes away, and the
  // browser keeps showing the recording indicator.
  useEffect(() => {
    return () => {
      clearInterval(timer.current);
      cancelAnimationFrame(raf.current);
      audioCtx.current?.close();
      const r = recorder.current;
      if (r && r.state !== "inactive") r.stop();
    };
  }, []);

  // The meter is not decoration here. Somebody recording into a muted or wrong input otherwise
  // gets a confident verdict on silence, and a flat line is the only warning they will notice
  // before they have already uploaded it.
  function meter(stream) {
    const el = canvas.current;
    if (!el) return;

    const ctx = el.getContext("2d");
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.current = ac;
    const analyser = ac.createAnalyser();
    analyser.fftSize = 1024;
    ac.createMediaStreamSource(stream).connect(analyser);
    const buf = new Uint8Array(analyser.fftSize);

    const history = [];
    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const { width, height } = el.getBoundingClientRect();
      el.width = Math.max(1, Math.floor(width * dpr));
      el.height = Math.max(1, Math.floor(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      analyser.getByteTimeDomainData(buf);
      let peak = 0;
      for (let i = 0; i < buf.length; i++) {
        peak = Math.max(peak, Math.abs(buf[i] - 128) / 128);
      }
      history.push(peak);
      const columns = Math.floor(width / 4);
      while (history.length > columns) history.shift();

      ctx.clearRect(0, 0, width, height);
      const mid = height / 2;
      ctx.strokeStyle = "rgba(243, 241, 236, 0.08)";
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(width, mid);
      ctx.stroke();

      history.forEach((v, i) => {
        const h = Math.max(1, v * height * 0.44);
        ctx.fillStyle = `rgba(255, 59, 78, ${(0.3 + v * 0.7).toFixed(3)})`;
        ctx.fillRect(i * 4, mid - h, 2, h * 2);
      });

      raf.current = requestAnimationFrame(draw);
    };
    draw();
  }

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const r = new MediaRecorder(stream);
      chunks.current = [];

      r.ondataavailable = (e) => e.data.size > 0 && chunks.current.push(e.data);
      r.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        cancelAnimationFrame(raf.current);
        audioCtx.current?.close();
        audioCtx.current = null;
        const blob = new Blob(chunks.current, { type: r.mimeType });
        onClip(blob, "recording.webm");
      };

      r.start();
      recorder.current = r;
      setRecording(true);
      setSeconds(0);
      meter(stream);

      timer.current = setInterval(() => {
        setSeconds((s) => {
          if (s + 1 >= MAX_SECONDS) stop();
          return s + 1;
        });
      }, 1000);
    } catch {
      setError("Microphone permission denied, or no input device available.");
    }
  }

  function stop() {
    clearInterval(timer.current);
    setRecording(false);
    const r = recorder.current;
    if (r && r.state !== "inactive") r.stop();
  }

  const clock = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;

  return (
    <div className="record-stage">
      <canvas ref={canvas} aria-hidden="true" />
      <button
        type="button"
        className={`record-btn ${recording ? "recording" : ""}`}
        onClick={recording ? stop : start}
        disabled={disabled}
        aria-label={recording ? "Stop recording" : "Start recording"}
      >
        <span className="dot" />
      </button>
      <span className="record-time mono">{clock}</span>
      <p className="hint">
        {recording
          ? `Recording, stops automatically at ${MAX_SECONDS} seconds.`
          : "Tap to record from your microphone. Speak normally, Hindi or English or both."}
      </p>
      {/* The recorder invites the one test these models cannot answer. Holding a phone up to a
          speaker swaps the synthetic recording chain for a real microphone and a real room, and
          these detectors were trained to separate synthetic files from real files rather than
          to hear a loudspeaker. A replayed deepfake comes back genuine, reliably. The button is
          the only place a warning arrives before the mistake rather than after it. */}
      {!recording && (
        <p className="hint caution">
          Record a live voice. Holding the microphone up to a speaker playing a suspected clip
          does not work: re-recording gives synthetic audio a genuine microphone and room, and
          these models read that as real. Upload the file itself instead.
        </p>
      )}
      {error && <p className="hint error">{error}</p>}
    </div>
  );
}
