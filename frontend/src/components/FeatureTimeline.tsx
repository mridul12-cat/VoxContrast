import React, { useRef, useEffect } from "react";
import { TimeSeriesData } from "../services/api";

interface FeatureTimelineProps {
  timeSeries: TimeSeriesData | null;
  duration: number;
  currentTime: number;
  onSeek: (seconds: number) => void;
}

export const FeatureTimeline: React.FC<FeatureTimelineProps> = ({
  timeSeries,
  duration,
  currentTime,
  onSeek,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current || !timeSeries) return;
    const cv = canvasRef.current;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const w = cv.parentElement?.clientWidth || 600;
    const h = cv.parentElement?.clientHeight || 176;
    cv.width = w * window.devicePixelRatio;
    cv.height = h * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    ctx.clearRect(0, 0, w, h);

    const n = timeSeries.time_stamps.length;
    if (n === 0) return;

    // Pitch (Indigo)
    ctx.strokeStyle = "#818cf8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const f0 = timeSeries.f0_contour[i];
      if (f0 > 0) {
        const y = h * 0.30 - Math.min(h * 0.28, ((f0 - 80) / 300) * (h * 0.28));
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      } else {
        started = false;
      }
    }
    ctx.stroke();

    // Energy (Amber)
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const db = timeSeries.rms_energy_db[i];
      const y = h * 0.64 - Math.max(0, Math.min(h * 0.30, ((db + 60) / 60) * (h * 0.30)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Speech Rate (Emerald)
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const wpm = timeSeries.speech_rate_wpm[i];
      const y = h - Math.max(0, Math.min(h * 0.30, ((wpm - 60) / 240) * (h * 0.30)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [timeSeries]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const targetSec = (clickX / rect.width) * duration;
    onSeek(targetSec);
  };

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Synchronized Feature Timelines</h3>
        <div className="flex items-center space-x-3 text-[11px] font-mono text-slate-400">
          <span className="text-indigo-400">● Pitch F0</span>
          <span className="text-amber-400">● Energy dB</span>
          <span className="text-emerald-400">● Rate WPM</span>
        </div>
      </div>

      <div
        onClick={handleCanvasClick}
        className="relative w-full h-44 bg-slate-950/80 rounded-xl overflow-hidden border border-slate-800 cursor-pointer"
      >
        <canvas ref={canvasRef} className="w-full h-full block" />
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-indigo-400 pointer-events-none"
          style={{ left: `${progressPct}%` }}
        />
      </div>
    </div>
  );
};
