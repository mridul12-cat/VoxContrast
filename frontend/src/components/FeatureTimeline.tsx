import React, { useRef, useEffect } from "react";
import { TimeSeriesData, DetectedFlaw } from "../services/api";

interface FeatureTimelineProps {
  timeSeries: TimeSeriesData | null;
  duration: number;
  currentTime: number;
  onSeek: (seconds: number) => void;
  flaws?: DetectedFlaw[];
}

export const FeatureTimeline: React.FC<FeatureTimelineProps> = ({
  timeSeries,
  duration,
  currentTime,
  onSeek,
  flaws = [],
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

    // Grid guide lines
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.33); ctx.lineTo(w, h * 0.33);
    ctx.moveTo(0, h * 0.66); ctx.lineTo(w, h * 0.66);
    ctx.stroke();

    // Pause Region Highlights on Timeline
    const totalDur = duration || (n > 1 ? timeSeries.time_stamps[n - 1] : 10);
    const SILENCE_DB = -36.0;
    interface PauseRegion {
      start_s: number;
      end_s: number;
      duration_s: number;
      isFlaw: boolean;
    }
    const pauses: PauseRegion[] = [];
    let pStart: number | null = null;
    const dt = timeSeries.time_step_s || (n > 1 ? (timeSeries.time_stamps[n - 1] - timeSeries.time_stamps[0]) / (n - 1) : 0.05);

    for (let i = 0; i < n; i++) {
      const t = timeSeries.time_stamps[i];
      const isQuiet = timeSeries.rms_energy_db[i] <= SILENCE_DB;
      if (isQuiet && pStart === null) {
        pStart = t;
      } else if (!isQuiet && pStart !== null) {
        const pEnd = t;
        const dur = pEnd - pStart;
        if (dur >= 0.35) {
          pauses.push({ start_s: pStart, end_s: pEnd, duration_s: dur, isFlaw: false });
        }
        pStart = null;
      }
    }
    if (pStart !== null) {
      const pEnd = timeSeries.time_stamps[n - 1] + dt;
      if (pEnd - pStart >= 0.35) {
        pauses.push({ start_s: pStart, end_s: pEnd, duration_s: pEnd - pStart, isFlaw: false });
      }
    }

    // Correlate with detected unnatural pause flaws
    flaws.forEach((f) => {
      if (f.category === "unnatural_pause") {
        const matched = pauses.find((p) => Math.abs(p.start_s - f.start_s) < 0.4 || (f.start_s >= p.start_s && f.end_s <= p.end_s));
        if (!matched) {
          pauses.push({ start_s: f.start_s, end_s: f.end_s, duration_s: f.end_s - f.start_s, isFlaw: true });
        } else {
          matched.isFlaw = true;
        }
      }
    });

    pauses.forEach((p) => {
      const startX = (p.start_s / (totalDur || 1)) * w;
      const endX = (p.end_s / (totalDur || 1)) * w;
      const pauseW = Math.max(2, endX - startX);

      ctx.fillStyle = p.isFlaw ? "rgba(245, 158, 11, 0.12)" : "rgba(51, 65, 85, 0.25)";
      ctx.fillRect(startX, 0, pauseW, h);

      ctx.strokeStyle = p.isFlaw ? "rgba(245, 158, 11, 0.5)" : "rgba(100, 116, 139, 0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(startX, 0); ctx.lineTo(startX, h);
      ctx.moveTo(endX, 0); ctx.lineTo(endX, h);
      ctx.stroke();
      ctx.setLineDash([]);
    });

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

    // Energy Silence Threshold (-38 dB) guide line
    const silenceY = h * 0.64 - Math.max(0, Math.min(h * 0.30, ((-38 + 60) / 60) * (h * 0.30)));
    ctx.strokeStyle = "rgba(239, 68, 68, 0.45)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(0, silenceY); ctx.lineTo(w, silenceY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#ef4444";
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.textAlign = "right";
    ctx.fillText("Silence Threshold (-38dB)", w - 6, silenceY - 3);

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
  }, [timeSeries, duration, flaws]);

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
