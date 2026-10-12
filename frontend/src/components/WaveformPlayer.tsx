import React, { useRef, useEffect } from "react";
import { TimeSeriesData, DetectedFlaw } from "../services/api";

interface WaveformPlayerProps {
  timeSeries: TimeSeriesData | null;
  duration: number;
  currentTime: number;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onSeek: (seconds: number) => void;
  activeType: "flawed" | "baseline";
  onSwitchType: (type: "flawed" | "baseline") => void;
  flaws?: DetectedFlaw[];
}

export const WaveformPlayer: React.FC<WaveformPlayerProps> = ({
  timeSeries,
  duration,
  currentTime,
  isPlaying,
  onTogglePlay,
  onSeek,
  activeType,
  onSwitchType,
  flaws = [],
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current || !timeSeries) return;
    const cv = canvasRef.current;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const w = cv.parentElement?.clientWidth || 600;
    const h = cv.parentElement?.clientHeight || 112;
    cv.width = w * window.devicePixelRatio;
    cv.height = h * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    ctx.clearRect(0, 0, w, h);

    const n = timeSeries.rms_energy_db.length;
    if (n === 0) return;
    const barWidth = Math.max(2, w / n);
    const centerY = h / 2;
    const totalDur = duration || n * (timeSeries.time_step_s || 0.05);

    // 1. Identify pause intervals
    const pauses: Array<{ start_s: number; end_s: number; duration_s: number; isFlaw?: boolean }> = [];
    let inPause = false;
    let pauseStart = 0;
    for (let i = 0; i < n; i++) {
      const t = timeSeries.time_stamps ? timeSeries.time_stamps[i] : i * 0.05;
      const db = timeSeries.rms_energy_db[i];
      if (db <= -36.0) {
        if (!inPause) {
          inPause = true;
          pauseStart = t;
        }
      } else {
        if (inPause) {
          inPause = false;
          const dur = t - pauseStart;
          if (dur >= 0.35) {
            pauses.push({ start_s: pauseStart, end_s: t, duration_s: dur, isFlaw: false });
          }
        }
      }
    }
    if (inPause) {
      const dur = totalDur - pauseStart;
      if (dur >= 0.35) {
        pauses.push({ start_s: pauseStart, end_s: totalDur, duration_s: dur, isFlaw: false });
      }
    }

    (flaws || []).forEach((f) => {
      if (f.category === "unnatural_pause") {
        const matched = pauses.find((p) => Math.abs(p.start_s - f.start_s) < 0.4);
        if (!matched) {
          pauses.push({ start_s: f.start_s, end_s: f.end_s, duration_s: f.end_s - f.start_s, isFlaw: true });
        } else {
          matched.isFlaw = true;
        }
      }
    });

    // 2. Draw pause backgrounds
    pauses.forEach((p) => {
      const startX = (p.start_s / (totalDur || 1)) * w;
      const endX = (p.end_s / (totalDur || 1)) * w;
      const pauseW = Math.max(3, endX - startX);

      ctx.fillStyle = p.isFlaw ? "rgba(245, 158, 11, 0.18)" : "rgba(51, 65, 85, 0.35)";
      ctx.fillRect(startX, 0, pauseW, h);

      ctx.strokeStyle = p.isFlaw ? "rgba(245, 158, 11, 0.7)" : "rgba(100, 116, 139, 0.5)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(startX, 0);
      ctx.lineTo(startX, h);
      ctx.moveTo(endX, 0);
      ctx.lineTo(endX, h);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // 3. Center guide line
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(w, centerY);
    ctx.stroke();

    // 4. Waveform bars
    const SILENCE_DB = -36.0;
    const PEAK_DB = -8.0;

    for (let i = 0; i < n; i++) {
      const x = i * barWidth;
      const t = timeSeries.time_stamps ? timeSeries.time_stamps[i] : i * 0.05;
      const db = timeSeries.rms_energy_db[i];
      const activeFlaw = (flaws || []).find((f) => t >= f.start_s && t <= f.end_s);

      if (db <= SILENCE_DB) {
        ctx.fillStyle = activeFlaw?.category === "unnatural_pause" ? "#fbbf24" : "#475569";
        ctx.fillRect(x, centerY - 0.75, Math.max(1, barWidth - 1), 1.5);
      } else {
        const ratio = Math.max(0, Math.min(1.0, (db - SILENCE_DB) / (PEAK_DB - SILENCE_DB)));
        const norm = Math.pow(ratio, 1.35);
        const barH = Math.max(3, norm * (h * 0.85));

        if (activeFlaw) {
          ctx.fillStyle = activeFlaw.category === "unnatural_pause" ? "#fbbf24" : "#f43f5e";
        } else {
          ctx.fillStyle = "#6366f1";
        }
        ctx.fillRect(x, centerY - barH / 2, Math.max(1, barWidth - 1), barH);
      }
    }
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
        <div className="flex items-center space-x-3">
          <button
            onClick={onTogglePlay}
            className="w-11 h-11 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center transition shadow-md shadow-indigo-600/30"
          >
            {isPlaying ? "⏸" : "▶"}
          </button>
          <div>
            <span className="text-sm font-semibold text-slate-200">
              {activeType === "flawed" ? "Flawed Audio Track" : "Baseline Audio Track"}
            </span>
            <div className="text-xs text-slate-400 font-mono">
              {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
            </div>
          </div>
        </div>

        <div className="flex items-center bg-slate-800/80 p-1 rounded-xl border border-slate-700 text-xs">
          <button
            onClick={() => onSwitchType("flawed")}
            className={`px-3 py-1.5 rounded-lg font-medium transition ${
              activeType === "flawed" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
            }`}
          >
            Flawed Audio
          </button>
          <button
            onClick={() => onSwitchType("baseline")}
            className={`px-3 py-1.5 rounded-lg font-medium transition ${
              activeType === "baseline" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
            }`}
          >
            Baseline Audio
          </button>
        </div>
      </div>

      <div
        onClick={handleCanvasClick}
        className="relative w-full h-28 bg-slate-950/80 rounded-xl overflow-hidden border border-slate-800 cursor-pointer"
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
