import React, { useRef, useEffect } from "react";
import { TimeSeriesData } from "../services/api";

interface WaveformPlayerProps {
  timeSeries: TimeSeriesData | null;
  duration: number;
  currentTime: number;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onSeek: (seconds: number) => void;
  activeType: "flawed" | "baseline";
  onSwitchType: (type: "flawed" | "baseline") => void;
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
    const barWidth = Math.max(2, w / n);
    const centerY = h / 2;

    for (let i = 0; i < n; i++) {
      const x = i * barWidth;
      const db = timeSeries.rms_energy_db[i];
      const norm = Math.max(0.04, Math.min(1.0, (db + 60) / 60));
      const barH = norm * (h * 0.85);

      ctx.fillStyle = db < -38.0 ? "#1e293b" : "#6366f1";
      ctx.fillRect(x, centerY - barH / 2, barWidth - 1, barH);
    }
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
