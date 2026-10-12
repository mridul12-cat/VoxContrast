import React from "react";
import { DetectedFlaw } from "../services/api";

interface FlawAnalysisPageProps {
  flaws: DetectedFlaw[];
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onSeek: (seconds: number) => void;
}

export const FlawAnalysisPage: React.FC<FlawAnalysisPageProps> = ({
  flaws,
  currentTime,
  duration,
  isPlaying,
  onTogglePlay,
  onSeek,
}) => {
  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, "0")}:${sec.toFixed(2).padStart(5, "0")}`;
  };

  return (
    <div className="space-y-6">
      {/* Flaw Analysis Header & Controller */}
      <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-bold text-rose-400 uppercase tracking-wider">
                Temporal Flaw Grounding
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20">
                {flaws.length} Detections
              </span>
            </div>
            <h2 className="text-lg font-bold text-white mt-1">Causal Acoustic Anomalies & Flaw Intervals</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Click any flaw card below to seek the audio playback directly to that temporal interval.
            </p>
          </div>

          {/* Mini Playback Controller */}
          <div className="flex items-center space-x-3 bg-slate-950/80 px-4 py-2.5 rounded-xl border border-slate-800">
            <button
              onClick={onTogglePlay}
              className="w-8 h-8 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center transition text-xs"
            >
              {isPlaying ? "❚❚" : "▶"}
            </button>
            <div className="text-xs mono">
              <span className="text-white font-bold">{formatTime(currentTime)}</span>
              <span className="text-slate-500"> / </span>
              <span className="text-slate-400">{formatTime(duration)}</span>
            </div>
          </div>
        </div>

        {/* Evaluation Protocol Notice */}
        <div className="bg-indigo-950/20 border border-indigo-500/30 rounded-xl p-3 text-xs text-indigo-300">
          <span className="font-bold">Evaluation Protocol:</span> Model detections are matched 1-to-1 against
          verified ground-truth annotations using greedy bipartite matching at IoU &ge; 0.50 threshold. Baseline
          reference recordings are verified flaw-free standard.
        </div>
      </section>

      {/* Flaw Cards Container */}
      {flaws.length === 0 ? (
        <div className="text-center py-16 text-emerald-400 text-xs flex flex-col items-center space-y-2 bg-slate-900/40 rounded-2xl border border-slate-800">
          <span className="text-3xl">✓</span>
          <span className="font-bold text-sm">Zero Temporal Flaws Detected</span>
          <span className="text-slate-400 text-xs max-w-md">
            All acoustic feature contours (pitch, energy, local speech rate, pause ratios) remain firmly within
            gold-standard baseline thresholds.
          </span>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {flaws.map((flaw) => {
            const sevClass =
              flaw.severity === "severe"
                ? "border-rose-500/40 bg-rose-500/5 hover:border-rose-500/80 hover:bg-rose-500/10"
                : flaw.severity === "moderate"
                ? "border-amber-500/40 bg-amber-500/5 hover:border-amber-500/80 hover:bg-amber-500/10"
                : "border-yellow-500/40 bg-yellow-500/5 hover:border-yellow-500/80 hover:bg-yellow-500/10";

            return (
              <div key={flaw.flaw_id} className={`p-4 rounded-xl border ${sevClass} transition shadow-sm space-y-3 text-xs`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="font-bold text-white uppercase tracking-wider text-[11px]">
                      {flaw.category.replace("_", " ")}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider ${
                        flaw.severity === "severe"
                          ? "bg-rose-500/20 text-rose-400"
                          : "bg-amber-500/20 text-amber-400"
                      }`}
                    >
                      {flaw.severity}
                    </span>
                  </div>
                  <button
                    onClick={() => onSeek(flaw.start_s)}
                    className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/40 text-[11px] font-semibold transition flex items-center space-x-1.5"
                  >
                    <span>Seek ({flaw.start_s.toFixed(2)}s)</span>
                  </button>
                </div>

                <div className="flex items-center justify-between text-[11px] bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80 mono">
                  <span className="text-slate-400">
                    Span: <strong className="text-white">{flaw.start_s.toFixed(2)}s - {flaw.end_s.toFixed(2)}s</strong>
                  </span>
                  <span className="text-slate-400">
                    Observed: <strong className="text-amber-400">{flaw.observed_value.toFixed(1)} {flaw.unit}</strong>
                  </span>
                  <span className="text-slate-400">
                    Baseline: <strong className="text-emerald-400">{flaw.baseline_value.toFixed(1)} {flaw.unit}</strong>
                  </span>
                </div>

                <p className="text-[11px] text-slate-300 leading-relaxed">{flaw.explanation}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
