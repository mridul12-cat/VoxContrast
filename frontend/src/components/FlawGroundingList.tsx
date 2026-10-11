import React from "react";
import { DetectedFlaw } from "../services/api";

interface FlawGroundingListProps {
  flaws: DetectedFlaw[];
  onSeekToFlaw: (start_s: number) => void;
}

export const FlawGroundingList: React.FC<FlawGroundingListProps> = ({ flaws, onSeekToFlaw }) => {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Temporal Flaw Grounding</h3>
        <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20">
          {flaws.length} detected
        </span>
      </div>
      <p className="text-xs text-slate-400">Click any flaw card below to seek the player to the exact anomaly timestamp.</p>

      <div className="space-y-3 max-h-[460px] overflow-y-auto pr-1">
        {flaws.length === 0 ? (
          <div className="text-center py-8 text-emerald-400 text-xs">
            0 Temporal Flaws Detected (Delivery within optimal bounds)
          </div>
        ) : (
          flaws.map((flaw) => {
            const sevClass =
              flaw.severity === "severe"
                ? "border-rose-500/40 bg-rose-500/5 hover:border-rose-500/70"
                : flaw.severity === "moderate"
                ? "border-amber-500/40 bg-amber-500/5 hover:border-amber-500/70"
                : "border-yellow-500/40 bg-yellow-500/5 hover:border-yellow-500/70";

            return (
              <div
                key={flaw.flaw_id}
                onClick={() => onSeekToFlaw(flaw.start_s)}
                className={`p-3.5 rounded-xl border ${sevClass} cursor-pointer transition shadow-sm space-y-2 text-xs group`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="font-bold text-white uppercase text-[11px] tracking-wider">
                      {flaw.category.replace("_", " ")}
                    </span>
                    <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold uppercase bg-slate-800 text-slate-300">
                      {flaw.severity}
                    </span>
                  </div>
                  <span className="font-mono text-[11px] text-indigo-400 group-hover:underline">
                    [{flaw.start_s.toFixed(2)}s - {flaw.end_s.toFixed(2)}s]
                  </span>
                </div>
                <div className="text-slate-300 italic text-[11px] bg-slate-950/40 px-2 py-1 rounded border border-slate-800/80">
                  "{flaw.transcript_slice}"
                </div>
                <p className="text-slate-300 leading-relaxed text-[11px]">{flaw.explanation}</p>
                <div className="text-[10px] text-slate-500 border-t border-slate-800/60 pt-1.5">
                  {flaw.caveat}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
