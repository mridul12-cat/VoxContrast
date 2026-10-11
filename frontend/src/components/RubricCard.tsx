import React from "react";
import { RubricScores } from "../services/api";

interface RubricCardProps {
  rubric: RubricScores | null;
}

export const RubricCard: React.FC<RubricCardProps> = ({ rubric }) => {
  if (!rubric) return null;

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Deterministic Rubric (100 Pt Standard)</h3>
        <span className="text-[10px] font-mono bg-slate-800 text-slate-400 px-2 py-0.5 rounded border border-slate-700">
          100% Deterministic
        </span>
      </div>

      <div className="flex items-center space-x-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
        <div className="text-3xl font-extrabold text-indigo-400">{rubric.overall_score.toFixed(1)}</div>
        <div className="space-y-1">
          <div className="text-sm font-bold text-slate-200">
            {rubric.overall_score >= 85 ? "Exemplary Delivery" : rubric.overall_score >= 70 ? "Effective Speech" : "Delivery Flaws Grounded"}
          </div>
          <p className="text-xs text-slate-400">Speaker-normalized scoring across 5 acoustic dimensions.</p>
        </div>
      </div>

      <div className="space-y-2.5">
        {Object.entries(rubric.dimensions).map(([key, dim]) => (
          <div key={key} className="bg-slate-950/40 p-3 rounded-xl border border-slate-800/80 space-y-1.5 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-medium text-slate-300">{dim.dimension_name}</span>
              <div className="font-mono space-x-1">
                <span className="font-bold text-white">{dim.score.toFixed(1)}</span>
                <span className="text-slate-500 text-[10px]">/ 100</span>
                {dim.raw_deduction > 0 && (
                  <span className="text-rose-400 text-[11px]">(-{dim.raw_deduction.toFixed(1)})</span>
                )}
              </div>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  dim.score >= 80 ? "bg-emerald-500" : dim.score >= 60 ? "bg-indigo-500" : "bg-amber-500"
                }`}
                style={{ width: `${Math.round(dim.score)}%` }}
              />
            </div>
            {dim.deduction_reasons.length > 0 && (
              <div className="text-[11px] text-slate-400 mt-1 space-y-0.5">
                {dim.deduction_reasons.map((r, i) => (
                  <div key={i} className="truncate text-slate-400">
                    • {r}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
