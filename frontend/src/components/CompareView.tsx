import React from "react";
import { ComparisonResponse } from "../services/api";

interface CompareViewProps {
  comparison: ComparisonResponse | null;
  onClose: () => void;
}

export const CompareView: React.FC<CompareViewProps> = ({ comparison, onClose }) => {
  if (!comparison) return null;

  const { baseline_analysis, flawed_analysis, score_delta, dimension_deltas, flaw_contrast_summary } = comparison;

  return (
    <div className="fixed inset-0 bg-slate-950/85 backdrop-blur-md z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-4xl w-full p-6 sm:p-8 space-y-6 shadow-2xl my-8">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-lg font-bold text-white">Contrastive Pair Analysis</span>
              <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                Ground-Truth Contrast
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Direct mathematical delta comparison between canonical baseline and flawed delivery.
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center transition"
          >
            ✕
          </button>
        </div>

        {/* Score Comparison Headline */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-slate-950/60 border border-emerald-500/20 rounded-2xl p-4 text-center">
            <span className="text-xs uppercase tracking-wider text-emerald-400 font-semibold">Baseline Score</span>
            <div className="text-3xl font-extrabold text-emerald-400 mt-1">
              {baseline_analysis.rubric_scores.overall_score.toFixed(1)}
            </div>
            <span className="text-[11px] text-slate-400">Exemplary Reference</span>
          </div>

          <div className="bg-slate-950/60 border border-rose-500/20 rounded-2xl p-4 text-center">
            <span className="text-xs uppercase tracking-wider text-rose-400 font-semibold">Flawed Delivery Score</span>
            <div className="text-3xl font-extrabold text-rose-400 mt-1">
              {flawed_analysis.rubric_scores.overall_score.toFixed(1)}
            </div>
            <span className="text-[11px] text-slate-400">Perturbed Speech</span>
          </div>

          <div className="bg-slate-950/60 border border-indigo-500/20 rounded-2xl p-4 text-center">
            <span className="text-xs uppercase tracking-wider text-indigo-400 font-semibold">Net Score Delta</span>
            <div className="text-3xl font-extrabold text-indigo-400 mt-1">
              {score_delta > 0 ? `+${score_delta.toFixed(1)}` : score_delta.toFixed(1)}
            </div>
            <span className="text-[11px] text-slate-400">100-pt Scaled Penalty</span>
          </div>
        </div>

        {/* Dimension Deltas */}
        <div className="bg-slate-950/40 border border-slate-800 rounded-2xl p-5 space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            5-Dimension Rubric Deltas (Baseline vs Flawed)
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {Object.entries(dimension_deltas).map(([dim, delta]) => {
              const baseScore = baseline_analysis.rubric_scores.dimensions[dim]?.score ?? 0;
              const flawScore = flawed_analysis.rubric_scores.dimensions[dim]?.score ?? 0;
              const isNegative = delta < 0;

              return (
                <div key={dim} className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 text-center space-y-1">
                  <span className="text-[11px] font-medium text-slate-400 capitalize">{dim}</span>
                  <div className={`text-base font-bold font-mono ${isNegative ? "text-rose-400" : "text-emerald-400"}`}>
                    {delta > 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1)}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    {flawScore.toFixed(0)} vs {baseScore.toFixed(0)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Flaw Contrast Summary Table */}
        <div className="space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Acoustic Parameter Contrast Metrics
          </h4>
          <div className="border border-slate-800 rounded-2xl overflow-hidden">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] font-semibold tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4">Metric</th>
                  <th className="py-3 px-4">Baseline</th>
                  <th className="py-3 px-4">Flawed</th>
                  <th className="py-3 px-4">Delta</th>
                  <th className="py-3 px-4">Interpretation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
                {flaw_contrast_summary.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/30 transition">
                    <td className="py-3 px-4 font-medium text-white">{row.metric}</td>
                    <td className="py-3 px-4 font-mono">{typeof row.baseline === "number" ? row.baseline.toFixed(2) : row.baseline}</td>
                    <td className="py-3 px-4 font-mono">{typeof row.flawed === "number" ? row.flawed.toFixed(2) : row.flawed}</td>
                    <td className="py-3 px-4 font-mono font-semibold text-indigo-400">
                      {row.delta > 0 ? `+${row.delta.toFixed(2)}` : row.delta.toFixed(2)}
                    </td>
                    <td className="py-3 px-4 text-slate-400">{row.interpretation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition shadow-sm"
          >
            Close Contrast View
          </button>
        </div>
      </div>
    </div>
  );
};
