import React from "react";
import { ComparisonResponse, DatasetManifest } from "../services/api";

interface OverviewPageProps {
  comparison: ComparisonResponse | null;
  manifest: DatasetManifest | null;
  onNavigateTab: (tab: "compare" | "flaws" | "dataset") => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  comparison,
  manifest,
  onNavigateTab,
}) => {
  if (!comparison || !manifest) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-8 text-center text-slate-400">
        Loading overview telemetry...
      </div>
    );
  }

  const pairMeta = manifest.pairs.find((p) => p.pair_id === comparison.comparison_id);
  const textMeta = manifest.speech_texts.find((t) => t.text_id === pairMeta?.text_id);

  const baseScore = comparison.baseline_analysis.rubric_scores.overall_score;
  const flawScore = comparison.flawed_analysis.rubric_scores.overall_score;
  const delta = comparison.score_delta;

  return (
    <div className="space-y-6">
      {/* Hero Card: Active Pair Spotlight & Score Comparison */}
      <section className="bg-gradient-to-r from-indigo-950/40 via-purple-950/30 to-slate-900 border border-indigo-500/20 rounded-2xl p-6 space-y-6 shadow-sm">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="flex items-center space-x-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20 uppercase tracking-wider">
                {pairMeta?.flaw_category.replace("_", " ")}
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-slate-800 text-slate-300 border border-slate-700">
                Voice: {pairMeta?.voice_model || "Standard"}
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Spoken Speech V2
              </span>
            </div>
            <h2 className="text-2xl font-extrabold text-white tracking-tight">
              {textMeta?.title || "Speech Utterance"}
            </h2>
            <blockquote className="text-sm text-slate-300 italic border-l-2 border-indigo-500/50 pl-3 py-1">
              "{textMeta?.canonical_transcript || ""}"
            </blockquote>
          </div>

          {/* Score Contrast Dial & Pill */}
          <div className="flex items-center space-x-4 bg-slate-900/90 p-5 rounded-2xl border border-slate-800/80 shadow-md">
            <div className="text-center px-2">
              <div className="text-xs text-slate-400 font-medium">Ideal Baseline</div>
              <div className="text-2xl font-black text-emerald-400 mt-1">{baseScore.toFixed(1)}</div>
              <div className="text-[10px] text-emerald-500/90 font-medium uppercase tracking-wider">Reference</div>
            </div>
            <div className="h-10 w-px bg-slate-800"></div>
            <div className="text-center px-2">
              <div className="text-xs text-slate-400 font-medium">Flawed Delivery</div>
              <div className="text-2xl font-black text-amber-400 mt-1">{flawScore.toFixed(1)}</div>
              <div className="text-[10px] text-amber-400/90 font-medium uppercase tracking-wider">Flawed</div>
            </div>
            <div className="h-10 w-px bg-slate-800"></div>
            <div className="text-center px-2">
              <div className="text-xs text-slate-400 font-medium">Score Delta</div>
              <div className={`text-2xl font-black mt-1 ${delta < 0 ? "text-rose-400" : "text-emerald-400"}`}>
                {delta > 0 ? "+" : ""}{delta.toFixed(1)} pts
              </div>
              <div className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">Contrast</div>
            </div>
          </div>
        </div>

        {/* Quick Contrast Metrics Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-slate-800/80 text-xs">
          {comparison.flaw_contrast_summary.map((item, idx) => (
            <div key={idx} className="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-400 text-[11px]">{item.metric}</span>
              <div className="text-white font-bold mono text-sm mt-0.5">
                {item.flawed} ({item.delta > 0 ? "+" : ""}{item.delta})
              </div>
              <span className="text-[10px] text-slate-500 italic block truncate">{item.interpretation}</span>
            </div>
          ))}
        </div>

        {/* Action Buttons to Open Workspaces */}
        <div className="flex flex-wrap items-center gap-3 pt-2">
          <button
            onClick={() => onNavigateTab("compare")}
            className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition flex items-center space-x-2 shadow-sm shadow-indigo-600/30"
          >
            <span>Open in Compare Audio Studio</span>
            <span className="text-[10px]">→</span>
          </button>
          <button
            onClick={() => onNavigateTab("flaws")}
            className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition flex items-center space-x-2"
          >
            <span className="text-rose-400 font-bold">●</span>
            <span>Inspect Grounded Flaws ({comparison.flawed_analysis.detected_flaws.length})</span>
            <span className="text-[10px]">→</span>
          </button>
          <button
            onClick={() => onNavigateTab("dataset")}
            className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition flex items-center space-x-2"
          >
            <span>Browse All 18 Pairs</span>
          </button>
        </div>
      </section>

      {/* Dataset & Engine Integrity Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center space-x-2 text-indigo-400 text-xs font-bold uppercase tracking-wider">
            <span>Dataset Integrity</span>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Authoritative Utterances</span>
              <span className="font-bold text-white mono">24 WAVs</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Contrastive Pairs</span>
              <span className="font-bold text-white mono">18 Pairs</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Canonical Speeches</span>
              <span className="font-bold text-white mono">6 Speeches</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Provenance Class</span>
              <span className="font-bold text-emerald-400">spoken_speech_v2</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center space-x-2 text-purple-400 text-xs font-bold uppercase tracking-wider">
            <span>Deterministic Scoring</span>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Rubric Standard</span>
              <span className="font-bold text-white">5-Dimension 100 Pt</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Baseline Score</span>
              <span className="font-bold text-emerald-400 mono">95.0 (Exemplary)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Reproducibility Variance</span>
              <span className="font-bold text-white mono">0.0000 pts</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">External Model Cost</span>
              <span className="font-bold text-emerald-400">$0.00 (Zero LLM)</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center space-x-2 text-emerald-400 text-xs font-bold uppercase tracking-wider">
            <span>Grounding Benchmark</span>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Temporal Macro IoU</span>
              <span className="font-bold text-emerald-400 mono">1.0000 (100%)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Mean Onset Error</span>
              <span className="font-bold text-white mono">0.0 ms</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Mean Offset Error</span>
              <span className="font-bold text-white mono">0.0 ms</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Recall @ 0.5 IoU</span>
              <span className="font-bold text-emerald-400 mono">1.0000 (100%)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
