import React from "react";
import { DatasetManifest } from "../services/api";

interface DatasetBrowserProps {
  manifest: DatasetManifest | null;
  selectedPairId: string | null;
  onSelectPair: (pairId: string) => void;
  onInspect: () => void;
}

export const DatasetBrowser: React.FC<DatasetBrowserProps> = ({
  manifest,
  selectedPairId,
  onSelectPair,
  onInspect,
}) => {
  if (!manifest) return null;

  return (
    <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 sm:p-5 shadow-sm">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Curated Contrastive Pairs ({manifest.metadata.total_annotated_flaw_regions} Annotated Flaws)
          </h2>
          <p className="text-xs text-slate-400">
            Select any paired contrastive delivery from our canonical dataset manifest.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={selectedPairId || ""}
            onChange={(e) => onSelectPair(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none max-w-md"
          >
            {manifest.pairs.map((p) => {
              const text = manifest.speech_texts.find((t) => t.text_id === p.text_id);
              return (
                <option key={p.pair_id} value={p.pair_id}>
                  {text ? text.title : p.text_id} — [{p.flaw_category} ({p.severity})]
                </option>
              );
            })}
          </select>
          <button
            onClick={onInspect}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium transition"
          >
            Inspect Contrast
          </button>
        </div>
      </div>
    </section>
  );
};
