import React from "react";
import { DatasetManifest } from "../services/api";

interface DatasetBrowserProps {
  manifest: DatasetManifest | null;
  selectedPairId: string | null;
  datasetVersion: string;
  onSelectDatasetVersion: (version: string) => void;
  onSelectPair: (pairId: string) => void;
  onInspect: () => void;
}

export const DatasetBrowser: React.FC<DatasetBrowserProps> = ({
  manifest,
  selectedPairId,
  datasetVersion,
  onSelectDatasetVersion,
  onSelectPair,
  onInspect,
}) => {
  if (!manifest) return null;

  const isV2 =
    manifest.provenance_class === "spoken_speech_v2" ||
    manifest.provenance_class === "tts_spoken_speech_v2";
  const totalFlaws =
    manifest.metadata.total_confirmed_flaws || manifest.metadata.total_annotated_flaw_regions;

  return (
    <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 sm:p-5 shadow-sm">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
              {isV2
                ? `Spoken Speech Contrastive Pairs (Authoritative Dataset — ${totalFlaws} Confirmed Flaws)`
                : `Curated Contrastive Pairs (Historical Benchmark — ${totalFlaws} Flaws)`}
            </h2>
            <span
              className={`px-2 py-0.5 text-[10px] font-semibold rounded-full ${
                isV2
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                  : "bg-indigo-500/10 text-indigo-400 border border-indigo-500/20"
              }`}
            >
              {isV2 ? "Natural Speech" : "Synthetic Glottal Model"}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            {isV2
              ? "Active Dataset: Natural spoken speech (Samantha en_US & Daniel en_GB RP) with exact temporal ground truth."
              : "Historical reference dataset (sinusoidal glottal acoustic model, 54 flaw regions)."}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400">Dataset:</span>
            <select
              value={datasetVersion}
              onChange={(e) => onSelectDatasetVersion(e.target.value)}
              className="bg-slate-800 border border-indigo-500/30 text-indigo-300 text-xs rounded-xl px-2.5 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            >
              <option value="v2">🎙️ Spoken Speech Dataset (Authoritative)</option>
            </select>
          </div>
          <select
            value={selectedPairId || ""}
            onChange={(e) => onSelectPair(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none max-w-xs sm:max-w-md"
          >
            {manifest.pairs.map((p) => {
              const text = manifest.speech_texts.find((t) => t.text_id === p.text_id);
              const voiceInfo = p.speaker_metadata?.voice ? ` [${p.speaker_metadata.voice}]` : "";
              return (
                <option key={p.pair_id} value={p.pair_id}>
                  {text ? text.title : p.text_id} — [{p.flaw_category} ({p.severity})]{voiceInfo}
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
