import React, { useState } from "react";
import { DatasetManifest, SessionUploadItem } from "../services/api";

interface DatasetExplorerPageProps {
  manifest: DatasetManifest | null;
  selectedPairId: string | null;
  onSelectPair: (pairId: string) => void;
  onOpenUpload: () => void;
  sessionUploads?: SessionUploadItem[];
  onSelectUpload?: (uploadId: string) => void;
  onAnalyzeUpload?: (uploadId: string) => void;
  onRemoveUpload?: (uploadId: string) => void;
  onClearSessionUploads?: () => void;
}

export const DatasetExplorerPage: React.FC<DatasetExplorerPageProps> = ({
  manifest,
  selectedPairId,
  onSelectPair,
  onOpenUpload,
  sessionUploads = [],
  onSelectUpload,
  onAnalyzeUpload,
  onRemoveUpload,
  onClearSessionUploads,
}) => {
  const [search, setSearch] = useState("");
  const [splitFilter, setSplitFilter] = useState("all");

  const filteredPairs = manifest
    ? manifest.pairs.filter((p: any) => {
        const text = manifest.speech_texts.find((t) => t.text_id === p.text_id);
        const title = text ? text.title : p.text_id;
        const cat = p.flaw_category;
        const split = p.split || "train_dev";

        if (splitFilter !== "all" && split !== splitFilter) return false;
        if (
          search &&
          !title.toLowerCase().includes(search.toLowerCase()) &&
          !p.pair_id.toLowerCase().includes(search.toLowerCase()) &&
          !cat.toLowerCase().includes(search.toLowerCase())
        ) {
          return false;
        }
        return true;
      })
    : [];

  return (
    <div className="space-y-6">
      {/* ==================================================================== */}
      {/* SECTION A: SESSION UPLOADS (CURRENT SESSION ONLY)                    */}
      {/* ==================================================================== */}
      <section className="bg-gradient-to-r from-slate-900 via-indigo-950/20 to-slate-900 border border-indigo-500/30 rounded-2xl p-5 space-y-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
                Session Uploads
              </span>
              <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                Session-Scoped (Current Session Only)
              </span>
              <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                Exploratory / No Assumed Ground Truth
              </span>
            </div>
            <h2 className="text-lg font-bold text-white mt-1">Custom Audio Uploads (Session Workspace)</h2>
            <p className="text-xs text-slate-400">
              Audio uploaded here is isolated to your current browser session. It is excluded from official benchmarks and never modifies authoritative dataset files.
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={onOpenUpload}
              className="px-3.5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition flex items-center space-x-1.5 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <span>Upload Audio</span>
            </button>
            {sessionUploads.length > 0 && onClearSessionUploads && (
              <button
                onClick={() => {
                  if (confirm("Are you sure you want to clear all session uploads?")) {
                    onClearSessionUploads();
                  }
                }}
                className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-rose-900/40 border border-slate-700 hover:border-rose-500/50 text-slate-300 hover:text-rose-300 text-xs font-medium transition"
              >
                Clear Session
              </button>
            )}
          </div>
        </div>

        {/* Session Uploads Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th className="p-3">File Name</th>
                <th className="p-3">Duration</th>
                <th className="p-3">Sample Rate</th>
                <th className="p-3">Upload Status</th>
                <th className="p-3">Analysis Status</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {sessionUploads.length === 0 ? (
                <tr>
                  <td colSpan={6} className="p-6 text-center text-slate-500 text-xs">
                    No audio uploaded in this session yet. Click <strong className="text-indigo-400 font-medium">'Upload Audio'</strong> to add exploratory speech files.
                  </td>
                </tr>
              ) : (
                sessionUploads.map((u) => (
                  <tr key={u.upload_id} className="hover:bg-slate-800/40 transition">
                    <td className="p-3 font-medium text-white flex items-center space-x-2">
                      <span className="truncate max-w-xs">{u.filename}</span>
                    </td>
                    <td className="p-3 mono text-slate-300">{u.duration_s.toFixed(2)}s</td>
                    <td className="p-3 mono text-slate-400">{(u.sample_rate || 16000).toLocaleString()} Hz</td>
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">
                        {u.upload_status}
                      </span>
                    </td>
                    <td className="p-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${
                          u.analysis_status === "analyzed"
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                            : u.analysis_status === "failed"
                            ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                            : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                        }`}
                      >
                        {u.analysis_status}
                      </span>
                    </td>
                    <td className="p-3 text-right space-x-1.5 whitespace-nowrap">
                      {onSelectUpload && (
                        <button
                          onClick={() => onSelectUpload(u.upload_id)}
                          className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/30 text-[11px] font-semibold transition"
                        >
                          Open in Compare
                        </button>
                      )}
                      {onAnalyzeUpload && (
                        <button
                          onClick={() => onAnalyzeUpload(u.upload_id)}
                          className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-[11px] font-semibold transition"
                        >
                          Analyze
                        </button>
                      )}
                      {onRemoveUpload && (
                        <button
                          onClick={() => onRemoveUpload(u.upload_id)}
                          title="Remove Upload"
                          className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-rose-900/40 text-slate-400 hover:text-rose-300 border border-slate-700 hover:border-rose-500/40 text-[11px] transition"
                        >
                          Remove
                        </button>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* SECTION B: AUTHORITATIVE CONTRASTIVE SPEECH DATASET                  */}
      {/* ==================================================================== */}
      <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
                Authoritative Dataset
              </span>
              <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Version {manifest?.dataset_version || "2.0.0"} (Spoken Speech V2)
              </span>
              <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                Official Benchmark Ground Truth
              </span>
            </div>
            <h2 className="text-lg font-bold text-white mt-1">Paired Contrastive Speech Dataset</h2>
            <p className="text-xs text-slate-400">
              18 authoritative contrastive pairs across 6 historical speeches with expert temporal ground truth annotations.
            </p>
          </div>
          <div className="flex items-center space-x-2 text-xs">
            <span className="text-slate-400">Filter Split:</span>
            <select
              value={splitFilter}
              onChange={(e) => setSplitFilter(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-xl px-2.5 py-1.5 focus:ring-2 focus:ring-indigo-500"
            >
              <option value="all">All Splits (18 Pairs)</option>
              <option value="train_dev">Train / Dev (12 Pairs)</option>
              <option value="final_test">Final Test (6 Pairs)</option>
            </select>
          </div>
        </div>

        {/* Dataset Summary Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <span className="text-slate-400 text-[11px]">Authoritative Utterances</span>
            <div className="text-lg font-bold text-white mono mt-0.5">24 WAVs</div>
            <span className="text-[10px] text-slate-500">6 Baseline + 18 Flawed</span>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <span className="text-slate-400 text-[11px]">Confirmed Ground Truth Flaws</span>
            <div className="text-lg font-bold text-white mono mt-0.5">18 Intervals</div>
            <span className="text-[10px] text-slate-500">Zero uncertain entries</span>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <span className="text-slate-400 text-[11px]">Flaw Categories</span>
            <div className="text-lg font-bold text-white mono mt-0.5">6 Classes</div>
            <span className="text-[10px] text-slate-500">Speed, Pause, Pitch, Energy</span>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <span className="text-slate-400 text-[11px]">Audio Sample Rate</span>
            <div className="text-lg font-bold text-emerald-400 mono mt-0.5">16,000 Hz</div>
            <span className="text-[10px] text-slate-500">16-bit Mono Linear PCM</span>
          </div>
        </div>
      </section>

      {/* Search & Pairs Catalog Table */}
      <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="relative flex-1 max-w-md">
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              type="text"
              placeholder="Search by speech title, pair ID, or flaw category..."
              className="w-full bg-slate-950/80 border border-slate-800 text-xs rounded-xl pl-3 pr-3 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <div className="text-xs text-slate-400">Showing {filteredPairs.length} pairs</div>
        </div>

        {/* Responsive Pairs Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th className="p-3">Speech Text</th>
                <th className="p-3">Flaw Category</th>
                <th className="p-3">Severity</th>
                <th className="p-3">Split</th>
                <th className="p-3">Voice</th>
                <th className="p-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {filteredPairs.map((p: any) => {
                const text = manifest?.speech_texts.find((t) => t.text_id === p.text_id);
                const isSelected = selectedPairId === p.pair_id;
                return (
                  <tr
                    key={p.pair_id}
                    onClick={() => onSelectPair(p.pair_id)}
                    className="hover:bg-slate-800/40 transition cursor-pointer"
                  >
                    <td className="p-3 font-semibold text-white flex items-center space-x-2">
                      {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>}
                      <span>{text ? text.title : p.text_id}</span>
                    </td>
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">
                        {p.flaw_category.replace("_", " ")}
                      </span>
                    </td>
                    <td className="p-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                          p.severity === "severe" ? "text-rose-400" : "text-amber-400"
                        }`}
                      >
                        {p.severity}
                      </span>
                    </td>
                    <td className="p-3 mono text-[11px] text-slate-400">{p.split || "train_dev"}</td>
                    <td className="p-3 mono text-[11px] text-slate-400">{p.voice_model || "standard"}</td>
                    <td className="p-3 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectPair(p.pair_id);
                        }}
                        className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/30 text-[11px] font-semibold transition"
                      >
                        Select
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
