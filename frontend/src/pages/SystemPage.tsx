import React, { useState } from "react";
import { BenchmarksPage } from "./BenchmarksPage";

interface SystemPageProps {
  onClearSessionUploads?: () => void;
  sessionUploadCount?: number;
}

export const SystemPage: React.FC<SystemPageProps> = ({
  onClearSessionUploads,
  sessionUploadCount = 0,
}) => {
  const [playbackRate, setPlaybackRate] = useState<string>("1.0");

  const handleClearCache = () => {
    localStorage.clear();
    window.location.reload();
  };

  return (
    <div className="space-y-6">
      {/* ==================================================================== */}
      {/* SECTION A: OFFICIAL BENCHMARK EVALUATION (NESTED UNDER SYSTEM)      */}
      {/* ==================================================================== */}
      <BenchmarksPage />

      {/* ==================================================================== */}
      {/* SECTION B: ENGINE TELEMETRY & SPECIFICATIONS                         */}
      {/* ==================================================================== */}
      <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
        <div className="flex items-center space-x-2">
          <h2 className="text-sm font-semibold text-slate-200">Deterministic Engine Specifications</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-2.5">
            <h3 className="font-bold text-slate-300 uppercase tracking-wider text-[11px]">Audio Pipeline Specifications</h3>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Target Sample Rate</span>
              <span className="font-bold text-white mono">16,000 Hz</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Channel Mode</span>
              <span className="font-bold text-white mono">Mono Downmix</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Time-Series Frame Step</span>
              <span className="font-bold text-white mono">50 ms (20 fps)</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Silence Gate Threshold</span>
              <span className="font-bold text-rose-400 mono">-45.0 dBFS</span>
            </div>
          </div>

          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-2.5">
            <h3 className="font-bold text-slate-300 uppercase tracking-wider text-[11px]">Feature Extraction Algorithms</h3>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Pitch (F0) Estimator</span>
              <span className="font-bold text-indigo-400">YIN Algorithm (50 - 500 Hz)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Energy Contour</span>
              <span className="font-bold text-amber-400">RMS Energy dBFS</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/80">
              <span className="text-slate-400">Speech Rate</span>
              <span className="font-bold text-emerald-400">Syllable Peak VAD Density (WPM)</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Normalization</span>
              <span className="font-bold text-white">Speaker-Median Semitones</span>
            </div>
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* SECTION C: CLIENT PREFERENCES & PLAYBACK SETTINGS                   */}
      {/* ==================================================================== */}
      <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
        <h3 className="text-sm font-semibold text-slate-200">Client Preferences &amp; Playback Settings</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-2">
            <label htmlFor="pref-rate" className="block font-medium text-slate-300">Default Playback Rate</label>
            <select
              id="pref-rate"
              value={playbackRate}
              onChange={(e) => setPlaybackRate(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-xl p-2 focus:ring-2 focus:ring-indigo-500"
            >
              <option value="0.75">0.75x (Slow)</option>
              <option value="1.0">1.0x (Normal)</option>
              <option value="1.25">1.25x (Fast)</option>
              <option value="1.5">1.5x (Double Pace)</option>
            </select>
            <span className="text-[10px] text-slate-500 block">Controls audio playback rate in the studio player.</span>
          </div>

          {/* ==================================================================== */}
          {/* SECTION D: SESSION CONTROLS & RESET CONTROLS                         */}
          {/* ==================================================================== */}
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-3">
            <span className="block font-medium text-slate-300">Session Controls &amp; Cache</span>
            <div className="flex flex-col gap-2">
              {onClearSessionUploads && (
                <button
                  onClick={() => {
                    if (confirm("Are you sure you want to clear all session uploads? This will revoke audio object URLs and delete temporary server files.")) {
                      onClearSessionUploads();
                    }
                  }}
                  className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-rose-900/40 border border-slate-700 hover:border-rose-500/50 text-slate-300 hover:text-rose-300 text-xs font-medium transition flex items-center space-x-2"
                >
                  <span>Clear Session Uploads ({sessionUploadCount} Active)</span>
                </button>
              )}
              <button
                onClick={handleClearCache}
                className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 text-xs font-medium transition flex items-center space-x-2"
              >
                <span>Reset State &amp; Reload Dataset</span>
              </button>
            </div>
            <span className="text-[10px] text-slate-500 block">
              Clearing session releases all temporary files and object URLs. Resetting state refetches the authoritative dataset manifest.
            </span>
          </div>
        </div>
      </section>
    </div>
  );
};
