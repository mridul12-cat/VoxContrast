import React from "react";
import { WaveformPlayer } from "../components/WaveformPlayer";
import { FeatureTimeline } from "../components/FeatureTimeline";
import { RubricCard } from "../components/RubricCard";
import { AnalysisResponse, ComparisonResponse } from "../services/api";

interface CompareAudioPageProps {
  activeAnalysis: AnalysisResponse | null;
  comparison: ComparisonResponse | null;
  activeType: "flawed" | "baseline";
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  onTogglePlay: () => void;
  onSeek: (seconds: number) => void;
  onSwitchType: (type: "flawed" | "baseline") => void;
  onOpenFlaws: () => void;
  isSessionUpload?: boolean;
  selectedBaselineRef?: string;
  onSelectBaselineRef?: (baselineId: string) => void;
  onReturnToDatasetPair?: () => void;
}

export const CompareAudioPage: React.FC<CompareAudioPageProps> = ({
  activeAnalysis,
  activeType,
  isPlaying,
  currentTime,
  duration,
  onTogglePlay,
  onSeek,
  onSwitchType,
  onOpenFlaws,
  isSessionUpload = false,
  selectedBaselineRef = "",
  onSelectBaselineRef,
  onReturnToDatasetPair,
}) => {
  if (!activeAnalysis) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-8 text-center text-slate-400">
        Loading audio analysis workspace...
      </div>
    );
  }

  const flaws = activeAnalysis.detected_flaws || [];

  return (
    <div className="space-y-6">
      {/* Session Upload Baseline Reference Bar */}
      {isSessionUpload && (
        <div className="bg-indigo-950/40 border border-indigo-500/30 rounded-xl p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center space-x-2">
            <span className="text-indigo-400 font-semibold">Pair with Baseline:</span>
            <select
              value={selectedBaselineRef}
              onChange={(e) => onSelectBaselineRef && onSelectBaselineRef(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1 focus:ring-1 focus:ring-indigo-500"
            >
              <option value="">-- Standalone Evaluation (No Baseline) --</option>
              <option value="speech_01_gettysburg_baseline.wav">Gettysburg Address (Ideal Baseline)</option>
              <option value="speech_02_churchill_baseline.wav">Churchill Finest Hour (Ideal Baseline)</option>
              <option value="speech_03_kennedy_baseline.wav">Kennedy Inaugural (Ideal Baseline)</option>
              <option value="speech_04_keynote_baseline.wav">Keynote Presentation (Ideal Baseline)</option>
              <option value="speech_05_defense_baseline.wav">Legal Defense (Ideal Baseline)</option>
              <option value="speech_06_debate_baseline.wav">Public Debate (Ideal Baseline)</option>
            </select>
          </div>
          {onReturnToDatasetPair && (
            <button
              onClick={onReturnToDatasetPair}
              className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition"
            >
              Return to Dataset Pair
            </button>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (7 cols): Waveform Player & Feature Timelines */}
        <div className="lg:col-span-7 space-y-6">
          <WaveformPlayer
            timeSeries={activeAnalysis.time_series || null}
            duration={duration || activeAnalysis.audio_metadata.duration_s || 10}
            currentTime={currentTime}
            isPlaying={isPlaying}
            onTogglePlay={onTogglePlay}
            onSeek={onSeek}
            activeType={activeType}
            onSwitchType={onSwitchType}
            flaws={flaws}
          />

          <FeatureTimeline
            timeSeries={activeAnalysis.time_series || null}
            duration={duration || activeAnalysis.audio_metadata.duration_s || 10}
            currentTime={currentTime}
            onSeek={onSeek}
            flaws={flaws}
          />

          {/* Acoustic Telemetry Card */}
          <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Acoustic & Alignment Telemetry
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                <span className="text-slate-400 text-[11px]">Speech Rate</span>
                <div className="text-base font-bold text-white mt-0.5">
                  {activeAnalysis.features_summary.average_wpm.toFixed(1)} WPM
                </div>
              </div>
              <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                <span className="text-slate-400 text-[11px]">Pitch Median / Std</span>
                <div className="text-base font-bold text-white mt-0.5">
                  {activeAnalysis.features_summary.pitch_median_hz.toFixed(0)} Hz / {activeAnalysis.features_summary.pitch_std_hz.toFixed(0)}
                </div>
              </div>
              <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                <span className="text-slate-400 text-[11px]">Pitch Inflection</span>
                <div className="text-base font-bold text-white mt-0.5">
                  {activeAnalysis.features_summary.pitch_semitone_std.toFixed(2)} ST
                </div>
              </div>
              <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                <span className="text-slate-400 text-[11px]">Total Pauses</span>
                <div className="text-base font-bold text-white mt-0.5">
                  {activeAnalysis.features_summary.pause_count} ({activeAnalysis.features_summary.total_pause_duration_s.toFixed(2)}s)
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column (5 cols): Rubric Scorecard & Flaw Summary */}
        <div className="lg:col-span-5 space-y-6">
          <RubricCard rubricScores={activeAnalysis.rubric_scores} />

          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Grounded Flaws Summary
              </span>
              <button
                onClick={onOpenFlaws}
                className="text-xs text-indigo-400 hover:text-indigo-300 font-medium flex items-center space-x-1"
              >
                <span>Full Flaws View</span>
                <span>→</span>
              </button>
            </div>
            <div className="text-xs text-slate-400">
              {flaws.length === 0 ? (
                <span className="text-emerald-400 font-semibold">
                  0 Flaws detected (Exemplary baseline reference delivery)
                </span>
              ) : (
                <span className="text-rose-400 font-semibold">
                  {flaws.length} anomalous temporal flaw regions grounded in delivery.
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
