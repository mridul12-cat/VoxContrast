import React, { useState, useEffect, useRef } from "react";
import { Header } from "./components/Header";
import { DatasetBrowser } from "./components/DatasetBrowser";
import { WaveformPlayer } from "./components/WaveformPlayer";
import { FeatureTimeline } from "./components/FeatureTimeline";
import { FlawGroundingList } from "./components/FlawGroundingList";
import { RubricCard } from "./components/RubricCard";
import { CompareView } from "./components/CompareView";
import { AudioUpload } from "./components/AudioUpload";
import {
  fetchDatasetManifest,
  comparePair,
  analyzeAudio,
  DatasetManifest,
  ComparisonResponse,
  AnalysisResponse,
} from "./services/api";

export const App: React.FC = () => {
  const [manifest, setManifest] = useState<DatasetManifest | null>(null);
  const [selectedPairId, setSelectedPairId] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [activeAnalysis, setActiveAnalysis] = useState<AnalysisResponse | null>(null);
  const [activeType, setActiveType] = useState<"flawed" | "baseline">("flawed");

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const [showCompare, setShowCompare] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [loading, setLoading] = useState(true);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Initialize dataset manifest
  useEffect(() => {
    fetchDatasetManifest()
      .then((m) => {
        setManifest(m);
        if (m.pairs && m.pairs.length > 0) {
          setSelectedPairId(m.pairs[0].pair_id);
        }
      })
      .catch((err) => console.error("Failed to load manifest:", err))
      .finally(() => setLoading(false));
  }, []);

  // Fetch comparison data when selected pair changes
  useEffect(() => {
    if (!selectedPairId) return;

    setLoading(true);
    comparePair(selectedPairId)
      .then((comp) => {
        setComparison(comp);
        setActiveAnalysis(activeType === "flawed" ? comp.flawed_analysis : comp.baseline_analysis);
      })
      .catch((err) => console.error("Failed to load comparison:", err))
      .finally(() => setLoading(false));
  }, [selectedPairId]);

  // Update active analysis when user toggles between flawed and baseline
  useEffect(() => {
    if (!comparison) return;
    const target = activeType === "flawed" ? comparison.flawed_analysis : comparison.baseline_analysis;
    setActiveAnalysis(target);
  }, [activeType, comparison]);

  // Handle HTML Audio element setup
  useEffect(() => {
    if (!selectedPairId || !manifest) return;
    const pair = manifest.pairs.find((p) => p.pair_id === selectedPairId);
    if (!pair) return;

    const audioUrl = activeType === "flawed" ? `/${pair.flawed_audio}` : `/${pair.baseline_audio}`;

    if (!audioRef.current) {
      audioRef.current = new Audio(audioUrl);
    } else {
      audioRef.current.src = audioUrl;
      audioRef.current.currentTime = 0;
    }

    const audio = audioRef.current;

    const onTimeUpdate = () => setCurrentTime(audio.currentTime);
    const onLoadedMetadata = () => setDuration(audio.duration);
    const onEnded = () => setIsPlaying(false);

    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("ended", onEnded);

    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("ended", onEnded);
      audio.pause();
    };
  }, [selectedPairId, activeType, manifest]);

  const handleTogglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().catch(console.error);
      setIsPlaying(true);
    }
  };

  const handleSeek = (targetSec: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = targetSec;
    setCurrentTime(targetSec);
  };

  const handleCustomUpload = async (file: File, transcript: string) => {
    const analysis = await analyzeAudio(file, transcript);
    setActiveAnalysis(analysis);
    const url = URL.createObjectURL(file);
    if (audioRef.current) {
      audioRef.current.src = url;
      audioRef.current.currentTime = 0;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      <Header onOpenUpload={() => setShowUpload(true)} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Dataset Browser Bar */}
        <DatasetBrowser
          manifest={manifest}
          selectedPairId={selectedPairId}
          onSelectPair={(id) => {
            setSelectedPairId(id);
            setActiveType("flawed");
          }}
          onInspect={() => setShowCompare(true)}
        />

        {loading ? (
          <div className="flex items-center justify-center py-20 space-x-3 text-slate-400 text-sm">
            <span className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></span>
            <span>Processing acoustic features & deterministic rubrics...</span>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left 2 Columns: Waveform, Feature Timeline, Metadata */}
            <div className="lg:col-span-2 space-y-6">
              <WaveformPlayer
                timeSeries={activeAnalysis?.time_series || null}
                duration={duration || activeAnalysis?.audio_metadata.duration_s || 10}
                currentTime={currentTime}
                isPlaying={isPlaying}
                onTogglePlay={handleTogglePlay}
                onSeek={handleSeek}
                activeType={activeType}
                onSwitchType={(type) => {
                  setActiveType(type);
                  setIsPlaying(false);
                }}
              />

              <FeatureTimeline
                timeSeries={activeAnalysis?.time_series || null}
                duration={duration || activeAnalysis?.audio_metadata.duration_s || 10}
                currentTime={currentTime}
                onSeek={handleSeek}
              />

              {/* Acoustic Summary Card */}
              {activeAnalysis && (
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
                        {activeAnalysis.features_summary.pitch_median_hz.toFixed(0)} Hz / ±
                        {activeAnalysis.features_summary.pitch_semitone_std.toFixed(1)} ST
                      </div>
                    </div>
                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                      <span className="text-slate-400 text-[11px]">Pause Count / Total</span>
                      <div className="text-base font-bold text-white mt-0.5">
                        {activeAnalysis.features_summary.pause_count} ({activeAnalysis.features_summary.total_pause_duration_s.toFixed(1)}s)
                      </div>
                    </div>
                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                      <span className="text-slate-400 text-[11px]">Signal-to-Noise Ratio</span>
                      <div className="text-base font-bold text-emerald-400 mt-0.5">
                        {activeAnalysis.audio_metadata.snr_db.toFixed(1)} dB
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column: Flaw Cards & Rubric Card */}
            <div className="space-y-6">
              <RubricCard rubric={activeAnalysis?.rubric_scores || null} />

              <FlawGroundingList
                flaws={activeAnalysis?.detected_flaws || []}
                onSeekToFlaw={handleSeek}
              />
            </div>
          </div>
        )}
      </main>

      {/* Modals */}
      <CompareView
        comparison={comparison}
        onClose={() => setShowCompare(false)}
      />

      <AudioUpload
        isOpen={showUpload}
        onClose={() => setShowUpload(false)}
        onSubmit={handleCustomUpload}
      />
    </div>
  );
};

export default App;
