import React, { useState, useEffect, useRef } from "react";
import { Header } from "./components/Header";
import { PrimaryNavigation, TabId } from "./components/PrimaryNavigation";
import { OverviewPage } from "./pages/OverviewPage";
import { CompareAudioPage } from "./pages/CompareAudioPage";
import { FlawAnalysisPage } from "./pages/FlawAnalysisPage";
import { DatasetExplorerPage } from "./pages/DatasetExplorerPage";
import { SystemPage } from "./pages/SystemPage";
import { AudioUpload } from "./components/AudioUpload";
import {
  fetchDatasetManifest,
  comparePair,
  analyzeAudio,
  initSessionApi,
  fetchSessionUploadsApi,
  uploadSessionAudioApi,
  analyzeSessionUploadApi,
  deleteSessionUploadApi,
  clearSessionUploadsApi,
  DatasetManifest,
  ComparisonResponse,
  AnalysisResponse,
  SessionUploadItem,
} from "./services/api";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [manifest, setManifest] = useState<DatasetManifest | null>(null);
  const [selectedPairId, setSelectedPairId] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [activeAnalysis, setActiveAnalysis] = useState<AnalysisResponse | null>(null);
  const [activeType, setActiveType] = useState<"flawed" | "baseline">("flawed");

  // Session state (strictly session-scoped, not persisted to localStorage/IndexedDB)
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionUploads, setSessionUploads] = useState<SessionUploadItem[]>([]);
  const [activeSessionUploadId, setActiveSessionUploadId] = useState<string | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showUpload, setShowUpload] = useState(false);
  const [loading, setLoading] = useState(true);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Initialize session namespace on mount
  useEffect(() => {
    initSessionApi()
      .then((sId) => {
        setSessionId(sId);
        return fetchSessionUploadsApi(sId);
      })
      .then((uploads) => {
        setSessionUploads(uploads || []);
      })
      .catch((err) => console.error("Session init failed:", err));
  }, []);

  // Initialize dataset manifest
  useEffect(() => {
    setLoading(true);
    fetchDatasetManifest("v2")
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

  // Handle HTML Audio element setup for authoritative dataset pair
  useEffect(() => {
    if (activeSessionUploadId) return; // Managed by session upload audio loader
    if (!selectedPairId || !manifest) return;
    const pair = manifest.pairs.find((p) => p.pair_id === selectedPairId);
    if (!pair) return;

    const rawPath = activeType === "flawed" ? pair.flawed_audio : pair.baseline_audio;
    const audioUrl = rawPath.startsWith("/") ? rawPath : `/${rawPath}`;

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
    };
  }, [selectedPairId, activeType, manifest, activeSessionUploadId]);

  const handleTogglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch((e) => console.error(e));
    }
  };

  const handleSeek = (seconds: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = seconds;
      setCurrentTime(seconds);
    }
  };

  // Upload handler via session API
  const handleUploadSubmit = async (file: File, transcript: string) => {
    setLoading(true);
    try {
      if (sessionId) {
        const uploadItem = await uploadSessionAudioApi(sessionId, file, transcript);
        setSessionUploads((prev) => [uploadItem, ...prev.filter((u) => u.upload_id !== uploadItem.upload_id)]);
        const res = await analyzeSessionUploadApi(sessionId, uploadItem.upload_id, transcript);
        setActiveAnalysis(res.analysis);
        if (res.comparison) {
          setComparison(res.comparison);
        }
        setActiveSessionUploadId(uploadItem.upload_id);
        const refreshed = await fetchSessionUploadsApi(sessionId);
        setSessionUploads(refreshed);

        if (audioRef.current) {
          audioRef.current.src = `/api/session/audio/${uploadItem.upload_id}`;
          audioRef.current.currentTime = 0;
        }
      } else {
        const result = await analyzeAudio(file, transcript);
        setActiveAnalysis(result);
        if (audioRef.current) {
          audioRef.current.src = URL.createObjectURL(file);
          audioRef.current.currentTime = 0;
        }
      }
      setShowUpload(false);
      setActiveTab("compare");
    } catch (err: any) {
      alert("Analysis failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectUpload = async (uploadId: string) => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const res = await analyzeSessionUploadApi(sessionId, uploadId);
      setActiveAnalysis(res.analysis);
      if (res.comparison) {
        setComparison(res.comparison);
      }
      setActiveSessionUploadId(uploadId);
      if (audioRef.current) {
        audioRef.current.src = `/api/session/audio/${uploadId}`;
        audioRef.current.currentTime = 0;
      }
      setActiveTab("compare");
    } catch (err: any) {
      alert("Failed to load upload: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeUpload = async (uploadId: string) => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const res = await analyzeSessionUploadApi(sessionId, uploadId);
      setActiveAnalysis(res.analysis);
      if (res.comparison) {
        setComparison(res.comparison);
      }
      setActiveSessionUploadId(uploadId);
      const updatedUploads = await fetchSessionUploadsApi(sessionId);
      setSessionUploads(updatedUploads);
      if (audioRef.current) {
        audioRef.current.src = `/api/session/audio/${uploadId}`;
        audioRef.current.currentTime = 0;
      }
      setActiveTab("compare");
    } catch (err: any) {
      alert("Analysis failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectBaselineRef = async (baselineRefId: string) => {
    if (!sessionId || !activeSessionUploadId) return;
    setLoading(true);
    try {
      const res = await analyzeSessionUploadApi(sessionId, activeSessionUploadId, undefined, baselineRefId || undefined);
      setActiveAnalysis(res.analysis);
      if (res.comparison) {
        setComparison(res.comparison);
      }
      const updatedUploads = await fetchSessionUploadsApi(sessionId);
      setSessionUploads(updatedUploads);
    } catch (err: any) {
      alert("Baseline contrast failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const restoreAuthoritativePairPlayer = () => {
    setActiveSessionUploadId(null);
    if (comparison) {
      setActiveAnalysis(activeType === "flawed" ? comparison.flawed_analysis : comparison.baseline_analysis);
    }
    if (selectedPairId && manifest) {
      const pair = manifest.pairs.find((p) => p.pair_id === selectedPairId);
      if (pair && audioRef.current) {
        const rawPath = activeType === "flawed" ? pair.flawed_audio : pair.baseline_audio;
        const audioUrl = rawPath.startsWith("/") ? rawPath : `/${rawPath}`;
        audioRef.current.pause();
        setIsPlaying(false);
        audioRef.current.src = audioUrl;
        audioRef.current.currentTime = 0;
      }
    }
  };

  const handleRemoveUpload = async (uploadId: string) => {
    if (!sessionId) return;
    try {
      await deleteSessionUploadApi(sessionId, uploadId);
      setSessionUploads((prev) => prev.filter((u) => u.upload_id !== uploadId));
      if (activeSessionUploadId === uploadId) {
        restoreAuthoritativePairPlayer();
      }
    } catch (err: any) {
      alert("Failed to remove upload: " + err.message);
    }
  };

  const handleClearSessionUploads = async () => {
    if (!sessionId) return;
    try {
      await clearSessionUploadsApi(sessionId);
      setSessionUploads([]);
      if (activeSessionUploadId) {
        restoreAuthoritativePairPlayer();
      }
    } catch (err: any) {
      alert("Failed to clear session: " + err.message);
    }
  };

  // Cleanup session temp files when browser tab or session ends
  useEffect(() => {
    const handlePageHide = () => {
      if (sessionId && sessionUploads.length > 0) {
        const fd = new FormData();
        fd.append("session_id", sessionId);
        if (navigator.sendBeacon) {
          navigator.sendBeacon("/api/session/clear", fd);
        } else {
          fetch("/api/session/clear", { method: "POST", body: fd, keepalive: true }).catch(() => {});
        }
      }
    };
    window.addEventListener("pagehide", handlePageHide);
    return () => window.removeEventListener("pagehide", handlePageHide);
  }, [sessionId, sessionUploads]);

  const selectedPair = manifest?.pairs.find((p) => p.pair_id === selectedPairId);
  const selectedText = manifest?.speech_texts.find((t) => t.text_id === selectedPair?.text_id);
  const flawCount = activeAnalysis?.detected_flaws.length || 0;
  const activeSessionItem = sessionUploads.find((u) => u.upload_id === activeSessionUploadId);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      <Header onOpenUpload={() => setShowUpload(true)} />

      <PrimaryNavigation
        activeTab={activeTab}
        onSelectTab={(tab) => setActiveTab(tab)}
        flawCount={flawCount}
        sessionUploadCount={sessionUploads.length}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Persistent Pair Context Bar */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-sm">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center border border-indigo-500/30">
              <span className="text-xs font-bold">{activeSessionUploadId ? "SU" : "VC"}</span>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                {activeSessionUploadId ? "Session Upload Workspace" : "Active Contrastive Pair"}
              </div>
              <div className="text-sm font-bold text-white truncate max-w-md">
                {activeSessionUploadId
                  ? `${activeSessionItem?.filename || "Custom Audio"} (Session-Scoped)`
                  : `${selectedText ? selectedText.title : "Speech"} — ${selectedPair?.flaw_category.replace("_", " ")}`}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {activeSessionUploadId ? (
              <button
                onClick={() => setActiveSessionUploadId(null)}
                className="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2 transition"
              >
                Back to Dataset Pair
              </button>
            ) : (
              <select
                value={selectedPairId || ""}
                onChange={(e) => {
                  setSelectedPairId(e.target.value);
                  setActiveType("flawed");
                  setActiveSessionUploadId(null);
                }}
                className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none max-w-xs sm:max-w-md"
              >
                {manifest?.pairs.map((p) => {
                  const text = manifest.speech_texts.find((t) => t.text_id === p.text_id);
                  return (
                    <option key={p.pair_id} value={p.pair_id}>
                      {text ? text.title : p.text_id} — [{p.flaw_category} ({p.severity})]
                    </option>
                  );
                })}
              </select>
            )}
          </div>
        </div>

        {/* Tab Content Panes */}
        {loading && !comparison && !activeAnalysis ? (
          <div className="flex items-center justify-center py-20 space-x-3 text-slate-400 text-sm">
            <span className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></span>
            <span>Loading speech telemetry...</span>
          </div>
        ) : (
          <>
            {activeTab === "overview" && (
              <OverviewPage
                comparison={comparison}
                manifest={manifest}
                onNavigateTab={(tab) => setActiveTab(tab)}
              />
            )}

            {activeTab === "compare" && (
              <CompareAudioPage
                activeAnalysis={activeAnalysis}
                comparison={comparison}
                activeType={activeType}
                isPlaying={isPlaying}
                currentTime={currentTime}
                duration={duration}
                onTogglePlay={handleTogglePlay}
                onSeek={handleSeek}
                onSwitchType={(type) => {
                  setActiveType(type);
                  setIsPlaying(false);
                }}
                onOpenFlaws={() => setActiveTab("flaws")}
                isSessionUpload={Boolean(activeSessionUploadId)}
                selectedBaselineRef={activeSessionItem?.reference_baseline_id || ""}
                onSelectBaselineRef={handleSelectBaselineRef}
                onReturnToDatasetPair={restoreAuthoritativePairPlayer}
              />
            )}

            {activeTab === "flaws" && (
              <FlawAnalysisPage
                flaws={activeAnalysis?.detected_flaws || []}
                currentTime={currentTime}
                duration={duration}
                isPlaying={isPlaying}
                onTogglePlay={handleTogglePlay}
                onSeek={handleSeek}
              />
            )}

            {activeTab === "dataset" && (
              <DatasetExplorerPage
                manifest={manifest}
                selectedPairId={selectedPairId}
                onSelectPair={(id) => {
                  setSelectedPairId(id);
                  setActiveType("flawed");
                  setActiveSessionUploadId(null);
                  setActiveTab("compare");
                }}
                onOpenUpload={() => setShowUpload(true)}
                sessionUploads={sessionUploads}
                onSelectUpload={handleSelectUpload}
                onAnalyzeUpload={handleAnalyzeUpload}
                onRemoveUpload={handleRemoveUpload}
                onClearSessionUploads={handleClearSessionUploads}
              />
            )}

            {activeTab === "system" && (
              <SystemPage
                onClearSessionUploads={handleClearSessionUploads}
                sessionUploadCount={sessionUploads.length}
              />
            )}
          </>
        )}
      </main>

      <AudioUpload
        isOpen={showUpload}
        onClose={() => setShowUpload(false)}
        onSubmit={handleUploadSubmit}
      />
    </div>
  );
};
