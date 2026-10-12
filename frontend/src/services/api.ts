/**
 * VoxContrast - Typed API Service Client
 */

export interface AudioMetadata {
  duration_s: number;
  sample_rate: number;
  channels: number;
  snr_db: number;
  clipped_ratio: number;
}

export interface AlignedWord {
  word: string;
  start_s: number;
  end_s: number;
  confidence: number;
}

export interface AlignmentStatus {
  status: "SUCCESS" | "DEGRADED" | "FAILED";
  confidence: number;
  word_count: number;
  aligned_words: AlignedWord[];
  failure_reason?: string;
}

export interface FeaturesSummary {
  average_wpm: number;
  pitch_mean_hz: number;
  pitch_std_hz: number;
  pitch_median_hz: number;
  pitch_semitone_std: number;
  energy_mean_db: number;
  energy_std_db: number;
  pause_count: number;
  total_pause_duration_s: number;
  spectral_centroid_mean_hz: number;
}

export interface TimeSeriesData {
  time_step_s: number;
  time_stamps: number[];
  f0_contour: number[];
  f0_semitones: number[];
  rms_energy_db: number[];
  speech_rate_wpm: number[];
  spectral_centroid_hz: number[];
}

export interface RubricDimensionScore {
  dimension_name: string;
  score: number;
  max_points: number;
  weight: number;
  raw_deduction: number;
  deduction_reasons: string[];
}

export interface RubricScores {
  overall_score: number;
  version: string;
  dimensions: Record<string, RubricDimensionScore>;
}

export interface DetectedFlaw {
  flaw_id: string;
  category: string;
  start_s: number;
  end_s: number;
  transcript_slice: string;
  observed_value: number;
  baseline_value: number;
  threshold_value: number;
  unit: string;
  severity: "minor" | "moderate" | "severe";
  confidence: number;
  explanation: string;
  caveat: string;
}

export interface AnalysisResponse {
  analysis_id: string;
  audio_metadata: AudioMetadata;
  alignment_status: AlignmentStatus;
  features_summary: FeaturesSummary;
  rubric_scores: RubricScores;
  detected_flaws: DetectedFlaw[];
  time_series: TimeSeriesData;
  speaker_baseline?: Record<string, number>;
}

export interface ComparisonResponse {
  comparison_id: string;
  baseline_analysis: AnalysisResponse;
  flawed_analysis: AnalysisResponse;
  score_delta: number;
  dimension_deltas: Record<string, number>;
  flaw_contrast_summary: Array<{
    metric: string;
    baseline: number;
    flawed: number;
    delta: number;
    interpretation: string;
  }>;
}

export interface DatasetManifest {
  dataset_version: string;
  license: string;
  metadata: Record<string, any>;
  speech_texts: Array<{
    text_id: string;
    title: string;
    canonical_transcript: string;
    word_count: number;
    source_provenance: string;
  }>;
  pairs: Array<{
    pair_id: string;
    text_id: string;
    baseline_audio: string;
    flawed_audio: string;
    flaw_category: string;
    severity: string;
    speaker_metadata: Record<string, any>;
    ground_truth_flaws: Array<any>;
  }>;
}

export interface SessionUploadItem {
  upload_id: string;
  filename: string;
  duration_s: number;
  sample_rate: number;
  channels: number;
  snr_db: number;
  file_size_bytes: number;
  transcript: string;
  upload_time: number;
  upload_status: string;
  analysis_status: string;
  has_analysis: boolean;
  has_comparison: boolean;
  reference_baseline_id?: string;
  error_message?: string;
  objectUrl?: string;
}

const API_BASE = "";

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export async function fetchDatasetManifest(dataset: string = "v2"): Promise<DatasetManifest> {
  const res = await fetch(`${API_BASE}/api/dataset/manifest?dataset=${dataset}`);
  if (!res.ok) throw new Error("Failed to fetch manifest");
  return res.json();
}

export async function analyzeAudio(audioFile: File, transcript: string): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("audio_file", audioFile);
  formData.append("transcript", transcript);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Analysis failed");
  }

  return res.json();
}

export async function comparePair(pairId: string): Promise<ComparisonResponse> {
  const formData = new FormData();
  formData.append("pair_id", pairId);

  const res = await fetch(`${API_BASE}/api/compare`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Comparison failed");
  }

  return res.json();
}

export interface BenchmarkCategoryStats {
  mean_iou: number;
  mean_onset_err_s: number;
  mean_offset_err_s: number;
  gt_count: number;
  det_count: number;
  tp_count: number;
  fp_count: number;
  fn_count: number;
  precision_05: number;
  recall_05: number;
  f1_05: number;
}

export interface BenchmarkResults {
  split: string;
  manifest_path: string;
  macro_iou: number;
  category_macro_iou: number;
  mean_onset_error_s: number;
  mean_offset_error_s: number;
  precision_at_05_iou: number;
  recall_at_05_iou: number;
  f1_at_05_iou: number;
  true_positives_05: number;
  false_positives_05: number;
  false_negatives_05: number;
  total_detections: number;
  total_ground_truth_flaws: number;
  category_breakdown: Record<string, BenchmarkCategoryStats>;
}

export async function getBenchmarks(refresh: boolean = false): Promise<BenchmarkResults> {
  const res = await fetch(`${API_BASE}/api/benchmarks${refresh ? "?refresh=true" : ""}`);
  if (!res.ok) throw new Error("Failed to fetch benchmark gates");
  return res.json();
}

// Session API Client
export async function initSessionApi(): Promise<string> {
  const res = await fetch(`${API_BASE}/api/session/init`, { method: "POST" });
  const data = await res.json();
  return data.session_id;
}

export async function fetchSessionUploadsApi(sessionId: string): Promise<SessionUploadItem[]> {
  const res = await fetch(`${API_BASE}/api/session/uploads`, {
    headers: { "X-Session-ID": sessionId },
  });
  const data = await res.json();
  return data.uploads || [];
}

export async function uploadSessionAudioApi(
  sessionId: string,
  file: File,
  transcript: string = ""
): Promise<SessionUploadItem> {
  const formData = new FormData();
  formData.append("audio_file", file);
  formData.append("transcript", transcript);

  const res = await fetch(`${API_BASE}/api/session/upload`, {
    method: "POST",
    headers: { "X-Session-ID": sessionId },
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Upload failed");
  }

  const data = await res.json();
  return data.upload;
}

export async function analyzeSessionUploadApi(
  sessionId: string,
  uploadId: string,
  transcript?: string,
  referenceBaselineId?: string
): Promise<{ upload_id: string; analysis: AnalysisResponse; comparison?: ComparisonResponse }> {
  const formData = new FormData();
  if (transcript) formData.append("transcript", transcript);
  if (referenceBaselineId) formData.append("reference_baseline_id", referenceBaselineId);

  const res = await fetch(`${API_BASE}/api/session/analyze/${uploadId}`, {
    method: "POST",
    headers: { "X-Session-ID": sessionId },
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Analysis failed");
  }

  return res.json();
}

export async function deleteSessionUploadApi(sessionId: string, uploadId: string): Promise<void> {
  await fetch(`${API_BASE}/api/session/upload/${uploadId}`, {
    method: "DELETE",
    headers: { "X-Session-ID": sessionId },
  });
}

export async function clearSessionUploadsApi(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/api/session/clear`, {
    method: "POST",
    headers: { "X-Session-ID": sessionId },
  });
}
