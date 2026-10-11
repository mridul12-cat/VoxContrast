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

const API_BASE = "";

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export async function fetchDatasetManifest(): Promise<DatasetManifest> {
  const res = await fetch(`${API_BASE}/api/dataset/manifest`);
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
