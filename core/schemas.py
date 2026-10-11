from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class AudioMetadata(BaseModel):
    duration_s: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")
    channels: int = Field(default=1, description="Audio channel count (normalized to mono)")
    snr_db: float = Field(..., description="Estimated Signal-to-Noise Ratio in dB")
    clipped_ratio: float = Field(default=0.0, description="Ratio of clipped samples exceeding dynamic range")


class AlignedWord(BaseModel):
    word: str
    start_s: float
    end_s: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AlignmentStatus(BaseModel):
    status: Literal["SUCCESS", "DEGRADED", "FAILED"] = "SUCCESS"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    word_count: int
    aligned_words: List[AlignedWord] = Field(default_factory=list)
    failure_reason: Optional[str] = None


class PauseInterval(BaseModel):
    start_s: float
    end_s: float
    duration_s: float
    is_excessive: bool = False


class FeaturesSummary(BaseModel):
    average_wpm: float
    pitch_mean_hz: float
    pitch_std_hz: float
    pitch_median_hz: float
    pitch_semitone_std: float
    energy_mean_db: float
    energy_std_db: float
    pause_count: int
    total_pause_duration_s: float
    spectral_centroid_mean_hz: float


class TimeSeriesData(BaseModel):
    time_step_s: float = 0.05
    time_stamps: List[float]
    f0_contour: List[float]
    f0_semitones: List[float]
    rms_energy_db: List[float]
    speech_rate_wpm: List[float]
    spectral_centroid_hz: List[float]


class RubricDimensionScore(BaseModel):
    dimension_name: str
    score: float = Field(..., ge=0.0, le=100.0)
    max_points: float
    weight: float
    raw_deduction: float
    deduction_reasons: List[str] = Field(default_factory=list)


class RubricScores(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=100.0)
    version: str = "rubric_v1.0"
    dimensions: Dict[str, RubricDimensionScore]


class DetectedFlaw(BaseModel):
    flaw_id: str
    category: Literal[
        "excessive_speed",
        "slow_speech",
        "unnatural_pause",
        "monotone_energy",
        "unstable_energy",
        "monotone_pitch",
        "pitch_irregularity",
        "vocal_clarity",
        "muffled_articulation"
    ]
    start_s: float
    end_s: float
    transcript_slice: str
    observed_value: float
    baseline_value: float
    threshold_value: float
    unit: str
    severity: Literal["minor", "moderate", "severe"]
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    explanation: str
    caveat: str


class GroundTruthFlaw(BaseModel):
    flaw_id: str
    category: str
    start_s: float
    end_s: float
    transcript_slice: str
    observed_value: Optional[float] = None
    baseline_value: Optional[float] = None
    severity: str
    annotation_method: str = "expert_human_verification_with_forced_alignment"


class SpeakerMetadata(BaseModel):
    speaker_id: str
    gender: Literal["male", "female", "neutral"] = "neutral"
    native_language: str = "en"
    median_f0_hz: Optional[float] = None


class ContrastivePairMeta(BaseModel):
    pair_id: str
    text_id: str
    baseline_audio: str
    flawed_audio: str
    flaw_category: str
    severity: str
    speaker_metadata: SpeakerMetadata
    ground_truth_flaws: List[GroundTruthFlaw]


class SpeechTextMeta(BaseModel):
    text_id: str
    title: str
    canonical_transcript: str
    word_count: int
    source_provenance: str


class DatasetManifest(BaseModel):
    dataset_version: str = "1.0.0"
    license: str
    metadata: Dict[str, Any]
    speech_texts: List[SpeechTextMeta]
    pairs: List[ContrastivePairMeta]


class AnalysisResponse(BaseModel):
    analysis_id: str
    audio_metadata: AudioMetadata
    alignment_status: AlignmentStatus
    features_summary: FeaturesSummary
    rubric_scores: RubricScores
    detected_flaws: List[DetectedFlaw]
    time_series: TimeSeriesData
    speaker_baseline: Optional[Dict[str, float]] = None


class ComparisonResponse(BaseModel):
    comparison_id: str
    baseline_analysis: AnalysisResponse
    flawed_analysis: AnalysisResponse
    score_delta: float
    dimension_deltas: Dict[str, float]
    flaw_contrast_summary: List[Dict[str, Any]]
