from __future__ import annotations

import time
import uuid
import numpy as np

from core.audio_io import process_audio
from core.features import extract_acoustic_contours, summarize_features
from core.alignment import align_transcript_to_audio
from core.normalizer import SpeakerNormalizer
from core.detector import detect_temporal_flaws
from core.rubric import calculate_rubric_scores
from core.schemas import AnalysisResponse, ComparisonResponse


def analyze_speech(
    audio_input: bytes | str | np.ndarray,
    transcript: str = "",
    analysis_id: str | None = None
) -> AnalysisResponse:
    """
    Executes full end-to-end speech evaluation pipeline:
    1. Ingestion, 16kHz resampling, mono downmix, peak normalization, silence/SNR checks
    2. 50ms-resolution acoustic feature contour extraction (F0, RMS, pauses, spectral centroid)
    3. Transcript forced alignment / VAD phrase segmentation
    4. Speaker normalization (semitone F0, normalized energy)
    5. Temporal flaw grounding with causal mathematical explanations
    6. Deterministic 5-dimension rubric scoring
    """
    if analysis_id is None:
        analysis_id = f"ana_{uuid.uuid4().hex[:10]}"

    # Stage 1: Audio Ingestion & Normalization
    audio_16k, metadata = process_audio(audio_input)

    # Stage 2: Acoustic Feature Extraction
    time_series, pauses = extract_acoustic_contours(audio_16k, sr=16000)

    # Stage 3: Forced Alignment & Segmentation
    alignment_status, time_series = align_transcript_to_audio(
        transcript=transcript,
        time_series=time_series,
        pauses=pauses,
        audio_duration_s=metadata.duration_s
    )

    # Stage 4: Feature Summary & Speaker Normalization
    summary = summarize_features(time_series, pauses)
    speaker_baseline = SpeakerNormalizer.compute_speaker_baseline(time_series)

    # Stage 5: Temporal Flaw Grounding
    detected_flaws = detect_temporal_flaws(
        time_series=time_series,
        pauses=pauses,
        alignment=alignment_status,
        features_summary=summary
    )

    # Stage 6: Deterministic Rubric Scoring
    rubric_scores = calculate_rubric_scores(
        features=summary,
        detected_flaws=detected_flaws,
        metadata=metadata
    )

    return AnalysisResponse(
        analysis_id=analysis_id,
        audio_metadata=metadata,
        alignment_status=alignment_status,
        features_summary=summary,
        rubric_scores=rubric_scores,
        detected_flaws=detected_flaws,
        time_series=time_series,
        speaker_baseline=speaker_baseline
    )


def compare_speech_pair(
    baseline_audio: bytes | str | np.ndarray,
    flawed_audio: bytes | str | np.ndarray,
    transcript: str = "",
    pair_id: str | None = None
) -> ComparisonResponse:
    """
    Performs contrastive evaluation of an ideal baseline delivery versus a flawed delivery
    of the identical speech text.
    """
    if pair_id is None:
        pair_id = f"comp_{uuid.uuid4().hex[:8]}"

    baseline_ana = analyze_speech(baseline_audio, transcript, f"{pair_id}_baseline")
    flawed_ana = analyze_speech(flawed_audio, transcript, f"{pair_id}_flawed")

    score_delta = round(flawed_ana.rubric_scores.overall_score - baseline_ana.rubric_scores.overall_score, 1)

    dimension_deltas = {}
    for dim_key in baseline_ana.rubric_scores.dimensions:
        b_score = baseline_ana.rubric_scores.dimensions[dim_key].score
        f_score = flawed_ana.rubric_scores.dimensions[dim_key].score
        dimension_deltas[dim_key] = round(f_score - b_score, 1)

    # Synthesize contrast summary
    contrast_summary = []
    contrast_summary.append({
        "metric": "Average Speech Rate (WPM)",
        "baseline": baseline_ana.features_summary.average_wpm,
        "flawed": flawed_ana.features_summary.average_wpm,
        "delta": round(flawed_ana.features_summary.average_wpm - baseline_ana.features_summary.average_wpm, 1),
        "interpretation": "Pacing variation between deliveries"
    })
    contrast_summary.append({
        "metric": "Total Pause Duration (s)",
        "baseline": baseline_ana.features_summary.total_pause_duration_s,
        "flawed": flawed_ana.features_summary.total_pause_duration_s,
        "delta": round(flawed_ana.features_summary.total_pause_duration_s - baseline_ana.features_summary.total_pause_duration_s, 2),
        "interpretation": "Pause duration and hesitation divergence"
    })
    contrast_summary.append({
        "metric": "Pitch Dynamic Modulation (Semitones Std)",
        "baseline": baseline_ana.features_summary.pitch_semitone_std,
        "flawed": flawed_ana.features_summary.pitch_semitone_std,
        "delta": round(flawed_ana.features_summary.pitch_semitone_std - baseline_ana.features_summary.pitch_semitone_std, 2),
        "interpretation": "Intonation dynamic expression contrast"
    })
    contrast_summary.append({
        "metric": "Voiced Energy Dynamic Range (dB Std)",
        "baseline": baseline_ana.features_summary.energy_std_db,
        "flawed": flawed_ana.features_summary.energy_std_db,
        "delta": round(flawed_ana.features_summary.energy_std_db - baseline_ana.features_summary.energy_std_db, 1),
        "interpretation": "Vocal volume dynamics contrast"
    })

    return ComparisonResponse(
        comparison_id=pair_id,
        baseline_analysis=baseline_ana,
        flawed_analysis=flawed_ana,
        score_delta=score_delta,
        dimension_deltas=dimension_deltas,
        flaw_contrast_summary=contrast_summary
    )
