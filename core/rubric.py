from __future__ import annotations

from core.schemas import (
    FeaturesSummary,
    DetectedFlaw,
    AudioMetadata,
    RubricScores,
    RubricDimensionScore
)


def calculate_rubric_scores(
    features: FeaturesSummary,
    detected_flaws: list[DetectedFlaw],
    metadata: AudioMetadata
) -> RubricScores:
    """
    Evaluates speech performance deterministically across 5 dimensions (0 - 100 points).
    Guarantees 0.0000 variation across repeated identical inputs.
    """
    dimensions: dict[str, RubricDimensionScore] = {}

    # ---------------------------------------------------------
    # 1. Pacing & Rate Stability (25 Points)
    # ---------------------------------------------------------
    pacing_max = 25.0
    pacing_deduction = 0.0
    pacing_reasons = []

    avg_wpm = features.average_wpm
    if avg_wpm > 180.0:
        over = avg_wpm - 180.0
        ded = min(15.0, over * 0.4)
        pacing_deduction += ded
        pacing_reasons.append(f"Average speech rate ({avg_wpm:.1f} WPM) exceeds 180 WPM threshold (-{ded:.1f} pts)")
    elif avg_wpm < 110.0:
        under = 110.0 - avg_wpm
        ded = min(15.0, under * 0.3)
        pacing_deduction += ded
        pacing_reasons.append(f"Average speech rate ({avg_wpm:.1f} WPM) falls below 110 WPM threshold (-{ded:.1f} pts)")

    # Deduct for specific grounded speed flaws
    speed_flaws = [f for f in detected_flaws if f.category in ("excessive_speed", "slow_speech")]
    for f in speed_flaws:
        ded = 4.0 if f.severity == "severe" else (2.5 if f.severity == "moderate" else 1.5)
        pacing_deduction += ded
        pacing_reasons.append(f"Detected {f.category} between {f.start_s:.2f}s and {f.end_s:.2f}s (-{ded:.1f} pts)")

    pacing_deduction = min(pacing_max, pacing_deduction)
    pacing_score = max(0.0, pacing_max - pacing_deduction)
    dimensions["pacing_stability"] = RubricDimensionScore(
        dimension_name="Pacing & Rate Stability",
        score=round((pacing_score / pacing_max) * 100.0, 1),
        max_points=pacing_max,
        weight=0.25,
        raw_deduction=round(pacing_deduction, 2),
        deduction_reasons=pacing_reasons
    )

    # ---------------------------------------------------------
    # 2. Pause Control (20 Points)
    # ---------------------------------------------------------
    pause_max = 20.0
    pause_deduction = 0.0
    pause_reasons = []

    prolonged_flaws = [f for f in detected_flaws if f.category == "unnatural_pause"]
    for f in prolonged_flaws:
        ded = 4.5 if f.severity == "severe" else (3.0 if f.severity == "moderate" else 1.5)
        pause_deduction += ded
        pause_reasons.append(f"Awkward/prolonged silence of {f.observed_value:.2f}s at {f.start_s:.2f}s (-{ded:.1f} pts)")

    # Run-on speech check: long duration without any pauses
    if metadata.duration_s > 10.0 and features.pause_count == 0:
        pause_deduction += 5.0
        pause_reasons.append("Run-on delivery without breathing/phrase boundaries over 10+ seconds (-5.0 pts)")

    pause_deduction = min(pause_max, pause_deduction)
    pause_score = max(0.0, pause_max - pause_deduction)
    dimensions["pause_control"] = RubricDimensionScore(
        dimension_name="Pause Control",
        score=round((pause_score / pause_max) * 100.0, 1),
        max_points=pause_max,
        weight=0.20,
        raw_deduction=round(pause_deduction, 2),
        deduction_reasons=pause_reasons
    )

    # ---------------------------------------------------------
    # 3. Vocal Energy & Projection (20 Points)
    # ---------------------------------------------------------
    energy_max = 20.0
    energy_deduction = 0.0
    energy_reasons = []

    # Monotone dynamic range check
    if features.energy_std_db < 2.5:
        ded = min(8.0, (2.5 - features.energy_std_db) * 5.0)
        energy_deduction += ded
        energy_reasons.append(f"Monotone energy dynamic range ({features.energy_std_db:.1f} dB std) (-{ded:.1f} pts)")

    # Clipping penalty
    if metadata.clipped_ratio > 0.005:
        ded = min(6.0, metadata.clipped_ratio * 100.0)
        energy_deduction += ded
        energy_reasons.append(f"Signal clipping detected on {metadata.clipped_ratio * 100:.2f}% of samples (-{ded:.1f} pts)")

    # Deduct for specific energy flaws
    energy_flaws = [f for f in detected_flaws if f.category in ("monotone_energy", "unstable_energy")]
    for f in energy_flaws:
        ded = 3.5 if f.severity == "severe" else (2.0 if f.severity == "moderate" else 1.0)
        energy_deduction += ded
        energy_reasons.append(f"Acoustic flaw: {f.category} at {f.start_s:.2f}s (-{ded:.1f} pts)")

    energy_deduction = min(energy_max, energy_deduction)
    energy_score = max(0.0, energy_max - energy_deduction)
    dimensions["vocal_energy"] = RubricDimensionScore(
        dimension_name="Vocal Energy & Projection",
        score=round((energy_score / energy_max) * 100.0, 1),
        max_points=energy_max,
        weight=0.20,
        raw_deduction=round(energy_deduction, 2),
        deduction_reasons=energy_reasons
    )

    # ---------------------------------------------------------
    # 4. Pitch Modulation & Dynamic Contour (20 Points)
    # ---------------------------------------------------------
    pitch_max = 20.0
    pitch_deduction = 0.0
    pitch_reasons = []

    # Check if voiced pitch exists
    has_voiced = (features.pitch_median_hz > 50.0 and features.pitch_semitone_std >= 0.0)

    if not has_voiced:
        # Rescale: disable pitch dimension if unvoiced whisper or unvoiced audio
        pitch_weight = 0.0
        pitch_score = 100.0
        pitch_reasons.append("Pitch contour unvoiced or unmeasurable; dimension rescaled across remaining rubrics.")
    else:
        pitch_weight = 0.20
        # Semitone standard deviation ideal is [2.0, 5.5] ST
        st_std = features.pitch_semitone_std
        if st_std < 1.2:
            ded = min(10.0, (1.2 - st_std) * 8.0)
            pitch_deduction += ded
            pitch_reasons.append(f"Robot/flat intonation ({st_std:.2f} semitones std) below 1.20 ST threshold (-{ded:.1f} pts)")
        elif st_std > 7.0:
            ded = min(10.0, (st_std - 7.0) * 3.0)
            pitch_deduction += ded
            pitch_reasons.append(f"Excessive pitch dispersion ({st_std:.2f} semitones std) exceeding 7.00 ST (-{ded:.1f} pts)")

        pitch_flaws = [f for f in detected_flaws if f.category in ("monotone_pitch", "pitch_irregularity")]
        for f in pitch_flaws:
            ded = 3.5 if f.severity == "severe" else (2.0 if f.severity == "moderate" else 1.0)
            pitch_deduction += ded
            pitch_reasons.append(f"Acoustic flaw: {f.category} at {f.start_s:.2f}s (-{ded:.1f} pts)")

        pitch_deduction = min(pitch_max, pitch_deduction)
        pitch_score = max(0.0, pitch_max - pitch_deduction)

    dimensions["pitch_modulation"] = RubricDimensionScore(
        dimension_name="Pitch Modulation & Inflection",
        score=round((pitch_score / pitch_max) * 100.0 if pitch_weight > 0 else 100.0, 1),
        max_points=pitch_max,
        weight=pitch_weight,
        raw_deduction=round(pitch_deduction, 2),
        deduction_reasons=pitch_reasons
    )

    # ---------------------------------------------------------
    # 5. Vocal Clarity & Articulation (15 Points)
    # ---------------------------------------------------------
    clarity_max = 15.0
    clarity_deduction = 0.0
    clarity_reasons = []

    cent = features.spectral_centroid_mean_hz
    if cent < 1250.0:
        ded = min(10.0, ((1250.0 - cent) / 100.0) * 1.5)
        clarity_deduction += ded
        clarity_reasons.append(f"Low spectral centroid ({cent:.0f} Hz) indicating muffled articulation (-{ded:.1f} pts)")

    clarity_flaws = [f for f in detected_flaws if f.category in ("vocal_clarity", "muffled_articulation")]
    for f in clarity_flaws:
        ded = 3.0 if f.severity == "severe" else (1.5 if f.severity == "moderate" else 1.0)
        clarity_deduction += ded
        clarity_reasons.append(f"Muffled articulation grounded at {f.start_s:.2f}s (-{ded:.1f} pts)")

    clarity_deduction = min(clarity_max, clarity_deduction)
    clarity_score = max(0.0, clarity_max - clarity_deduction)
    dimensions["vocal_clarity"] = RubricDimensionScore(
        dimension_name="Vocal Clarity & Articulation",
        score=round((clarity_score / clarity_max) * 100.0, 1),
        max_points=clarity_max,
        weight=0.15,
        raw_deduction=round(clarity_deduction, 2),
        deduction_reasons=clarity_reasons
    )

    # ---------------------------------------------------------
    # Final Overall Score Calculation
    # ---------------------------------------------------------
    total_weight = sum(dim.weight for dim in dimensions.values())
    if total_weight > 0:
        overall = sum(dim.score * dim.weight for dim in dimensions.values()) / total_weight
    else:
        overall = 100.0

    overall = round(max(0.0, min(100.0, overall)), 1)

    return RubricScores(
        overall_score=overall,
        version="rubric_v1.0",
        dimensions=dimensions
    )
