"""
VoxContrast - Temporal Flaw Grounding & Anomaly Detection Engine
Detects temporal intervals [start_s, end_s] of speech flaws across pacing, pauses,
energy stability, pitch modulation, and vocal clarity with mathematical explanations.
"""

from __future__ import annotations

import numpy as np
from core.schemas import (
    TimeSeriesData,
    PauseInterval,
    AlignmentStatus,
    DetectedFlaw,
    FeaturesSummary
)
from core.explainer import generate_flaw_explanation

# Calibrated Acoustic Thresholds
THRESH_FAST_WPM = 180.0
THRESH_SLOW_WPM = 85.0
BASELINE_WPM = 140.0

THRESH_PROLONGED_PAUSE_S = 0.50
BASELINE_PAUSE_S = 0.40

THRESH_MONOTONE_ENERGY_STD_DB = 2.45
THRESH_UNSTABLE_ENERGY_SPAN_DB = 10.0
BASELINE_ENERGY_STD_DB = 5.5

THRESH_MONOTONE_PITCH_ST = 2.30
THRESH_IRREGULAR_PITCH_ST = 8.50
BASELINE_PITCH_ST = 3.50

THRESH_MUFFLED_CENTROID_HZ = 750.0
BASELINE_CENTROID_HZ = 2100.0

VOICED_RMS_FLOOR_DB = -32.0

# Calibrated Minimum Duration Constraints (seconds)
MIN_DUR_FAST_S = 0.25
MIN_DUR_SLOW_S = 1.20
MIN_DUR_MONOTONE_ENERGY_S = 0.60
MIN_DUR_UNSTABLE_ENERGY_S = 0.50
MIN_DUR_MONOTONE_PITCH_S = 0.60
MIN_DUR_IRREGULAR_PITCH_S = 0.50
MIN_DUR_MUFFLED_S = 0.50


def extract_transcript_slice(
    alignment: AlignmentStatus,
    start_s: float,
    end_s: float
) -> str:
    """Finds words that overlap with the temporal flaw interval [start_s, end_s]."""
    if not alignment.aligned_words or alignment.status == "FAILED":
        return f"[Speech region {start_s:.2f}s - {end_s:.2f}s]"

    matching_words = []
    for w in alignment.aligned_words:
        if not (w.end_s < start_s or w.start_s > end_s):
            matching_words.append(w.word)

    if matching_words:
        return " ".join(matching_words)
    return f"[Speech region {start_s:.2f}s - {end_s:.2f}s]"


def merge_and_filter_intervals(
    intervals: list[tuple[float, float, float]],
    max_gap_s: float = 0.25,
    min_duration_s: float = 0.40
) -> list[tuple[float, float, float]]:
    """
    Merges contiguous intervals separated by <= max_gap_s and discards
    transient detections shorter than min_duration_s.
    """
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged: list[list[float]] = []

    for start_s, end_s, val in intervals:
        if not merged:
            merged.append([start_s, end_s, val])
        else:
            prev_start, prev_end, prev_val = merged[-1]
            if start_s - prev_end <= max_gap_s:
                merged[-1][1] = max(prev_end, end_s)
                merged[-1][2] = max(prev_val, val)
            else:
                merged.append([start_s, end_s, val])

    filtered = []
    for start_s, end_s, val in merged:
        if (end_s - start_s) >= min_duration_s:
            filtered.append((round(start_s, 2), round(end_s, 2), round(val, 2)))

    return filtered


def detect_temporal_flaws(
    time_series: TimeSeriesData,
    pauses: list[PauseInterval],
    alignment: AlignmentStatus,
    features_summary: FeaturesSummary
) -> list[DetectedFlaw]:
    """
    Runs multi-pass temporal flaw grounding across all 5 dimensions.
    Returns list of validated DetectedFlaw objects sorted chronologically.
    """
    flaws: list[DetectedFlaw] = []
    time_stamps = time_series.time_stamps
    dt = time_series.time_step_s
    n_frames = len(time_stamps)
    flaw_counter = 1

    # 1. Pacing: Excessive Speed
    fast_intervals = []
    in_fast = False
    f_start = 0.0
    f_peak = 0.0

    for i, wpm in enumerate(time_series.speech_rate_wpm):
        t = time_stamps[i]
        if wpm >= THRESH_FAST_WPM:
            if not in_fast:
                in_fast = True
                f_start = t
                f_peak = wpm
            else:
                f_peak = max(f_peak, wpm)
        else:
            if in_fast:
                in_fast = False
                fast_intervals.append((f_start, t, f_peak))

    if in_fast:
        fast_intervals.append((f_start, time_stamps[-1], f_peak))

    final_fast = merge_and_filter_intervals(fast_intervals, max_gap_s=0.25, min_duration_s=MIN_DUR_FAST_S)

    for start_s, end_s, observed_wpm in final_fast:
        start_s, end_s = round(start_s, 2), round(end_s, 2)
        delta = observed_wpm - THRESH_FAST_WPM
        severity = "severe" if delta > 40.0 else ("moderate" if delta > 20.0 else "minor")
        transcript_slice = extract_transcript_slice(alignment, start_s, end_s)
        expl, caveat = generate_flaw_explanation(
            "excessive_speed", start_s, end_s, observed_wpm,
            BASELINE_WPM, THRESH_FAST_WPM, "WPM", severity
        )
        flaws.append(
            DetectedFlaw(
                flaw_id=f"flaw_pacing_{flaw_counter:03d}",
                category="excessive_speed",
                start_s=start_s,
                end_s=end_s,
                transcript_slice=transcript_slice,
                observed_value=observed_wpm,
                baseline_value=BASELINE_WPM,
                threshold_value=THRESH_FAST_WPM,
                unit="WPM",
                severity=severity,
                confidence=0.92,
                explanation=expl,
                caveat=caveat
            )
        )
        flaw_counter += 1

    # 1b. Pacing: Slow Speech / Slow Delivery (requires active voiced speech away from pauses)
    slow_intervals = []
    in_slow = False
    s_start = 0.0
    s_min = 200.0

    for i, wpm in enumerate(time_series.speech_rate_wpm):
        t = time_stamps[i]
        # Must be actively voiced speech and not in transition margins adjacent to silence pauses
        is_voiced = (time_series.f0_contour[i] > 0 and time_series.rms_energy_db[i] > -28.0)
        is_near_pause = any((p.start_s - 0.35) <= t <= (p.end_s + 0.35) for p in pauses)
        if is_voiced and not is_near_pause and 0.0 < wpm <= THRESH_SLOW_WPM:
            if not in_slow:
                in_slow = True
                s_start = t
                s_min = wpm
            else:
                s_min = min(s_min, wpm)
        else:
            if in_slow:
                in_slow = False
                slow_intervals.append((s_start, t, s_min))

    if in_slow:
        slow_intervals.append((s_start, time_stamps[-1], s_min))

    final_slow = merge_and_filter_intervals(slow_intervals, max_gap_s=0.35, min_duration_s=1.20)

    for start_s, end_s, observed_wpm in final_slow:
        start_s, end_s = round(start_s, 2), round(end_s, 2)
        delta = THRESH_SLOW_WPM - observed_wpm
        severity = "severe" if delta > 30.0 else ("moderate" if delta > 15.0 else "minor")
        transcript_slice = extract_transcript_slice(alignment, start_s, end_s)
        expl, caveat = generate_flaw_explanation(
            "slow_speech", start_s, end_s, observed_wpm,
            BASELINE_WPM, THRESH_SLOW_WPM, "WPM", severity
        )
        flaws.append(
            DetectedFlaw(
                flaw_id=f"flaw_pacing_{flaw_counter:03d}",
                category="slow_speech",
                start_s=start_s,
                end_s=end_s,
                transcript_slice=transcript_slice,
                observed_value=observed_wpm,
                baseline_value=BASELINE_WPM,
                threshold_value=THRESH_SLOW_WPM,
                unit="WPM",
                severity=severity,
                confidence=0.90,
                explanation=expl,
                caveat=caveat
            )
        )
        flaw_counter += 1

    # 2. Pause Control: Unnatural / Prolonged Pauses
    for pause in pauses:
        if pause.is_excessive or pause.duration_s >= THRESH_PROLONGED_PAUSE_S:
            delta = pause.duration_s - THRESH_PROLONGED_PAUSE_S
            severity = "severe" if delta > 1.2 else ("moderate" if delta > 0.5 else "minor")
            transcript_slice = extract_transcript_slice(
                alignment, max(0.0, pause.start_s - 0.4), min(time_stamps[-1], pause.end_s + 0.4)
            )
            expl, caveat = generate_flaw_explanation(
                "unnatural_pause", pause.start_s, pause.end_s,
                pause.duration_s, BASELINE_PAUSE_S, THRESH_PROLONGED_PAUSE_S,
                "seconds", severity
            )
            flaws.append(
                DetectedFlaw(
                    flaw_id=f"flaw_pause_{flaw_counter:03d}",
                    category="unnatural_pause",
                    start_s=round(pause.start_s, 2),
                    end_s=round(pause.end_s, 2),
                    transcript_slice=transcript_slice,
                    observed_value=round(pause.duration_s, 2),
                    baseline_value=BASELINE_PAUSE_S,
                    threshold_value=THRESH_PROLONGED_PAUSE_S,
                    unit="s",
                    severity=severity,
                    confidence=0.95,
                    explanation=expl,
                    caveat=caveat
                )
            )
            flaw_counter += 1

    # Sliding window configuration for acoustic anomaly grounding
    win_frames = int(1.2 / dt)
    hop_frames = max(1, int(0.2 / dt))

    # 3. Vocal Energy: Monotone Energy vs Unstable Energy
    monotone_energy_intervals = []
    unstable_energy_intervals = []

    # Identify local syllabic RMS peaks
    rms_arr = np.array(time_series.rms_energy_db)
    f0_arr = np.array(time_series.f0_contour)
    syllable_peaks: list[tuple[float, float]] = []  # (timestamp, peak_db)

    for i in range(1, n_frames - 1):
        if rms_arr[i] > VOICED_RMS_FLOOR_DB and f0_arr[i] > 0:
            if rms_arr[i] >= rms_arr[i - 1] and rms_arr[i] >= rms_arr[i + 1]:
                syllable_peaks.append((time_stamps[i], float(rms_arr[i])))

    # Detect unstable energy from consecutive peak-to-peak volume swings
    for (t1, v1), (t2, v2) in zip(syllable_peaks[:-1], syllable_peaks[1:]):
        if t2 - t1 <= 1.20:
            peak_span = abs(v2 - v1)
            if peak_span >= 10.0:
                unstable_energy_intervals.append((max(0.0, (t1 + t2) / 2.0 - 0.10), min(time_stamps[-1], t2 + 0.20), peak_span))

    # Detect monotone energy from sustained lack of dynamic inflection across voiced speech
    for i in range(0, n_frames - win_frames, hop_frames):
        w_rms = time_series.rms_energy_db[i:i + win_frames]
        w_f0 = time_series.f0_contour[i:i + win_frames]
        v_idx = [j for j, (db, f) in enumerate(zip(w_rms, w_f0)) if f > 0 and db > VOICED_RMS_FLOOR_DB]

        if len(v_idx) >= int(win_frames * 0.40):
            e_std = float(np.std([w_rms[j] for j in v_idx]))
            w_start = time_stamps[i + v_idx[0]]
            w_end = time_stamps[i + v_idx[-1]] + dt

            if e_std < THRESH_MONOTONE_ENERGY_STD_DB:
                monotone_energy_intervals.append((w_start, w_end, e_std))

    for start_s, end_s, obs_std in merge_and_filter_intervals(monotone_energy_intervals, min_duration_s=MIN_DUR_MONOTONE_ENERGY_S):
        delta = THRESH_MONOTONE_ENERGY_STD_DB - obs_std
        severity = "severe" if delta > 0.8 else ("moderate" if delta > 0.4 else "minor")
        transcript_slice = extract_transcript_slice(alignment, start_s, end_s)
        expl, caveat = generate_flaw_explanation(
            "monotone_energy", start_s, end_s, obs_std,
            BASELINE_ENERGY_STD_DB, THRESH_MONOTONE_ENERGY_STD_DB, "dB std", severity
        )
        flaws.append(
            DetectedFlaw(
                flaw_id=f"flaw_energy_{flaw_counter:03d}",
                category="monotone_energy",
                start_s=start_s,
                end_s=end_s,
                transcript_slice=transcript_slice,
                observed_value=obs_std,
                baseline_value=BASELINE_ENERGY_STD_DB,
                threshold_value=THRESH_MONOTONE_ENERGY_STD_DB,
                unit="dB",
                severity=severity,
                confidence=0.88,
                explanation=expl,
                caveat=caveat
            )
        )
        flaw_counter += 1

    for start_s, end_s, obs_span in merge_and_filter_intervals(unstable_energy_intervals, max_gap_s=0.75, min_duration_s=MIN_DUR_UNSTABLE_ENERGY_S):
        delta = obs_span - THRESH_UNSTABLE_ENERGY_SPAN_DB
        severity = "severe" if delta > 6.0 else ("moderate" if delta > 3.0 else "minor")
        transcript_slice = extract_transcript_slice(alignment, start_s, end_s)
        expl, caveat = generate_flaw_explanation(
            "unstable_energy", start_s, end_s, obs_span,
            8.0, THRESH_UNSTABLE_ENERGY_SPAN_DB, "dB range", severity
        )
        flaws.append(
            DetectedFlaw(
                flaw_id=f"flaw_energy_{flaw_counter:03d}",
                category="unstable_energy",
                start_s=start_s,
                end_s=end_s,
                transcript_slice=transcript_slice,
                observed_value=obs_span,
                baseline_value=8.0,
                threshold_value=THRESH_UNSTABLE_ENERGY_SPAN_DB,
                unit="dB",
                severity=severity,
                confidence=0.89,
                explanation=expl,
                caveat=caveat
            )
        )
        flaw_counter += 1

    # 4. Pitch Modulation: Monotone Pitch vs Pitch Irregularity
    win_pitch_frames = max(6, int(0.8 / dt))
    monotone_pitch_intervals = []
    for i in range(0, n_frames - win_pitch_frames, hop_frames):
        w_f0 = time_series.f0_contour[i:i + win_pitch_frames]
        v_idx = [j for j, f in enumerate(w_f0) if f > 0]

        if len(v_idx) >= int(win_pitch_frames * 0.40):
            st_std = float(np.std([time_series.f0_semitones[i + j] for j in v_idx]))
            w_start = time_stamps[i + v_idx[0]]
            w_end = time_stamps[i + v_idx[-1]] + dt

            if st_std < THRESH_MONOTONE_PITCH_ST:
                monotone_pitch_intervals.append((w_start, w_end, st_std))

    for start_s, end_s, obs_st in merge_and_filter_intervals(monotone_pitch_intervals, min_duration_s=MIN_DUR_MONOTONE_PITCH_S):
        delta = THRESH_MONOTONE_PITCH_ST - obs_st
        severity = "severe" if delta > 0.5 else ("moderate" if delta > 0.25 else "minor")
        transcript_slice = extract_transcript_slice(alignment, start_s, end_s)
        expl, caveat = generate_flaw_explanation(
            "monotone_pitch", start_s, end_s, obs_st,
            BASELINE_PITCH_ST, THRESH_MONOTONE_PITCH_ST, "semitones", severity
        )
        flaws.append(
            DetectedFlaw(
                flaw_id=f"flaw_pitch_{flaw_counter:03d}",
                category="monotone_pitch",
                start_s=start_s,
                end_s=end_s,
                transcript_slice=transcript_slice,
                observed_value=obs_st,
                baseline_value=BASELINE_PITCH_ST,
                threshold_value=THRESH_MONOTONE_PITCH_ST,
                unit="ST",
                severity=severity,
                confidence=0.91,
                explanation=expl,
                caveat=caveat
            )
        )
        flaw_counter += 1

    # Focused window for dynamic pitch irregularity (sustained vocal tremor / octave breaking)
    irr_win_frames = max(8, int(0.60 / dt))
    irregular_pitch_intervals = []
    for i in range(0, n_frames - irr_win_frames, 1):
        w_f0 = time_series.f0_contour[i:i + irr_win_frames]
        v_idx = [j for j, f in enumerate(w_f0) if f > 0]
        if len(v_idx) >= int(irr_win_frames * 0.60):
            st_std = float(np.std([time_series.f0_semitones[i + j] for j in v_idx]))
            if st_std > THRESH_IRREGULAR_PITCH_ST:
                w_start = time_stamps[i + v_idx[0]]
                w_end = time_stamps[i + v_idx[-1]] + dt
                irregular_pitch_intervals.append((w_start, w_end, st_std))

    for start_s, end_s, obs_st in merge_and_filter_intervals(irregular_pitch_intervals, max_gap_s=0.20, min_duration_s=MIN_DUR_IRREGULAR_PITCH_S):
        delta = obs_st - THRESH_IRREGULAR_PITCH_ST
        severity = "severe" if delta > 2.0 else ("moderate" if delta > 1.0 else "minor")
        transcript_slice = extract_transcript_slice(alignment, start_s, end_s)
        expl, caveat = generate_flaw_explanation(
            "pitch_irregularity", start_s, end_s, obs_st,
            BASELINE_PITCH_ST, THRESH_IRREGULAR_PITCH_ST, "semitones", severity
        )
        flaws.append(
            DetectedFlaw(
                flaw_id=f"flaw_pitch_{flaw_counter:03d}",
                category="pitch_irregularity",
                start_s=start_s,
                end_s=end_s,
                transcript_slice=transcript_slice,
                observed_value=obs_st,
                baseline_value=BASELINE_PITCH_ST,
                threshold_value=THRESH_IRREGULAR_PITCH_ST,
                unit="ST",
                severity=severity,
                confidence=0.87,
                explanation=expl,
                caveat=caveat
            )
        )
        flaw_counter += 1

    # 5. Vocal Clarity: Muffled Articulation (Low Spectral Centroid in Voiced Speech)
    muffled_frames = []
    for i in range(n_frames):
        if time_series.f0_contour[i] > 0 and time_series.rms_energy_db[i] > VOICED_RMS_FLOOR_DB:
            if time_series.spectral_centroid_hz[i] < THRESH_MUFFLED_CENTROID_HZ:
                muffled_frames.append((time_stamps[i], time_stamps[i] + dt, time_series.spectral_centroid_hz[i]))

    for start_s, end_s, obs_cent in merge_and_filter_intervals(muffled_frames, max_gap_s=0.35, min_duration_s=MIN_DUR_MUFFLED_S):
        delta = THRESH_MUFFLED_CENTROID_HZ - obs_cent
        severity = "severe" if delta > 300.0 else ("moderate" if delta > 150.0 else "minor")
        transcript_slice = extract_transcript_slice(alignment, start_s, end_s)
        expl, caveat = generate_flaw_explanation(
            "muffled_articulation", start_s, end_s, obs_cent,
            BASELINE_CENTROID_HZ, THRESH_MUFFLED_CENTROID_HZ, "Hz", severity
        )
        flaws.append(
            DetectedFlaw(
                flaw_id=f"flaw_clarity_{flaw_counter:03d}",
                category="muffled_articulation",
                start_s=start_s,
                end_s=end_s,
                transcript_slice=transcript_slice,
                observed_value=obs_cent,
                baseline_value=BASELINE_CENTROID_HZ,
                threshold_value=THRESH_MUFFLED_CENTROID_HZ,
                unit="Hz",
                severity=severity,
                confidence=0.92,
                explanation=expl,
                caveat=caveat
            )
        )
        flaw_counter += 1

    # Sort all flaws chronologically by start timestamp
    flaws.sort(key=lambda f: f.start_s)
    return flaws
