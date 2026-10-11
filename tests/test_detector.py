"""
VoxContrast - Unit Tests for Temporal Flaw Grounding & Explanations
"""

from __future__ import annotations

import pytest
import numpy as np

from core.detector import (
    detect_temporal_flaws,
    merge_and_filter_intervals,
    THRESH_FAST_WPM,
    THRESH_PROLONGED_PAUSE_S
)
from core.schemas import TimeSeriesData, PauseInterval, AlignmentStatus, FeaturesSummary


def test_merge_and_filter_intervals():
    """Verifies that close intervals (< 250ms gap) merge and sub-300ms blips are filtered."""
    # Two intervals separated by 0.1s -> should merge
    # One short interval of 0.15s -> should be discarded
    raw = [
        (1.0, 1.8, 200.0),
        (1.9, 2.7, 210.0),  # gap is 0.1s <= 0.25s
        (4.0, 4.15, 220.0)  # duration is 0.15s < 0.30s
    ]

    merged = merge_and_filter_intervals(raw, max_gap_s=0.25, min_duration_s=0.30)
    assert len(merged) == 1
    assert merged[0][0] == 1.0
    assert merged[0][1] == 2.7
    assert merged[0][2] == 210.0


def test_excessive_speed_detection():
    """Verifies that sustained high WPM frames are detected as excessive_speed flaw."""
    n_frames = 100
    t_stamps = [round(i * 0.05, 3) for i in range(n_frames)]

    # Normal speed at start, then high speed 220 WPM from frame 20 to 60 (1.0s to 3.0s)
    wpm = [135.0] * n_frames
    for i in range(20, 60):
        wpm[i] = 230.0

    ts = TimeSeriesData(
        time_step_s=0.05,
        time_stamps=t_stamps,
        f0_contour=[140.0] * n_frames,
        f0_semitones=[0.0] * n_frames,
        rms_energy_db=[-22.0] * n_frames,
        speech_rate_wpm=wpm,
        spectral_centroid_hz=[1800.0] * n_frames
    )

    alignment = AlignmentStatus(status="SUCCESS", confidence=0.9, word_count=20, aligned_words=[])
    features = FeaturesSummary(
        average_wpm=160.0,
        pitch_mean_hz=140.0,
        pitch_std_hz=15.0,
        pitch_median_hz=140.0,
        pitch_semitone_std=3.0,
        energy_mean_db=-22.0,
        energy_std_db=5.0,
        pause_count=0,
        total_pause_duration_s=0.0,
        spectral_centroid_mean_hz=1800.0
    )

    flaws = detect_temporal_flaws(ts, [], alignment, features)
    speed_flaws = [f for f in flaws if f.category == "excessive_speed"]

    assert len(speed_flaws) >= 1
    f = speed_flaws[0]
    assert f.start_s == pytest.approx(1.0, abs=0.2)
    assert f.end_s == pytest.approx(3.0, abs=0.2)
    assert f.observed_value >= THRESH_FAST_WPM
    assert "exceeding the calibrated maximum threshold" in f.explanation
    assert len(f.caveat) > 10


def test_unnatural_pause_flaw():
    """Verifies that an excessive pause interval is converted into a grounded flaw."""
    n_frames = 80
    t_stamps = [round(i * 0.05, 3) for i in range(n_frames)]
    ts = TimeSeriesData(
        time_step_s=0.05,
        time_stamps=t_stamps,
        f0_contour=[140.0] * n_frames,
        f0_semitones=[0.0] * n_frames,
        rms_energy_db=[-22.0] * n_frames,
        speech_rate_wpm=[140.0] * n_frames,
        spectral_centroid_hz=[1800.0] * n_frames
    )

    pauses = [
        PauseInterval(start_s=1.5, end_s=3.8, duration_s=2.3, is_excessive=True)
    ]
    alignment = AlignmentStatus(status="SUCCESS", confidence=0.9, word_count=10, aligned_words=[])
    features = FeaturesSummary(
        average_wpm=130.0,
        pitch_mean_hz=140.0,
        pitch_std_hz=15.0,
        pitch_median_hz=140.0,
        pitch_semitone_std=3.0,
        energy_mean_db=-22.0,
        energy_std_db=5.0,
        pause_count=1,
        total_pause_duration_s=2.3,
        spectral_centroid_mean_hz=1800.0
    )

    flaws = detect_temporal_flaws(ts, pauses, alignment, features)
    pause_flaws = [f for f in flaws if f.category == "unnatural_pause"]

    assert len(pause_flaws) == 1
    f = pause_flaws[0]
    assert f.start_s == 1.5
    assert f.end_s == 3.8
    assert f.observed_value == 2.3
    assert "unvoiced silence interval" in f.explanation
