"""
VoxContrast - Unit Tests for Acoustic Feature Extraction
"""

from __future__ import annotations

import pytest
import numpy as np

from core.features import (
    compute_yin_f0,
    extract_acoustic_contours,
    summarize_features,
    TIME_STEP_S
)
from core.audio_io import TARGET_SAMPLE_RATE


def test_yin_pitch_accuracy():
    """Verifies that YIN algorithm calculates fundamental frequency within 3Hz of ground truth."""
    sr = TARGET_SAMPLE_RATE
    target_hz = 150.0
    t = np.linspace(0, 0.08, int(0.08 * sr), endpoint=False, dtype=np.float32)
    # Synthetic harmonic tone
    frame = 0.6 * np.sin(2 * np.pi * target_hz * t) + 0.3 * np.sin(2 * np.pi * 2 * target_hz * t)

    detected_f0 = compute_yin_f0(frame, sr)
    assert detected_f0 == pytest.approx(target_hz, abs=3.0)


def test_yin_pitch_female_range():
    """Verifies that YIN works accurately in higher female vocal range (220 Hz)."""
    sr = TARGET_SAMPLE_RATE
    target_hz = 220.0
    t = np.linspace(0, 0.08, int(0.08 * sr), endpoint=False, dtype=np.float32)
    frame = 0.7 * np.sin(2 * np.pi * target_hz * t)

    detected_f0 = compute_yin_f0(frame, sr)
    assert detected_f0 == pytest.approx(target_hz, abs=3.0)


def test_yin_unvoiced_silence():
    """Verifies that silence or white noise returns 0.0 Hz (unvoiced)."""
    sr = TARGET_SAMPLE_RATE
    silence_frame = np.zeros(1024, dtype=np.float32)
    detected_f0 = compute_yin_f0(silence_frame, sr)
    assert detected_f0 == 0.0


def test_pause_interval_detection():
    """Verifies that silence gaps >= 0.4s are accurately detected as pauses."""
    sr = TARGET_SAMPLE_RATE
    # 2 seconds tone + 2 seconds silence + 2 seconds tone
    t2 = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 150.0 * t2)
    silence = np.zeros(2 * sr, dtype=np.float32)

    signal = np.concatenate([tone, silence, tone])
    time_series, pauses = extract_acoustic_contours(signal, sr)

    assert len(pauses) == 1
    pause = pauses[0]
    assert pause.start_s == pytest.approx(2.0, abs=0.15)
    assert pause.end_s == pytest.approx(4.0, abs=0.15)
    assert pause.duration_s == pytest.approx(2.0, abs=0.20)
    assert pause.is_excessive is True


def test_spectral_centroid_scaling():
    """Verifies that higher frequency signals yield strictly higher spectral centroid."""
    sr = TARGET_SAMPLE_RATE
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)

    low_tone = 0.6 * np.sin(2 * np.pi * 200 * t)
    high_tone = 0.6 * np.sin(2 * np.pi * 2000 * t)

    ts_low, _ = extract_acoustic_contours(low_tone, sr)
    ts_high, _ = extract_acoustic_contours(high_tone, sr)

    mean_cent_low = np.mean(ts_low.spectral_centroid_hz)
    mean_cent_high = np.mean(ts_high.spectral_centroid_hz)

    assert mean_cent_high > mean_cent_low * 2.5
