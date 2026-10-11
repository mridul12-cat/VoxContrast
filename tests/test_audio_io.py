"""
VoxContrast - Unit Tests for Audio Ingestion, Decoding, and Validation
"""

from __future__ import annotations

import io
import pytest
import numpy as np
import soundfile as sf

from core.audio_io import (
    process_audio,
    resample_audio,
    convert_to_mono,
    AudioTooQuietError,
    DurationExceededError,
    AudioFormatError,
    TARGET_SAMPLE_RATE
)
from tests.conftest import wav_to_bytes


def test_clean_audio_processing(synthetic_clean_speech):
    """Verifies that clean audio is ingested, normalized, and outputs 16kHz float32."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    proc_audio, meta = process_audio(audio_bytes)
    assert isinstance(proc_audio, np.ndarray)
    assert proc_audio.dtype == np.float32
    assert meta.sample_rate == TARGET_SAMPLE_RATE
    assert meta.duration_s == pytest.approx(4.0, abs=0.05)
    assert meta.channels == 1
    assert meta.snr_db > 10.0
    assert np.max(np.abs(proc_audio)) == pytest.approx(0.95, abs=0.02)


def test_stereo_to_mono_conversion():
    """Verifies that multi-channel stereo audio is correctly downmixed to mono."""
    t = np.linspace(0, 2.0, int(2.0 * TARGET_SAMPLE_RATE), endpoint=False, dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 200 * t)
    right = 0.3 * np.sin(2 * np.pi * 400 * t)
    stereo = np.column_stack([left, right])

    mono = convert_to_mono(stereo)
    assert mono.ndim == 1
    assert len(mono) == len(left)
    assert mono[100] == pytest.approx((left[100] + right[100]) / 2.0, abs=1e-5)


def test_resampling_from_44k():
    """Verifies that 44.1kHz audio is accurately resampled to 16kHz."""
    sr_orig = 44100
    t = np.linspace(0, 1.0, sr_orig, endpoint=False, dtype=np.float32)
    sig_44k = 0.5 * np.sin(2 * np.pi * 300 * t)

    resampled = resample_audio(sig_44k, sr_orig, TARGET_SAMPLE_RATE)
    assert len(resampled) == TARGET_SAMPLE_RATE
    assert resampled.dtype == np.float32


def test_silence_rejection_error(synthetic_silence):
    """Verifies that pure silence raises AudioTooQuietError."""
    sig, sr = synthetic_silence
    audio_bytes = wav_to_bytes(sig, sr)

    with pytest.raises(AudioTooQuietError):
        process_audio(audio_bytes)


def test_duration_exceeded_error():
    """Verifies that audio longer than 300 seconds is rejected."""
    long_sig = np.zeros(int(301.0 * TARGET_SAMPLE_RATE), dtype=np.float32)
    # Put a blip to not be purely silent
    long_sig[1000:2000] = 0.5

    with pytest.raises(DurationExceededError):
        process_audio(long_sig, TARGET_SAMPLE_RATE)


def test_clipping_detection(synthetic_clipped_wave):
    """Verifies that severe clipping is properly measured in metadata."""
    sig, sr = synthetic_clipped_wave
    audio_bytes = wav_to_bytes(sig, sr)

    proc_audio, meta = process_audio(audio_bytes)
    assert meta.clipped_ratio > 0.50


def test_corrupt_audio_rejection():
    """Verifies that invalid or random bytes raise AudioFormatError."""
    corrupt_bytes = b"NOT_A_VALID_AUDIO_HEADER_XYZ1234567890"

    with pytest.raises(AudioFormatError):
        process_audio(corrupt_bytes)
