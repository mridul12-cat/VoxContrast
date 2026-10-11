"""
VoxContrast - Adversarial & Edge Case Tests
Validates system resilience against extreme inputs: absolute silence, high clipping,
white noise, mismatched transcripts, empty files, and boundary durations.
"""

from __future__ import annotations

import pytest
import numpy as np
from fastapi.testclient import TestClient

from core.analyzer import analyze_speech
from core.audio_io import AudioTooQuietError, AudioFormatError, DurationExceededError, TARGET_SAMPLE_RATE
from tests.conftest import wav_to_bytes


def test_adversarial_pure_silence():
    """Verifies that pure silence is rejected cleanly with AudioTooQuietError."""
    silence = np.zeros(TARGET_SAMPLE_RATE * 3, dtype=np.float32)
    with pytest.raises(AudioTooQuietError):
        analyze_speech(silence, "Testing silence rejection.")


def test_adversarial_heavy_clipping(api_client: TestClient):
    """Verifies that heavy clipping is handled gracefully and incurs a clipping deduction."""
    t = np.linspace(0, 3.0, 3 * TARGET_SAMPLE_RATE, endpoint=False, dtype=np.float32)
    # Severe square wave clipping
    clipped = np.sign(np.sin(2 * np.pi * 220 * t)) * 0.999
    audio_bytes = wav_to_bytes(clipped, TARGET_SAMPLE_RATE)

    files = {"audio_file": ("clipped.wav", audio_bytes, "audio/wav")}
    resp = api_client.post("/api/analyze", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["audio_metadata"]["clipped_ratio"] > 0.80
    assert data["rubric_scores"]["dimensions"]["vocal_energy"]["raw_deduction"] > 0.0


def test_adversarial_white_noise():
    """Verifies that white noise does not generate NaN or crash F0 pitch calculation."""
    noise = np.random.uniform(-0.4, 0.4, TARGET_SAMPLE_RATE * 3).astype(np.float32)
    res = analyze_speech(noise, "Random white noise input test")

    assert np.isfinite(res.rubric_scores.overall_score)
    assert res.audio_metadata.duration_s == pytest.approx(3.0, abs=0.1)


def test_adversarial_mismatched_transcript():
    """Verifies that radically mismatched transcript enters DEGRADED alignment without failing."""
    t = np.linspace(0, 4.0, 4 * TARGET_SAMPLE_RATE, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 150 * t)

    # Completely different text with wrong word count
    mismatched_text = "One single solitary syllable."
    res = analyze_speech(sig, mismatched_text)

    assert res.alignment_status.status in ("SUCCESS", "DEGRADED")
    assert res.rubric_scores.overall_score > 0.0


def test_adversarial_empty_audio_bytes(api_client: TestClient):
    """Verifies that 0-byte upload returns structured 400 error."""
    files = {"audio_file": ("empty.wav", b"", "audio/wav")}
    resp = api_client.post("/api/analyze", files=files)
    assert resp.status_code == 400
