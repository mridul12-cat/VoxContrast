"""
VoxContrast - Unit Tests for Forced Alignment and Segmentation
"""

from __future__ import annotations

import pytest
import numpy as np

from core.alignment import align_transcript_to_audio, clean_transcript
from core.features import extract_acoustic_contours
from core.audio_io import TARGET_SAMPLE_RATE


def test_clean_transcript_tokenization():
    """Verifies that punctuation is stripped and valid tokens are extracted."""
    raw = "Four score, and seven years ago—our fathers brought forth!"
    tokens = clean_transcript(raw)
    assert tokens == ["Four", "score", "and", "seven", "years", "ago", "our", "fathers", "brought", "forth"]


def test_successful_alignment():
    """Verifies that matching speech audio aligns cleanly with SUCCESS status."""
    sr = TARGET_SAMPLE_RATE
    # 5 seconds of active speech signal
    t = np.linspace(0, 5.0, 5 * sr, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 150 * t)

    ts, pauses = extract_acoustic_contours(sig, sr)
    transcript = "Four score and seven years ago our fathers brought forth"

    alignment, updated_ts = align_transcript_to_audio(
        transcript=transcript,
        time_series=ts,
        pauses=pauses,
        audio_duration_s=5.0
    )

    assert alignment.status == "SUCCESS"
    assert alignment.confidence >= 0.70
    assert len(alignment.aligned_words) == 10

    # Ensure strictly monotonic progression
    for i in range(len(alignment.aligned_words) - 1):
        w_cur = alignment.aligned_words[i]
        w_next = alignment.aligned_words[i + 1]
        assert w_cur.start_s < w_cur.end_s
        assert w_cur.start_s <= w_next.start_s


def test_degraded_alignment_on_word_density_mismatch():
    """Verifies that extreme transcript mismatch triggers DEGRADED status."""
    sr = TARGET_SAMPLE_RATE
    # 2 seconds audio
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 150 * t)

    ts, pauses = extract_acoustic_contours(sig, sr)
    # 100 words in 2 seconds is impossible (> 3000 WPM)
    giant_text = "word " * 100

    alignment, _ = align_transcript_to_audio(
        transcript=giant_text,
        time_series=ts,
        pauses=pauses,
        audio_duration_s=2.0
    )

    assert alignment.status == "DEGRADED"
    assert alignment.failure_reason is not None


def test_failed_alignment_on_empty_transcript():
    """Verifies that empty transcript returns FAILED status."""
    sr = TARGET_SAMPLE_RATE
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 150 * t)

    ts, pauses = extract_acoustic_contours(sig, sr)

    alignment, _ = align_transcript_to_audio(
        transcript="   \n\t  ",
        time_series=ts,
        pauses=pauses,
        audio_duration_s=2.0
    )

    assert alignment.status == "FAILED"
    assert alignment.word_count == 0
    assert len(alignment.aligned_words) == 0
