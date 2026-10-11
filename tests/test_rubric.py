"""
VoxContrast - Unit Tests for Deterministic Rubric Scoring & Speaker Normalization
"""

from __future__ import annotations

import pytest
from core.rubric import calculate_rubric_scores
from core.schemas import FeaturesSummary, AudioMetadata, DetectedFlaw
from core.normalizer import SpeakerNormalizer


@pytest.fixture
def sample_features():
    return FeaturesSummary(
        average_wpm=145.0,
        pitch_mean_hz=145.0,
        pitch_std_hz=25.0,
        pitch_median_hz=142.0,
        pitch_semitone_std=3.4,
        energy_mean_db=-21.0,
        energy_std_db=5.8,
        pause_count=2,
        total_pause_duration_s=1.2,
        spectral_centroid_mean_hz=2100.0
    )


@pytest.fixture
def sample_metadata():
    return AudioMetadata(
        duration_s=8.5,
        sample_rate=16000,
        channels=1,
        snr_db=28.5,
        clipped_ratio=0.0
    )


def test_rubric_strict_reproducibility(sample_features, sample_metadata):
    """
    CRITICAL GATE: Verifies 0.0000 score delta across 100 identical executions.
    """
    first_result = calculate_rubric_scores(sample_features, [], sample_metadata)
    base_overall = first_result.overall_score

    for _ in range(100):
        run_result = calculate_rubric_scores(sample_features, [], sample_metadata)
        assert run_result.overall_score == base_overall
        for k in first_result.dimensions:
            assert run_result.dimensions[k].score == first_result.dimensions[k].score
            assert run_result.dimensions[k].raw_deduction == first_result.dimensions[k].raw_deduction


def test_speaker_normalization_invariance(sample_metadata):
    """
    Verifies that male (120 Hz) vs female (220 Hz) base pitches receive identical
    scores when relative semitone modulation is the same.
    """
    male_features = FeaturesSummary(
        average_wpm=140.0,
        pitch_mean_hz=122.0,
        pitch_std_hz=20.0,
        pitch_median_hz=120.0,
        pitch_semitone_std=3.2,
        energy_mean_db=-20.0,
        energy_std_db=5.5,
        pause_count=1,
        total_pause_duration_s=0.6,
        spectral_centroid_mean_hz=1900.0
    )

    female_features = FeaturesSummary(
        average_wpm=140.0,
        pitch_mean_hz=224.0,
        pitch_std_hz=36.0,
        pitch_median_hz=220.0,
        pitch_semitone_std=3.2,  # Same semitone dynamic modulation
        energy_mean_db=-20.0,
        energy_std_db=5.5,
        pause_count=1,
        total_pause_duration_s=0.6,
        spectral_centroid_mean_hz=1900.0
    )

    male_scores = calculate_rubric_scores(male_features, [], sample_metadata)
    female_scores = calculate_rubric_scores(female_features, [], sample_metadata)

    assert male_scores.dimensions["pitch_modulation"].score == female_scores.dimensions["pitch_modulation"].score
    assert male_scores.overall_score == female_scores.overall_score


def test_unvoiced_whisper_rescaling(sample_metadata):
    """
    Verifies that unvoiced audio triggers pitch rescaling without crashing or breaking score normalization.
    """
    whisper_features = FeaturesSummary(
        average_wpm=140.0,
        pitch_mean_hz=0.0,
        pitch_std_hz=0.0,
        pitch_median_hz=0.0,
        pitch_semitone_std=0.0,
        energy_mean_db=-26.0,
        energy_std_db=5.0,
        pause_count=1,
        total_pause_duration_s=0.5,
        spectral_centroid_mean_hz=1600.0
    )

    scores = calculate_rubric_scores(whisper_features, [], sample_metadata)
    assert 0.0 <= scores.overall_score <= 100.0
    assert scores.dimensions["pitch_modulation"].weight == 0.0
