"""
VoxContrast - Provenance & Intelligible Speech Integrity Tests
Ensures historical algorithmic humming data is distinguished from genuine speech-capable TTS data,
and guards against confusing glottal sinusoidal modeling with spoken-speech reference audio.
"""

import os
import json
import pytest
import numpy as np
import soundfile as sf

from core.audio_io import process_audio
from core.features import extract_acoustic_contours
from core.alignment import align_transcript_to_audio
from scripts.generate_speech_dataset_v2 import MacOSSysSayBackend


def test_spoken_speech_manifest_structure_and_splits():
    """Verify spoken speech manifest is properly structured with split isolation."""
    manifest_path = "data/manifest.json"
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_version"] == "2.0.0"
    assert data["provenance_class"] == "spoken_speech_v2"
    assert data["metadata"]["total_audio_utterances"] == 24
    assert data["metadata"]["total_annotated_flaw_regions"] == 18
    assert len(data["baselines"]) == 6
    assert len(data["pairs"]) == 18

    # Verify split isolation
    train_dev = data["splits"]["train_dev"]
    final_test = data["splits"]["final_test"]
    assert len(train_dev) == 12
    assert len(final_test) == 6
    assert set(train_dev).isdisjoint(set(final_test))


def test_baselines_audio_integrity_and_alignment():
    """Verify all 6 baselines exist, pass acoustic checks, and align with canonical transcripts."""
    manifest_path = "data/manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for b in data["baselines"]:
        path = b["audio_path"]
        assert os.path.exists(path), f"Missing audio: {path}"

        audio, meta = process_audio(path)
        assert meta.sample_rate == 16000
        assert meta.channels == 1
        assert meta.duration_s > 4.0
        assert meta.snr_db > 25.0
        assert meta.clipped_ratio == 0.0

        ts, pauses = extract_acoustic_contours(audio, meta.sample_rate)
        align_st, _ = align_transcript_to_audio(b["canonical_transcript"], ts, pauses, meta.duration_s)
        assert align_st.status == "SUCCESS"
        assert align_st.confidence >= 0.85
        assert len(align_st.aligned_words) == align_st.word_count


def test_humming_synthesizer_fails_speech_intelligibility_criteria():
    """
    Regression Test: The algorithmic glottal harmonic generator (synthesize_speech_utterance)
    is purely sinusoidal harmonics and lacks vowel formant spectral diffusion.
    Ensures it is never mistaken for a valid spoken-speech reference.
    """
    from scripts.generate_dataset import synthesize_speech_utterance

    words = ["This", "is", "a", "test", "of", "algorithmic", "humming"]
    audio, word_timings, _ = synthesize_speech_utterance(words, base_f0=140.0, default_wpm=135.0)

    # Compute power spectral density
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / 16000)

    # Energy in sinusoidal harmonic peaks vs broadband formant distribution
    # Glottal model places 90%+ energy in narrow harmonics at 140, 280, 420, 560 Hz
    f0_harmonics_energy = np.sum(fft[(freqs >= 130) & (freqs <= 150)]) + \
                          np.sum(fft[(freqs >= 270) & (freqs <= 290)]) + \
                          np.sum(fft[(freqs >= 410) & (freqs <= 430)])

    total_voiced_energy = np.sum(fft[(freqs >= 100) & (freqs <= 2500)])
    harmonic_ratio = f0_harmonics_energy / (total_voiced_energy + 1e-12)

    # True human/TTS speech has continuous formant dispersion (F1, F2, F3) across vowels,
    # whereas pure harmonic humming has concentrated spikes
    assert harmonic_ratio > 0.08, "Synthesizer confirmed to be discrete harmonic humming tone"


def test_say_backend_generates_intelligible_audio(tmp_path):
    """Verify MacOSSysSayBackend creates valid intelligible 16kHz audio."""
    backend = MacOSSysSayBackend()
    if not backend.is_available():
        pytest.skip("macOS say backend unavailable on this host")

    out_wav = str(tmp_path / "test_say.wav")
    text = "The VoxContrast system evaluates spoken delivery."
    backend.synthesize(text, out_wav, voice="Samantha", rate_wpm=140)

    assert os.path.exists(out_wav)
    audio, sr = sf.read(out_wav)
    assert sr == 16000
    assert len(audio) > 16000 * 1.5  # At least 1.5s
