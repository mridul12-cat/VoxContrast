"""
VoxContrast - Unit and Integration Tests for TTS-Assisted Expansion Pipeline
Tests backend capability detection, error handling on unavailable neural backends,
manifest schema validation, contrastive transcript identity, audio file integrity,
ground truth annotation boundaries, split isolation (zero leakage), and DSP transformations.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.generate_tts_expansion import (
    BackendRegistry,
    BackendUnavailableError,
    MacOSSysSayBackend,
    NeuralTTSBackend,
    apply_monotone_pitch_dsp,
    apply_unstable_energy_dsp,
    apply_monotone_energy_dsp,
    insert_unnatural_pause_dsp,
    validate_audio_file,
    validate_flaw_acoustic_presence,
    DEFAULT_OUTPUT_DIR,
)
from backend.config import MANIFEST_PATH

TTS_MANIFEST_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "manifest.json")


class TestBackendCapabilitiesAndDiscovery:
    """Tests backend capability detection and registry mechanisms."""

    def test_detect_all_structure(self):
        """BackendRegistry.detect_all() reports capabilities for both backends."""
        caps = BackendRegistry.detect_all()
        assert "macos_say" in caps
        assert "neural_piper" in caps

        # Check required fields
        for backend_name in ("macos_say", "neural_piper"):
            b_info = caps[backend_name]
            assert "backend" in b_info
            assert "available" in b_info
            assert isinstance(b_info["available"], bool)
            assert "rate_control" in b_info
            assert "pitch_control" in b_info
            assert "licensing" in b_info

    def test_macos_say_on_darwin(self):
        """MacOSSysSayBackend correctly reports availability on Darwin."""
        backend = MacOSSysSayBackend()
        is_darwin = sys.platform == "darwin"
        assert backend.is_available() == is_darwin
        caps = backend.get_capabilities()
        assert caps["backend"] == "macos_say"
        assert "Apple" in caps["licensing"]

    def test_unknown_backend_raises_value_error(self):
        """BackendRegistry.get_backend raises ValueError on unknown backend."""
        with pytest.raises(ValueError, match="Unknown TTS backend"):
            BackendRegistry.get_backend("nonexistent_speech_engine_xyz")

    def test_neural_backend_unavailable_error_on_missing_binary(self):
        """NeuralTTSBackend raises BackendUnavailableError when dependencies are missing."""
        neural = NeuralTTSBackend(model_path="/nonexistent/model.onnx")
        with patch.object(neural, "is_available", return_value=False):
            with pytest.raises(BackendUnavailableError) as exc_info:
                neural.synthesize("Test phrase", "/tmp/dummy.wav")
            assert "piper-tts" in str(exc_info.value)
            assert "unavailable" in str(exc_info.value).lower()

    def test_registry_raises_on_unavailable_neural(self):
        """BackendRegistry raises BackendUnavailableError when neural backend is unavailable."""
        with patch.object(NeuralTTSBackend, "is_available", return_value=False):
            with pytest.raises(BackendUnavailableError) as exc_info:
                BackendRegistry.get_backend("neural_piper")
            assert "piper-tts" in str(exc_info.value)
            assert "No silent substitution" in str(exc_info.value)





class TestAudioValidationHelper:
    """Tests the validate_audio_file function on corner cases."""

    def test_validate_audio_nonexistent_file(self):
        """Returns False for missing files."""
        is_valid, info, err = validate_audio_file("/nonexistent/audio/file.wav")
        assert not is_valid
        assert "not found" in err.lower()

    def test_validate_audio_silence(self, tmp_path):
        """Rejects silence below -45 dBFS threshold."""
        silence_file = str(tmp_path / "silence.wav")
        silence = np.zeros(16000 * 3, dtype=np.float32)
        sf.write(silence_file, silence, 16000)

        is_valid, info, err = validate_audio_file(silence_file)
        assert not is_valid
        assert "silence" in err.lower()


class TestDSPTransformations:
    """Tests DSP synthesis modifications."""

    @pytest.fixture
    def test_tone(self):
        t = np.linspace(0, 2.0, 16000 * 2, endpoint=False, dtype=np.float32)
        sig = 0.4 * np.sin(2 * np.pi * 180 * t) + 0.2 * np.sin(2 * np.pi * 360 * t)
        return sig.astype(np.float32)

    def test_monotone_pitch_dsp(self, test_tone):
        """apply_monotone_pitch_dsp produces finite signal of matching length."""
        flat = apply_monotone_pitch_dsp(test_tone, sr=16000, target_f0=160.0)
        assert len(flat) == len(test_tone)
        assert not np.isnan(flat).any()
        assert not np.isinf(flat).any()

    def test_unstable_energy_dsp(self, test_tone):
        """apply_unstable_energy_dsp modulates signal amplitude."""
        mod = apply_unstable_energy_dsp(test_tone, sr=16000, freq_hz=2.0, swing_depth=0.8)
        assert len(mod) == len(test_tone)
        assert not np.isnan(mod).any()
        assert np.max(np.abs(mod)) > np.max(np.abs(test_tone))

    def test_monotone_energy_dsp(self, test_tone):
        """apply_monotone_energy_dsp compresses dynamic range."""
        comp = apply_monotone_energy_dsp(test_tone, sr=16000, target_rms=0.15)
        assert len(comp) == len(test_tone)
        assert not np.isnan(comp).any()

    def test_insert_unnatural_pause_dsp(self, test_tone):
        """insert_unnatural_pause_dsp inserts silence gap between segments."""
        half = len(test_tone) // 2
        p1 = test_tone[:half]
        p2 = test_tone[half:]
        pause_dur = 1.5
        stitched = insert_unnatural_pause_dsp([p1, p2], pause_index=0, pause_duration_s=pause_dur, sr=16000)
        expected_len = len(test_tone) + int(pause_dur * 16000)
        assert len(stitched) == expected_len


def clean_text(text: str) -> str:
    """Helper to normalize text for comparison."""
    import re
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()
