"""
VoxContrast - Pytest Test Fixtures and Synthetic Generators
"""

from __future__ import annotations

import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app

TARGET_SR = 16000


@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def synthetic_clean_speech():
    """Generates 4-second synthetic vowel-like tone at 140Hz."""
    t = np.linspace(0, 4.0, int(4.0 * TARGET_SR), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 140 * t) + 0.25 * np.sin(2 * np.pi * 280 * t)
    sig = sig * (0.8 + 0.2 * np.sin(2 * np.pi * 1.5 * t))  # amplitude modulation
    return sig.astype(np.float32), TARGET_SR


@pytest.fixture
def synthetic_silence():
    """Generates 3 seconds of silence (< -60 dBFS)."""
    return np.zeros(int(3.0 * TARGET_SR), dtype=np.float32), TARGET_SR


@pytest.fixture
def synthetic_clipped_wave():
    """Generates clipped square-like wave."""
    t = np.linspace(0, 3.0, int(3.0 * TARGET_SR), endpoint=False, dtype=np.float32)
    sig = np.sign(np.sin(2 * np.pi * 200 * t)) * 0.999
    return sig.astype(np.float32), TARGET_SR


def wav_to_bytes(audio: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Helper to convert float32 numpy array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()
