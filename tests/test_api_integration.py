"""
VoxContrast - Integration Tests for Backend Endpoints
"""

from __future__ import annotations

import base64
import pytest
from fastapi.testclient import TestClient

from tests.conftest import wav_to_bytes


def test_health_endpoint(api_client: TestClient):
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_dataset_manifest_endpoint(api_client: TestClient):
    resp = api_client.get("/api/dataset/manifest")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["speech_texts"]) == 6
    assert len(data["pairs"]) == 18
    assert data["metadata"]["total_annotated_flaw_regions"] == 54


def test_audio_streaming_endpoint(api_client: TestClient):
    # Fetch first baseline audio file
    resp = api_client.get("/api/dataset/audio/baselines/speech_01_gettysburg_baseline.wav")
    assert resp.status_code == 200
    assert resp.headers["content-type"] in ("audio/wav", "audio/x-wav")
    assert len(resp.content) > 1000


def test_analyze_multipart_upload(api_client: TestClient, synthetic_clean_speech):
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    files = {"audio_file": ("test.wav", audio_bytes, "audio/wav")}
    data = {"transcript": "Four score and seven years ago"}

    resp = api_client.post("/api/analyze", files=files, data=data)
    assert resp.status_code == 200
    result = resp.json()

    assert "analysis_id" in result
    assert "audio_metadata" in result
    assert "rubric_scores" in result
    assert "time_series" in result
    assert 0.0 <= result["rubric_scores"]["overall_score"] <= 100.0


def test_analyze_json_base64(api_client: TestClient, synthetic_clean_speech):
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)
    b64_str = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "audio_base64": b64_str,
        "transcript": "Four score and seven years ago"
    }

    resp = api_client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert result["audio_metadata"]["duration_s"] == pytest.approx(4.0, abs=0.1)


def test_analyze_unsupported_extension(api_client: TestClient):
    files = {"audio_file": ("malicious.exe", b"MZ...", "application/octet-stream")}
    resp = api_client.post("/api/analyze", files=files)
    assert resp.status_code == 400


def test_compare_by_pair_id(api_client: TestClient):
    resp = api_client.post("/api/compare", data={"pair_id": "pair_speech_01_gettysburg_excessive_speed_1"})
    assert resp.status_code == 200
    result = resp.json()
    assert "baseline_analysis" in result
    assert "flawed_analysis" in result
    assert "score_delta" in result
    assert len(result["flaw_contrast_summary"]) >= 4
