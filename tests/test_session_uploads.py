"""
VoxContrast - Session-Scoped Audio Uploads & Isolation Tests
Validates all 12 required behaviors:
1. Uploading a valid file adds it to Session Uploads.
2. Multiple uploads are listed independently.
3. Invalid file types and oversized files are rejected.
4. An uploaded file can be selected for playback and analysis when supported.
5. Removing one upload leaves other uploads intact.
6. Clearing session uploads removes all current entries.
7. A fresh session does not display uploads from a previous session.
8. Session uploads do not modify the authoritative manifest, audio directories, or annotations.
9. Session uploads are excluded from official benchmark calculations by default.
10. Failed uploads and analysis failures display recoverable errors.
11. Navigation and existing compare/analysis interactions still work.
12. Temporary resources and server-side files follow the implemented cleanup policy.
"""

from __future__ import annotations

import os
import json
import pytest
from fastapi.testclient import TestClient

from backend.config import MANIFEST_PATH, DATA_DIR, MAX_UPLOAD_BYTES
from backend.session import session_manager, SESSION_TEMP_BASE
from tests.conftest import wav_to_bytes


def test_01_upload_valid_file_adds_to_session_uploads(api_client: TestClient, synthetic_clean_speech):
    """1. Uploading a valid file adds it to Session Uploads with metadata."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    init_res = api_client.post("/api/session/init")
    assert init_res.status_code == 200
    session_id = init_res.json()["session_id"]

    files = {"audio_file": ("speaker_trial_1.wav", audio_bytes, "audio/wav")}
    data = {"transcript": "Sample test utterance"}
    headers = {"X-Session-ID": session_id}

    resp = api_client.post("/api/session/upload", files=files, data=data, headers=headers)
    assert resp.status_code == 200
    res_data = resp.json()
    assert "upload" in res_data
    upload = res_data["upload"]
    assert upload["filename"] == "speaker_trial_1.wav"
    assert upload["duration_s"] > 0
    assert upload["sample_rate"] == 16000
    assert upload["upload_status"] == "uploaded"
    assert upload["analysis_status"] == "pending"

    # Verify listing includes the upload
    list_resp = api_client.get("/api/session/uploads", headers=headers)
    assert list_resp.status_code == 200
    uploads = list_resp.json()["uploads"]
    assert len(uploads) == 1
    assert uploads[0]["upload_id"] == upload["upload_id"]


def test_02_multiple_uploads_listed_independently(api_client: TestClient, synthetic_clean_speech):
    """2. Multiple uploads in the same session are listed independently without collisions."""
    sig, sr = synthetic_clean_speech
    audio_bytes1 = wav_to_bytes(sig[:16000 * 2], sr)  # 2s
    audio_bytes2 = wav_to_bytes(sig[:16000 * 3], sr)  # 3s

    init_res = api_client.post("/api/session/init")
    session_id = init_res.json()["session_id"]
    headers = {"X-Session-ID": session_id}

    # Upload 1
    r1 = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("first_speech.wav", audio_bytes1, "audio/wav")},
        headers=headers
    )
    assert r1.status_code == 200

    # Upload 2
    r2 = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("second_speech.wav", audio_bytes2, "audio/wav")},
        headers=headers
    )
    assert r2.status_code == 200

    list_resp = api_client.get("/api/session/uploads", headers=headers)
    uploads = list_resp.json()["uploads"]
    assert len(uploads) == 2
    filenames = [u["filename"] for u in uploads]
    assert "first_speech.wav" in filenames
    assert "second_speech.wav" in filenames


def test_03_invalid_file_types_and_oversized_files_rejected(api_client: TestClient, synthetic_clean_speech):
    """3. Invalid file types and oversized files are rejected with appropriate error codes."""
    init_res = api_client.post("/api/session/init")
    session_id = init_res.json()["session_id"]
    headers = {"X-Session-ID": session_id}

    # Invalid extension (.exe)
    resp_ext = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("malicious.exe", b"MZ...", "application/octet-stream")},
        headers=headers
    )
    assert resp_ext.status_code == 400
    assert "Unsupported audio format" in resp_ext.json()["detail"]

    # Oversized file
    oversized_bytes = b"0" * (MAX_UPLOAD_BYTES + 1024)
    resp_size = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("large_audio.wav", oversized_bytes, "audio/wav")},
        headers=headers
    )
    assert resp_size.status_code == 413


def test_04_uploaded_file_playback_and_analysis(api_client: TestClient, synthetic_clean_speech):
    """4. An uploaded file can be selected for playback streaming and analysis."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    init_res = api_client.post("/api/session/init")
    session_id = init_res.json()["session_id"]
    headers = {"X-Session-ID": session_id}

    # Upload file
    upload_res = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("speech_sample.wav", audio_bytes, "audio/wav")},
        data={"transcript": "Test speech for evaluation"},
        headers=headers
    )
    upload_id = upload_res.json()["upload"]["upload_id"]

    # Stream audio playback
    stream_res = api_client.get(f"/api/session/audio/{upload_id}", headers=headers)
    assert stream_res.status_code == 200
    assert len(stream_res.content) == len(audio_bytes)
    assert stream_res.headers["content-type"] in ("audio/wav", "audio/x-wav")

    # Run analysis without reference baseline
    ana_res = api_client.post(
        f"/api/session/analyze/{upload_id}",
        data={"transcript": "Test speech for evaluation"},
        headers=headers
    )
    assert ana_res.status_code == 200
    ana_data = ana_res.json()
    assert "analysis" in ana_data
    assert ana_data["analysis"]["audio_metadata"]["duration_s"] > 0
    assert 0.0 <= ana_data["analysis"]["rubric_scores"]["overall_score"] <= 100.0

    # Run analysis with authoritative reference baseline (paired comparison)
    cmp_res = api_client.post(
        f"/api/session/analyze/{upload_id}",
        data={"reference_baseline_id": "speech_01_gettysburg_baseline.wav"},
        headers=headers
    )
    assert cmp_res.status_code == 200
    cmp_data = cmp_res.json()
    assert cmp_data["comparison"] is not None
    assert "baseline_analysis" in cmp_data["comparison"]
    assert "score_delta" in cmp_data["comparison"]


def test_05_removing_one_upload_leaves_others_intact(api_client: TestClient, synthetic_clean_speech):
    """5. Removing one upload leaves other session uploads intact."""
    sig, sr = synthetic_clean_speech
    audio_bytes1 = wav_to_bytes(sig[:16000 * 2], sr)
    audio_bytes2 = wav_to_bytes(sig[:16000 * 3], sr)

    init_res = api_client.post("/api/session/init")
    session_id = init_res.json()["session_id"]
    headers = {"X-Session-ID": session_id}

    r1 = api_client.post("/api/session/upload", files={"audio_file": ("item1.wav", audio_bytes1, "audio/wav")}, headers=headers)
    id1 = r1.json()["upload"]["upload_id"]

    r2 = api_client.post("/api/session/upload", files={"audio_file": ("item2.wav", audio_bytes2, "audio/wav")}, headers=headers)
    id2 = r2.json()["upload"]["upload_id"]

    # Delete upload 1
    del_res = api_client.delete(f"/api/session/upload/{id1}", headers=headers)
    assert del_res.status_code == 200

    # Verify upload 2 remains
    list_res = api_client.get("/api/session/uploads", headers=headers)
    uploads = list_res.json()["uploads"]
    assert len(uploads) == 1
    assert uploads[0]["upload_id"] == id2


def test_06_clearing_session_uploads_removes_all(api_client: TestClient, synthetic_clean_speech):
    """6. Clearing session uploads removes all current entries."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    init_res = api_client.post("/api/session/init")
    session_id = init_res.json()["session_id"]
    headers = {"X-Session-ID": session_id}

    api_client.post("/api/session/upload", files={"audio_file": ("clear1.wav", audio_bytes, "audio/wav")}, headers=headers)

    clear_res = api_client.post("/api/session/clear", headers=headers)
    assert clear_res.status_code == 200

    list_res = api_client.get("/api/session/uploads", headers=headers)
    assert len(list_res.json()["uploads"]) == 0


def test_07_fresh_session_does_not_display_previous_uploads(api_client: TestClient, synthetic_clean_speech):
    """7. A fresh session does not display uploads from a previous session."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    # Session A
    sA = api_client.post("/api/session/init").json()["session_id"]
    api_client.post(
        "/api/session/upload",
        files={"audio_file": ("secret_session_a.wav", audio_bytes, "audio/wav")},
        headers={"X-Session-ID": sA}
    )

    # Session B (new fresh session)
    sB = api_client.post("/api/session/init").json()["session_id"]
    list_b = api_client.get("/api/session/uploads", headers={"X-Session-ID": sB})
    assert len(list_b.json()["uploads"]) == 0


def test_08_session_uploads_do_not_modify_authoritative_manifest_or_audio(api_client: TestClient, synthetic_clean_speech):
    """8. Session uploads do not modify the authoritative manifest, audio directories, or annotations."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        before_manifest_raw = f.read()

    baselines_before = set(os.listdir(os.path.join(DATA_DIR, "audio", "baselines")))
    flawed_before = set(os.listdir(os.path.join(DATA_DIR, "audio", "flawed")))

    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    init_res = api_client.post("/api/session/init")
    session_id = init_res.json()["session_id"]
    api_client.post(
        "/api/session/upload",
        files={"audio_file": ("exploratory_test.wav", audio_bytes, "audio/wav")},
        headers={"X-Session-ID": session_id}
    )

    # Check manifest file unchanged
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        after_manifest_raw = f.read()
    assert before_manifest_raw == after_manifest_raw

    # Check baseline & flawed audio directories unchanged
    baselines_after = set(os.listdir(os.path.join(DATA_DIR, "audio", "baselines")))
    flawed_after = set(os.listdir(os.path.join(DATA_DIR, "audio", "flawed")))
    assert baselines_before == baselines_after
    assert flawed_before == flawed_after


def test_09_session_uploads_excluded_from_official_benchmarks(api_client: TestClient, synthetic_clean_speech):
    """9. Session uploads are excluded from official benchmark calculations by default."""
    before_bm = api_client.get("/api/benchmarks").json()

    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    init_res = api_client.post("/api/session/init")
    session_id = init_res.json()["session_id"]
    api_client.post(
        "/api/session/upload",
        files={"audio_file": ("custom_flawed.wav", audio_bytes, "audio/wav")},
        headers={"X-Session-ID": session_id}
    )

    # Re-evaluate benchmarks
    after_bm = api_client.get("/api/benchmarks").json()
    assert before_bm["macro_iou"] == after_bm["macro_iou"]
    assert before_bm["total_ground_truth_flaws"] == after_bm["total_ground_truth_flaws"]


def test_10_failed_uploads_and_analysis_display_recoverable_errors(api_client: TestClient):
    """10. Failed uploads and analysis failures display recoverable errors."""
    init_res = api_client.post("/api/session/init")
    session_id = init_res.json()["session_id"]
    headers = {"X-Session-ID": session_id}

    # Upload corrupt audio
    resp = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("corrupt.wav", b"RIFF....corruptdata", "audio/wav")},
        headers=headers
    )
    assert resp.status_code == 400
    err_json = resp.json()
    assert "detail" in err_json

    # Request analysis on non-existent upload
    resp_bad_ana = api_client.post(
        "/api/session/analyze/non_existent_upload",
        headers=headers
    )
    assert resp_bad_ana.status_code == 404
    assert "not found" in resp_bad_ana.json()["detail"].lower()


def test_11_navigation_and_existing_compare_still_work(api_client: TestClient):
    """11. Navigation endpoints and existing compare/analysis interactions still work intact."""
    # Manifest endpoint
    manifest_res = api_client.get("/api/dataset/manifest")
    assert manifest_res.status_code == 200
    assert len(manifest_res.json()["pairs"]) == 18

    # Compare endpoint with authoritative pair_id
    cmp_res = api_client.post("/api/compare", data={"pair_id": "pair_speech_01_gettysburg_excessive_speed_1"})
    assert cmp_res.status_code == 200
    assert "baseline_analysis" in cmp_res.json()
    assert cmp_res.json()["baseline_analysis"]["rubric_scores"]["overall_score"] == 95.0


def test_12_temporary_resources_follow_cleanup_policy(api_client: TestClient, synthetic_clean_speech):
    """12. Temporary resources and server-side files follow the implemented cleanup policy."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    init_res = api_client.post("/api/session/init")
    session_id = init_res.json()["session_id"]
    headers = {"X-Session-ID": session_id}

    upload_res = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("cleanup_test.wav", audio_bytes, "audio/wav")},
        headers=headers
    )
    upload_id = upload_res.json()["upload"]["upload_id"]

    session_obj = session_manager.get_session(session_id)
    upload_item = session_obj.uploads[upload_id]
    assert os.path.exists(upload_item.file_path)

    # Delete individual upload -> file deleted from disk
    api_client.delete(f"/api/session/upload/{upload_id}", headers=headers)
    assert not os.path.exists(upload_item.file_path)

    # Re-upload and clear entire session -> directory removed from disk
    upload_res2 = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("cleanup_test_2.wav", audio_bytes, "audio/wav")},
        headers=headers
    )
    assert os.path.exists(session_obj.dir_path)

    api_client.post("/api/session/clear", headers=headers)
    assert not os.path.exists(session_obj.dir_path)


def test_13_unknown_session_id_auto_creates_valid_accessible_session(api_client: TestClient, synthetic_clean_speech):
    """13. Unknown/expired session ID in upload creates valid session and lists upload properly."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    resp = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("test_unknown_sid.wav", audio_bytes, "audio/wav")},
        headers={"X-Session-ID": "expired-unknown-sid-999"}
    )
    assert resp.status_code == 200
    returned_sid = resp.json()["session_id"]
    assert returned_sid != "expired-unknown-sid-999"

    # Verify upload is fully retrievable using the returned session ID
    list_resp = api_client.get("/api/session/uploads", headers={"X-Session-ID": returned_sid})
    assert list_resp.status_code == 200
    uploads = list_resp.json()["uploads"]
    assert len(uploads) == 1
    assert uploads[0]["filename"] == "test_unknown_sid.wav"


def test_14_session_id_path_traversal_sanitized_and_contained(api_client: TestClient, synthetic_clean_speech):
    """14. Session ID path traversal patterns are rejected/sanitized to prevent directory escape."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    resp = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("traversal_test.wav", audio_bytes, "audio/wav")},
        headers={"X-Session-ID": "../../../../tmp"}
    )
    assert resp.status_code == 200
    returned_sid = resp.json()["session_id"]
    assert ".." not in returned_sid and "/" not in returned_sid

    session_obj = session_manager.get_session(returned_sid)
    assert session_obj is not None
    assert os.path.realpath(session_obj.dir_path).startswith(os.path.realpath(SESSION_TEMP_BASE))


def test_15_reference_baseline_traversal_prevention_and_error(api_client: TestClient, synthetic_clean_speech):
    """15. Invalid reference baseline or traversal attempt is rejected with 400 error."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    init_res = api_client.post("/api/session/init")
    sid = init_res.json()["session_id"]
    headers = {"X-Session-ID": sid}

    up_res = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("speech_base_test.wav", audio_bytes, "audio/wav")},
        headers=headers
    )
    upload_id = up_res.json()["upload"]["upload_id"]

    # Attempt traversal
    ana_res = api_client.post(
        f"/api/session/analyze/{upload_id}",
        data={"reference_baseline_id": "../../../../etc/passwd"},
        headers=headers
    )
    assert ana_res.status_code == 400
    assert "not found" in ana_res.json()["detail"].lower()


def test_16_canonical_transcript_auto_applied_when_pairing_baseline(api_client: TestClient, synthetic_clean_speech):
    """16. Pairing with authoritative baseline auto-resolves canonical transcript when empty."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    init_res = api_client.post("/api/session/init")
    sid = init_res.json()["session_id"]
    headers = {"X-Session-ID": sid}

    up_res = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("speech_gettysburg_user.wav", audio_bytes, "audio/wav")},
        headers=headers
    )
    upload_id = up_res.json()["upload"]["upload_id"]

    cmp_res = api_client.post(
        f"/api/session/analyze/{upload_id}",
        data={"reference_baseline_id": "speech_01_gettysburg_baseline.wav"},
        headers=headers
    )
    assert cmp_res.status_code == 200
    comp_data = cmp_res.json()["comparison"]
    assert comp_data is not None
    assert comp_data["baseline_analysis"]["rubric_scores"]["overall_score"] > 80.0


def test_17_multiformat_upload_and_mime_type_streaming(api_client: TestClient, synthetic_clean_speech):
    """17. Multiformat audio upload streams with accurate MIME type."""
    sig, sr = synthetic_clean_speech
    audio_bytes = wav_to_bytes(sig, sr)

    init_res = api_client.post("/api/session/init")
    sid = init_res.json()["session_id"]
    headers = {"X-Session-ID": sid}

    # Upload named .mp3
    up_res = api_client.post(
        "/api/session/upload",
        files={"audio_file": ("speaker_utterance.mp3", audio_bytes, "audio/mpeg")},
        headers=headers
    )
    assert up_res.status_code == 200
    up_id = up_res.json()["upload"]["upload_id"]

    stream_res = api_client.get(f"/api/session/audio/{up_id}", headers=headers)
    assert stream_res.status_code == 200
    assert stream_res.headers["content-type"] == "audio/mpeg"
