"""
VoxContrast - Backend API Routes
Exposes typed REST endpoints for speech analysis, paired contrastive evaluation,
dataset manifest inspection, and session-isolated exploratory uploads.
"""

from __future__ import annotations

import os
import json
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.config import (
    BASE_DIR,
    DATA_DIR,
    MANIFEST_PATH,
    MAX_UPLOAD_BYTES,
    ALLOWED_AUDIO_EXTENSIONS
)
from backend.session import session_manager
from core.analyzer import analyze_speech, compare_speech_pair
from core.audio_io import (
    AudioTooQuietError,
    DurationExceededError,
    FileSizeExceededError,
    AudioFormatError
)
from core.schemas import AnalysisResponse, ComparisonResponse, DatasetManifest

router = APIRouter(prefix="/api")


import re

def _resolve_session_id(request: Request, explicit_id: Optional[str] = None, create_if_missing: bool = True) -> Optional[str]:
    """Extracts session ID from header, param, or instantiates a fresh session."""
    sid = explicit_id or request.headers.get("x-session-id") or request.query_params.get("session_id")
    if sid:
        sid = sid.strip()
        if not re.match(r"^[a-zA-Z0-9_-]{16,128}$", sid):
            sid = None
        elif not session_manager.get_session(sid):
            sid = None
    if not sid and create_if_missing:
        sid = session_manager.create_session()
    return sid


class AnalyzeJsonRequest(BaseModel):
    audio_base64: str
    transcript: str = ""
    baseline_audio_id: Optional[str] = None


@router.get("/health")
def health_check():
    """Service health and operational status."""
    return {
        "status": "healthy",
        "service": "VoxContrast Speech Analytics Engine",
        "version": "1.0.0"
    }


@router.get("/dataset/manifest")
def get_dataset_manifest(dataset: Optional[str] = None, version: Optional[str] = None) -> dict:
    """
    Returns the contrastive speech dataset manifest.
    """
    if not os.path.exists(MANIFEST_PATH):
        raise HTTPException(status_code=404, detail="Dataset manifest not found on disk.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


_cached_benchmarks: dict | None = None


@router.get("/benchmarks")
def get_benchmarks(refresh: bool = False) -> dict:
    """
    Returns authentic temporal flaw grounding benchmark evaluation metrics
    computed across the dataset using tests/evaluate_temporal_iou.py.
    """
    global _cached_benchmarks
    if _cached_benchmarks is None or refresh:
        if not os.path.exists(MANIFEST_PATH):
            raise HTTPException(status_code=404, detail="Dataset manifest not found.")
        from tests.evaluate_temporal_iou import run_temporal_grounding_benchmark
        _cached_benchmarks = run_temporal_grounding_benchmark(
            manifest_path=MANIFEST_PATH,
            split=None,
            confirmed_only=True
        )
    return _cached_benchmarks


@router.get("/dataset/{folder}/{filename}")
@router.get("/dataset/audio/{folder}/{filename}")
def get_audio_file(folder: str, filename: str):
    """
    Streams audio file from dataset directory (data/audio/).
    """
    if folder not in ("baselines", "flawed"):
        raise HTTPException(status_code=400, detail="Invalid audio folder.")

    audio_path = os.path.join(DATA_DIR, "audio", folder, filename)
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav")

    raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found.")


# ==============================================================================
# Session-Scoped Upload Endpoints (Isolated, Non-Persistent, Cleaned on demand)
# ==============================================================================

@router.post("/session/init")
def init_session(response: Response):
    """Generates an isolated, unpredictable session identifier."""
    sid = session_manager.create_session()
    response.headers["X-Session-ID"] = sid
    return {"session_id": sid, "status": "initialized"}


@router.get("/session/uploads")
def get_session_uploads(request: Request, session_id: Optional[str] = None):
    """Lists uploads scoped exclusively to the requester's active session."""
    sid = _resolve_session_id(request, session_id, create_if_missing=False)
    if not sid:
        return {"session_id": None, "uploads": []}
    uploads = session_manager.list_uploads(sid)
    return {"session_id": sid, "uploads": uploads}


@router.post("/session/upload")
async def upload_session_audio(
    request: Request,
    response: Response,
    audio_file: UploadFile = File(...),
    transcript: Optional[str] = Form(""),
    session_id: Optional[str] = Form(None)
):
    """
    Uploads an audio file into an isolated session namespace.
    Rejects invalid types, oversized files, and duplicate entries.
    Never persists to authoritative dataset manifest or directories.
    """
    sid = _resolve_session_id(request, session_id, create_if_missing=True)

    # Extension validation
    ext = os.path.splitext(audio_file.filename or "")[1].lower()
    if not ext or ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {sorted(list(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    audio_bytes = await audio_file.read()
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_BYTES // (1024*1024)}MB."
        )

    try:
        item = session_manager.add_upload(
            session_id=sid,
            filename=audio_file.filename or "uploaded_audio.wav",
            raw_bytes=audio_bytes,
            transcript=transcript or ""
        )
        actual_sid = item.session_id
        response.headers["X-Session-ID"] = actual_sid
        return {"session_id": actual_sid, "upload": item.to_dict()}
    except ValueError as e:
        # Duplicate upload
        raise HTTPException(status_code=409, detail=str(e))
    except AudioTooQuietError as e:
        raise HTTPException(status_code=400, detail=f"AUDIO_TOO_QUIET: {str(e)}")
    except DurationExceededError as e:
        raise HTTPException(status_code=413, detail=f"DURATION_EXCEEDED: {str(e)}")
    except (AudioFormatError, FileSizeExceededError) as e:
        raise HTTPException(status_code=400, detail=f"AUDIO_FORMAT_ERROR: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


@router.get("/session/audio/{upload_id}")
def stream_session_audio(upload_id: str, request: Request, session_id: Optional[str] = None):
    """Streams session audio file for waveform rendering and in-browser playback."""
    sid = _resolve_session_id(request, session_id, create_if_missing=False)
    if not sid:
        raise HTTPException(status_code=404, detail="Active session not found or expired.")

    item = session_manager.get_upload(sid, upload_id)
    if not item or not os.path.exists(item.file_path):
        raise HTTPException(status_code=404, detail="Session audio file not found or expired.")

    ext = os.path.splitext(item.filename)[1].lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }
    media_type = media_types.get(ext, "audio/wav")
    return FileResponse(item.file_path, media_type=media_type)


@router.post("/session/analyze/{upload_id}")
def analyze_session_upload(
    upload_id: str,
    request: Request,
    transcript: Optional[str] = Form(None),
    reference_baseline_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    """
    Executes full speech analysis pipeline on an uploaded session file.
    If a reference baseline is selected, performs paired contrastive comparison.
    Does not fabricate ground-truth annotations or modify benchmark manifests.
    """
    sid = _resolve_session_id(request, session_id, create_if_missing=False)
    if not sid:
        raise HTTPException(status_code=404, detail="Active session not found or expired.")

    try:
        result = session_manager.analyze_upload(
            session_id=sid,
            upload_id=upload_id,
            transcript=transcript,
            reference_baseline_id=reference_baseline_id
        )
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Upload '{upload_id}' not found in active session.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")


@router.delete("/session/upload/{upload_id}")
def delete_session_upload(upload_id: str, request: Request, session_id: Optional[str] = None):
    """Deletes an individual uploaded audio file and cleans up its temporary storage."""
    sid = _resolve_session_id(request, session_id, create_if_missing=False)
    if not sid:
        raise HTTPException(status_code=404, detail="Active session not found or expired.")

    success = session_manager.remove_upload(sid, upload_id)
    if not success:
        raise HTTPException(status_code=404, detail="Upload not found or already removed.")
    return {"status": "deleted", "upload_id": upload_id}


@router.post("/session/clear")
async def clear_session_uploads(request: Request, session_id: Optional[str] = Form(None)):
    """Cleans up all session-scoped uploads and releases all temporary storage for the session."""
    sid = _resolve_session_id(request, session_id, create_if_missing=False)
    if sid:
        session_manager.clear_session(sid)
    return {"status": "cleared", "session_id": sid or ""}


# ==============================================================================
# Legacy / Direct Pipeline Endpoints (Maintained for full backward compatibility)
# ==============================================================================

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    request: Request,
    audio_file: Optional[UploadFile] = File(None),
    transcript: Optional[str] = Form(""),
    baseline_audio_id: Optional[str] = Form(None)
):
    """
    Analyzes an audio file and optional transcript:
    Supports both multipart/form-data upload and application/json base64 payload.
    """
    audio_bytes: bytes = b""

    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body = await request.json()
            req = AnalyzeJsonRequest.model_validate(body)
            is_base = bool(req.baseline_audio_id)
            analysis = analyze_speech(req.audio_base64, req.transcript, is_baseline=is_base)
            return analysis
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON request: {str(e)}")

    # Multipart file upload handling
    if audio_file is None:
        raise HTTPException(status_code=400, detail="No audio file provided in request.")

    # Extension validation
    ext = os.path.splitext(audio_file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {list(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    audio_bytes = await audio_file.read()
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_BYTES // (1024*1024)}MB."
        )

    is_base = False
    if audio_file and audio_file.filename and "baseline" in audio_file.filename.lower():
        is_base = True
    elif baseline_audio_id:
        is_base = True

    try:
        analysis = analyze_speech(audio_bytes, transcript or "", is_baseline=is_base)
        return analysis
    except AudioTooQuietError as e:
        raise HTTPException(status_code=400, detail=f"AUDIO_TOO_QUIET: {str(e)}")
    except DurationExceededError as e:
        raise HTTPException(status_code=413, detail=f"DURATION_EXCEEDED: {str(e)}")
    except AudioFormatError as e:
        raise HTTPException(status_code=400, detail=f"AUDIO_FORMAT_ERROR: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal analysis failure: {str(e)}")


@router.post("/compare", response_model=ComparisonResponse)
async def compare_audio_pair(
    baseline_audio: Optional[UploadFile] = File(None),
    flawed_audio: Optional[UploadFile] = File(None),
    pair_id: Optional[str] = Form(None),
    transcript: Optional[str] = Form("")
):
    """
    Performs contrastive evaluation of baseline vs flawed speech:
    Can accept two uploaded audio files OR an existing dataset pair_id from manifest.
    """
    if pair_id:
        pair_meta = None
        manifest_data = None
        if os.path.exists(MANIFEST_PATH):
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            pair_meta = next((p for p in manifest_data.get("pairs", []) if p["pair_id"] == pair_id), None)

        if not pair_meta:
            raise HTTPException(status_code=404, detail=f"Pair ID '{pair_id}' not found in manifest.")

        base_audio_path = pair_meta["baseline_audio"]
        flaw_audio_path = pair_meta["flawed_audio"]
        base_path = base_audio_path if os.path.isabs(base_audio_path) else os.path.join(BASE_DIR, base_audio_path)
        flawed_path = flaw_audio_path if os.path.isabs(flaw_audio_path) else os.path.join(BASE_DIR, flaw_audio_path)

        text_meta = next((t for t in manifest_data.get("speech_texts", []) if t["text_id"] == pair_meta["text_id"]), None)
        canonical_text = text_meta["canonical_transcript"] if text_meta else transcript or ""

        try:
            comparison = compare_speech_pair(
                baseline_audio=base_path,
                flawed_audio=flawed_path,
                transcript=canonical_text,
                pair_id=pair_id
            )
            return comparison
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Comparison failure: {str(e)}")

    if baseline_audio is None or flawed_audio is None:
        raise HTTPException(status_code=400, detail="Must provide either pair_id or both baseline and flawed audio files.")

    base_bytes = await baseline_audio.read()
    flaw_bytes = await flawed_audio.read()

    try:
        comparison = compare_speech_pair(
            baseline_audio=base_bytes,
            flawed_audio=flaw_bytes,
            transcript=transcript or ""
        )
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failure: {str(e)}")
