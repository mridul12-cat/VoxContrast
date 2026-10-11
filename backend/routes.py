"""
VoxContrast - Backend API Routes
Exposes typed REST endpoints for speech analysis, paired contrastive evaluation,
and dataset manifest inspection.
"""

from __future__ import annotations

import os
import json
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.config import DATA_DIR, MANIFEST_PATH, MAX_UPLOAD_BYTES, ALLOWED_AUDIO_EXTENSIONS
from core.analyzer import analyze_speech, compare_speech_pair
from core.audio_io import (
    AudioTooQuietError,
    DurationExceededError,
    FileSizeExceededError,
    AudioFormatError
)
from core.schemas import AnalysisResponse, ComparisonResponse, DatasetManifest

router = APIRouter(prefix="/api")


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
def get_dataset_manifest() -> dict:
    """Returns the canonical contrastive speech dataset manifest."""
    if not os.path.exists(MANIFEST_PATH):
        raise HTTPException(status_code=404, detail="Dataset manifest not found on disk.")

    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)
    return data


@router.get("/dataset/{folder}/{filename}")
@router.get("/dataset/audio/{folder}/{filename}")
def get_audio_file(folder: str, filename: str):
    """Streams audio file from dataset directory."""
    if folder not in ("baselines", "flawed"):
        raise HTTPException(status_code=400, detail="Invalid audio folder.")

    file_path = os.path.join(DATA_DIR, "audio", folder, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found.")

    return FileResponse(file_path, media_type="audio/wav")


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
            analysis = analyze_speech(req.audio_base64, req.transcript)
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

    try:
        analysis = analyze_speech(audio_bytes, transcript or "")
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
        # Load pair from manifest
        if not os.path.exists(MANIFEST_PATH):
            raise HTTPException(status_code=404, detail="Dataset manifest not found.")
        with open(MANIFEST_PATH, "r") as f:
            manifest_data = json.load(f)

        pair_meta = next((p for p in manifest_data.get("pairs", []) if p["pair_id"] == pair_id), None)
        if not pair_meta:
            raise HTTPException(status_code=404, detail=f"Pair ID '{pair_id}' not found in manifest.")

        base_path = os.path.join(os.path.dirname(DATA_DIR), pair_meta["baseline_audio"])
        flawed_path = os.path.join(os.path.dirname(DATA_DIR), pair_meta["flawed_audio"])

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
