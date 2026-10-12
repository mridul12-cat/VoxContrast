"""
VoxContrast - Session Storage & Upload Isolation Manager
Provides thread-safe, session-isolated storage for exploratory audio uploads.
Enforces unpredictable session IDs, isolated temporary storage, deduplication,
automatic TTL cleanup, and complete isolation from authoritative datasets.
"""

from __future__ import annotations

import os
import re
import time
import shutil
import secrets
import tempfile
import threading
from typing import Optional, Dict, Any, List
import soundfile as sf
import numpy as np

from backend.config import (
    BASE_DIR,
    DATA_DIR,
    MANIFEST_PATH,
    MAX_UPLOAD_BYTES,
    ALLOWED_AUDIO_EXTENSIONS
)
from core.audio_io import (
    decode_audio_bytes,
    process_audio,
    AudioTooQuietError,
    DurationExceededError,
    FileSizeExceededError,
    AudioFormatError
)
from core.analyzer import analyze_speech, compare_speech_pair

# Base temporary directory for session storage
SESSION_TEMP_BASE = os.path.join(tempfile.gettempdir(), "voxcontrast_sessions")
SESSION_TTL_SECONDS = 7200  # 2 hours


SESSION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{16,128}$")


class SessionUploadItem:
    def __init__(
        self,
        upload_id: str,
        session_id: str,
        filename: str,
        file_path: str,
        file_size_bytes: int,
        duration_s: float,
        sample_rate: int,
        channels: int,
        snr_db: float,
        transcript: str = ""
    ):
        self.upload_id = upload_id
        self.session_id = session_id
        self.filename = filename
        self.file_path = file_path
        self.file_size_bytes = file_size_bytes
        self.duration_s = duration_s
        self.sample_rate = sample_rate
        self.channels = channels
        self.snr_db = snr_db
        self.transcript = transcript
        self.upload_time = time.time()
        self.upload_status = "uploaded"  # uploaded, failed
        self.analysis_status = "pending"  # pending, analyzed, failed
        self.analysis_data: Optional[Dict[str, Any]] = None
        self.comparison_data: Optional[Dict[str, Any]] = None
        self.reference_baseline_id: Optional[str] = None
        self.error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "session_id": self.session_id,
            "filename": self.filename,
            "duration_s": round(self.duration_s, 2),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "snr_db": round(self.snr_db, 1),
            "file_size_bytes": self.file_size_bytes,
            "transcript": self.transcript,
            "upload_time": self.upload_time,
            "upload_status": self.upload_status,
            "analysis_status": self.analysis_status,
            "has_analysis": self.analysis_data is not None,
            "has_comparison": self.comparison_data is not None,
            "reference_baseline_id": self.reference_baseline_id,
            "error_message": self.error_message,
        }


class SessionData:
    def __init__(self, session_id: str):
        if not SESSION_ID_REGEX.match(session_id):
            raise ValueError(f"Invalid session_id format: '{session_id}'")
        self.session_id = session_id
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.uploads: Dict[str, SessionUploadItem] = {}
        
        target_dir = os.path.realpath(os.path.join(SESSION_TEMP_BASE, session_id))
        base_dir = os.path.realpath(SESSION_TEMP_BASE)
        if not target_dir.startswith(base_dir):
            raise ValueError(f"Session path escape attempt: '{session_id}'")
        self.dir_path = target_dir
        os.makedirs(self.dir_path, exist_ok=True, mode=0o700)

    def touch(self):
        self.last_accessed = time.time()


class SessionManager:
    """
    Manages session-scoped uploads with isolated temporary directories and lifecycle cleanup.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: Dict[str, SessionData] = {}
        os.makedirs(SESSION_TEMP_BASE, exist_ok=True, mode=0o700)
        self._sweep_stale_directories_on_startup()

    def _sweep_stale_directories_on_startup(self):
        """Purges any abandoned session directories left from prior process executions."""
        try:
            if not os.path.exists(SESSION_TEMP_BASE):
                return
            for entry in os.listdir(SESSION_TEMP_BASE):
                entry_path = os.path.join(SESSION_TEMP_BASE, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
        except Exception:
            pass

    def create_session(self) -> str:
        with self._lock:
            self._cleanup_expired_unlocked()
            session_id = secrets.token_urlsafe(32)
            self._sessions[session_id] = SessionData(session_id)
            return session_id

    def get_or_create_session(self, session_id: Optional[str]) -> tuple[str, SessionData]:
        with self._lock:
            self._cleanup_expired_unlocked()
            if session_id and SESSION_ID_REGEX.match(session_id) and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session_id, session

            # Create new session if none provided or invalid/unknown
            new_id = secrets.token_urlsafe(32)
            session = SessionData(new_id)
            self._sessions[new_id] = session
            return new_id, session

    def get_session(self, session_id: str) -> Optional[SessionData]:
        with self._lock:
            if not session_id or not SESSION_ID_REGEX.match(session_id):
                return None
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    def add_upload(
        self,
        session_id: str,
        filename: str,
        raw_bytes: bytes,
        transcript: str = ""
    ) -> SessionUploadItem:
        with self._lock:
            actual_sid, session = self.get_or_create_session(session_id)

            # Check for duplicate within the same session
            for item in session.uploads.values():
                if item.filename == filename and item.file_size_bytes == len(raw_bytes):
                    raise ValueError(f"Duplicate upload: '{filename}' already exists in the current session.")

            # Validate audio decode & metadata
            try:
                sig, meta = process_audio(raw_bytes)
            except Exception as e:
                raise AudioFormatError(f"Audio processing failed: {str(e)}") from e

            # Save to isolated session dir with safe filename
            upload_id = f"up_{secrets.token_hex(8)}"
            safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            stored_filename = f"{upload_id}_{safe_name}"
            file_path = os.path.join(session.dir_path, stored_filename)

            with open(file_path, "wb") as f:
                f.write(raw_bytes)
            os.chmod(file_path, 0o600)

            item = SessionUploadItem(
                upload_id=upload_id,
                session_id=actual_sid,
                filename=filename,
                file_path=file_path,
                file_size_bytes=len(raw_bytes),
                duration_s=meta.duration_s,
                sample_rate=meta.sample_rate,
                channels=meta.channels,
                snr_db=meta.snr_db,
                transcript=transcript
            )

            session.uploads[upload_id] = item
            return item

    def get_upload(self, session_id: str, upload_id: str) -> Optional[SessionUploadItem]:
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return None
            return session.uploads.get(upload_id)

    def list_uploads(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return []
            return [item.to_dict() for item in session.uploads.values()]

    def remove_upload(self, session_id: str, upload_id: str) -> bool:
        with self._lock:
            session = self.get_session(session_id)
            if not session or upload_id not in session.uploads:
                return False

            item = session.uploads.pop(upload_id)
            if os.path.exists(item.file_path):
                try:
                    os.remove(item.file_path)
                except OSError:
                    pass
            return True

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                return False

            if os.path.exists(session.dir_path):
                try:
                    shutil.rmtree(session.dir_path, ignore_errors=True)
                except OSError:
                    pass
            return True

    def analyze_upload(
        self,
        session_id: str,
        upload_id: str,
        transcript: Optional[str] = None,
        reference_baseline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._lock:
            item = self.get_upload(session_id, upload_id)
            if not item:
                raise KeyError(f"Upload '{upload_id}' not found in session '{session_id}'.")

            effective_transcript = transcript if transcript is not None else item.transcript

            try:
                # 1. Run standalone speech analysis on uploaded file
                analysis = analyze_speech(
                    audio_input=item.file_path,
                    transcript=effective_transcript,
                    is_baseline=False
                )
                item.analysis_data = analysis.model_dump()
                item.analysis_status = "analyzed"
                item.transcript = effective_transcript
                item.error_message = None

                # 2. If a reference baseline is selected, run comparison
                comparison_result = None
                if reference_baseline_id:
                    ref_audio_path, ref_transcript = self._resolve_reference_audio(reference_baseline_id)
                    if not ref_audio_path or not os.path.exists(ref_audio_path):
                        raise ValueError(f"Reference baseline audio '{reference_baseline_id}' not found.")

                    comp_transcript = effective_transcript or ref_transcript or ""
                    comp = compare_speech_pair(
                        baseline_audio=ref_audio_path,
                        flawed_audio=item.file_path,
                        transcript=comp_transcript,
                        pair_id=f"session_cmp_{upload_id}"
                    )
                    comparison_result = comp.model_dump()
                    item.comparison_data = comparison_result
                    item.reference_baseline_id = reference_baseline_id

                return {
                    "upload_id": upload_id,
                    "analysis": item.analysis_data,
                    "comparison": comparison_result
                }
            except Exception as e:
                item.analysis_status = "failed"
                item.error_message = str(e)
                raise

    def _resolve_reference_audio(self, reference_id: str) -> tuple[Optional[str], Optional[str]]:
        """
        Resolves reference audio path and canonical transcript from manifest or baseline directory.
        Guarantees isolation against path traversal.
        """
        if not reference_id:
            return None, None

        # Sanitize reference_id against path traversal
        clean_ref = os.path.basename(reference_id.strip())

        manifest_data = None
        if os.path.exists(MANIFEST_PATH):
            import json
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                pass

        if manifest_data:
            # 1. Check if reference_id matches a pair_id
            for p in manifest_data.get("pairs", []):
                if p["pair_id"] == reference_id or p["pair_id"] == clean_ref:
                    p_audio = p["baseline_audio"]
                    full_audio = p_audio if os.path.isabs(p_audio) else os.path.join(BASE_DIR, p_audio)
                    text_meta = next((t for t in manifest_data.get("speech_texts", []) if t["text_id"] == p.get("text_id")), None)
                    transcript = text_meta.get("canonical_transcript", "") if text_meta else ""
                    return full_audio, transcript

            # 2. Check if reference_id matches baseline filename or text_id
            for p in manifest_data.get("pairs", []):
                if clean_ref in os.path.basename(p.get("baseline_audio", "")) or p.get("text_id") == clean_ref:
                    p_audio = p["baseline_audio"]
                    full_audio = p_audio if os.path.isabs(p_audio) else os.path.join(BASE_DIR, p_audio)
                    text_meta = next((t for t in manifest_data.get("speech_texts", []) if t["text_id"] == p.get("text_id")), None)
                    transcript = text_meta.get("canonical_transcript", "") if text_meta else ""
                    return full_audio, transcript

        # 3. Check directly in baselines folder (with strict directory confinement)
        baselines_dir = os.path.realpath(os.path.join(DATA_DIR, "audio", "baselines"))
        candidate_path = os.path.realpath(os.path.join(baselines_dir, clean_ref))
        if candidate_path.startswith(baselines_dir) and os.path.exists(candidate_path):
            transcript = ""
            if manifest_data:
                for p in manifest_data.get("pairs", []):
                    if os.path.basename(p.get("baseline_audio", "")) == clean_ref:
                        text_meta = next((t for t in manifest_data.get("speech_texts", []) if t["text_id"] == p.get("text_id")), None)
                        if text_meta:
                            transcript = text_meta.get("canonical_transcript", "")
                            break
            return candidate_path, transcript

        return None, None

    def _cleanup_expired_unlocked(self):
        """Purges sessions that exceed TTL."""
        now = time.time()
        expired_ids = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_accessed) > SESSION_TTL_SECONDS
        ]
        for sid in expired_ids:
            self.clear_session(sid)


# Global singleton SessionManager
session_manager = SessionManager()
