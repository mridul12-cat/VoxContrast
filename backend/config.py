"""
VoxContrast - Backend Service Configuration
Defines server parameters, upload boundaries, and dataset directories.
"""

from __future__ import annotations
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_DURATION_S = 300.0                # 5 minutes
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
