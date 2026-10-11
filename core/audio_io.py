from __future__ import annotations

import io
import base64
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from core.schemas import AudioMetadata

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_S = 300.0
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
MIN_RMS_DB = -45.0
CLIPPING_THRESHOLD = 0.995


class AudioProcessingError(Exception):
    """Base exception for audio processing failures."""
    pass


class AudioTooQuietError(AudioProcessingError):
    """Raised when audio signal energy is below usable speech threshold."""
    pass


class DurationExceededError(AudioProcessingError):
    """Raised when audio exceeds maximum duration limit."""
    pass


class FileSizeExceededError(AudioProcessingError):
    """Raised when raw audio file exceeds maximum byte size."""
    pass


class AudioFormatError(AudioProcessingError):
    """Raised when audio file cannot be decoded or has invalid magic bytes."""
    pass


def decode_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decodes audio bytes into float32 numpy array and native sample rate.
    Supports WAV, FLAC, OGG, and other libsndfile-supported formats.
    """
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise FileSizeExceededError(
            f"Audio file size ({len(raw_bytes)} bytes) exceeds max limit of {MAX_FILE_BYTES} bytes (50MB)."
        )

    try:
        buffer = io.BytesIO(raw_bytes)
        data, sr = sf.read(buffer, dtype="float32", always_2d=False)
    except Exception as e:
        raise AudioFormatError(f"Failed to decode audio bytes: {str(e)}") from e

    return data, sr


def decode_base64_audio(b64_string: str) -> tuple[np.ndarray, int]:
    """Decodes base64-encoded audio payload."""
    try:
        # Strip data URL prefix if present (e.g. data:audio/wav;base64,...)
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_string)
    except Exception as e:
        raise AudioFormatError(f"Invalid base64 audio payload: {str(e)}") from e

    return decode_audio_bytes(raw_bytes)


def load_audio_file(filepath: str) -> tuple[np.ndarray, int]:
    """Reads audio file from disk path."""
    try:
        data, sr = sf.read(filepath, dtype="float32", always_2d=False)
    except Exception as e:
        raise AudioFormatError(f"Unable to read audio file '{filepath}': {str(e)}") from e

    return data, sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resamples audio to target sample rate using polyphase filtering."""
    if orig_sr == target_sr:
        return audio

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = resample_poly(audio, up, down).astype(np.float32)
    return resampled


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmixes multi-channel audio to single mono channel."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        # Average across channels (shape: [samples, channels] or [channels, samples])
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            return np.mean(audio, axis=0).astype(np.float32)
        return np.mean(audio, axis=1).astype(np.float32)
    raise AudioFormatError(f"Unsupported audio array dimension: {audio.ndim}")


def estimate_snr_db(audio: np.ndarray, frame_len: int = 512) -> float:
    """
    Estimates SNR (dB) by comparing active speech peak energy (90th percentile)
    against the background noise floor (quietest frames or nominal floor).
    """
    if len(audio) < frame_len:
        return 20.0

    num_frames = len(audio) // frame_len
    frames = audio[:num_frames * frame_len].reshape(num_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(frame_rms)

    speech_peak_db = float(np.percentile(frame_db, 90))
    quiet_frames = frame_db[frame_db < -35.0]
    if len(quiet_frames) > 0:
        noise_floor_db = float(np.median(quiet_frames))
    else:
        # If entire signal is continuous speech without pauses, nominal floor is at least 30dB below peak
        noise_floor_db = min(-55.0, speech_peak_db - 35.0)

    snr = speech_peak_db - noise_floor_db
    return float(max(0.0, min(60.0, snr)))


def process_audio(
    raw_input: bytes | str | np.ndarray,
    orig_sr: int = TARGET_SAMPLE_RATE
) -> tuple[np.ndarray, AudioMetadata]:
    """
    Standard pipeline:
    1. Parse raw input (bytes, filepath, or np.ndarray)
    2. Convert stereo/multi-channel to mono
    3. Check duration <= 300s
    4. Resample to 16,000 Hz
    5. Check silence (max RMS < -45 dBFS)
    6. Detect clipping
    7. Peak normalize
    8. Estimate SNR
    Returns (normalized_audio_16k, AudioMetadata)
    """
    if isinstance(raw_input, bytes):
        audio, sr = decode_audio_bytes(raw_input)
    elif isinstance(raw_input, str):
        if raw_input.startswith("data:audio") or len(raw_input) > 500 and not raw_input.endswith((".wav", ".mp3", ".flac", ".ogg")):
            audio, sr = decode_base64_audio(raw_input)
        else:
            audio, sr = load_audio_file(raw_input)
    elif isinstance(raw_input, np.ndarray):
        audio = raw_input.astype(np.float32)
        sr = orig_sr
    else:
        raise AudioFormatError(f"Unsupported input type for audio: {type(raw_input)}")

    # Mono conversion
    audio = convert_to_mono(audio)

    # Duration validation
    duration_s = float(len(audio) / sr)
    if duration_s > MAX_DURATION_S:
        raise DurationExceededError(
            f"Audio duration ({duration_s:.1f}s) exceeds maximum allowed duration ({MAX_DURATION_S}s)."
        )

    # Resample to 16kHz
    if sr != TARGET_SAMPLE_RATE:
        audio = resample_audio(audio, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    duration_s = float(len(audio) / TARGET_SAMPLE_RATE)
    if duration_s < 0.1:
        raise AudioTooQuietError("Audio duration too short (< 100ms) for speech analysis.")

    # Check for NaN / Inf
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    # Silence check
    overall_rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    overall_rms_db = 20.0 * np.log10(overall_rms)
    peak_val = float(np.max(np.abs(audio)))

    if overall_rms_db < MIN_RMS_DB or peak_val < 1e-4:
        raise AudioTooQuietError(
            f"Audio energy ({overall_rms_db:.1f} dBFS) is below minimum threshold of {MIN_RMS_DB} dBFS."
        )

    # Clipping detection
    clipped_samples = np.sum(np.abs(audio) >= CLIPPING_THRESHOLD)
    clipped_ratio = float(clipped_samples / len(audio))

    # Peak normalization to -0.5 dBFS (peak = 0.95)
    if peak_val > 0.0:
        audio = (audio / peak_val) * 0.95

    snr_db = estimate_snr_db(audio)

    metadata = AudioMetadata(
        duration_s=round(duration_s, 3),
        sample_rate=TARGET_SAMPLE_RATE,
        channels=1,
        snr_db=round(snr_db, 2),
        clipped_ratio=round(clipped_ratio, 4)
    )

    return audio, metadata
