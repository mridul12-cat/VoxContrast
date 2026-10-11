from __future__ import annotations

import numpy as np
from core.schemas import TimeSeriesData, PauseInterval, FeaturesSummary

TIME_STEP_S = 0.05  # 50ms time step (20 Hz analysis rate)
FRAME_LEN = 1024    # 64ms window at 16kHz for low-frequency F0 stability
PITCH_MIN_HZ = 60.0
PITCH_MAX_HZ = 400.0
YIN_THRESHOLD = 0.20
SILENCE_ENERGY_DB = -38.0
MIN_PAUSE_DURATION_S = 0.40


def compute_yin_f0(
    frame: np.ndarray,
    sr: int = 16000,
    min_hz: float = PITCH_MIN_HZ,
    max_hz: float = PITCH_MAX_HZ,
    threshold: float = YIN_THRESHOLD
) -> float:
    """
    Computes fundamental frequency (F0) using the YIN algorithm with parabolic interpolation.
    Returns 0.0 for unvoiced frames.
    """
    tau_min = int(sr / max_hz)
    tau_max = int(sr / min_hz)
    w_len = len(frame) // 2

    if tau_max >= w_len:
        tau_max = w_len - 1

    # 1. Difference function d(tau)
    d = np.zeros(tau_max + 1, dtype=np.float32)
    for tau in range(1, tau_max + 1):
        diff = frame[:w_len] - frame[tau:w_len + tau]
        d[tau] = np.sum(diff ** 2)

    # 2. Cumulative mean normalized difference d'(tau)
    d_prime = np.ones(tau_max + 1, dtype=np.float32)
    cumsum = 0.0
    for tau in range(1, tau_max + 1):
        cumsum += d[tau]
        if cumsum > 0:
            d_prime[tau] = d[tau] / (cumsum / tau)
        else:
            d_prime[tau] = 1.0

    # 3. Absolute thresholding: find first minimum below threshold
    best_tau = None
    for tau in range(tau_min, tau_max):
        if d_prime[tau] < threshold:
            while tau + 1 < tau_max and d_prime[tau + 1] < d_prime[tau]:
                tau += 1
            best_tau = tau
            break

    # If no dip below threshold, check global minimum if sufficiently periodic
    if best_tau is None:
        min_idx = tau_min + np.argmin(d_prime[tau_min:tau_max])
        if d_prime[min_idx] < threshold * 1.5:
            best_tau = min_idx
        else:
            return 0.0

    # 4. Parabolic interpolation
    if 0 < best_tau < tau_max:
        s0 = d_prime[best_tau - 1]
        s1 = d_prime[best_tau]
        s2 = d_prime[best_tau + 1]
        denom = 2 * (s0 - 2 * s1 + s2)
        if abs(denom) > 1e-6:
            delta = (s0 - s2) / denom
            best_tau_interp = best_tau + delta
        else:
            best_tau_interp = float(best_tau)
    else:
        best_tau_interp = float(best_tau)

    if best_tau_interp <= 0:
        return 0.0

    f0 = sr / best_tau_interp
    if min_hz <= f0 <= max_hz:
        return float(f0)
    return 0.0


def extract_acoustic_contours(
    audio: np.ndarray,
    sr: int = 16000,
    time_step_s: float = TIME_STEP_S
) -> tuple[TimeSeriesData, list[PauseInterval]]:
    """
    Extracts time series contours (F0, RMS, spectral centroid, speech rate)
    and identifies pause intervals.
    """
    hop_len = int(sr * time_step_s)
    num_samples = len(audio)
    num_frames = max(1, int(np.ceil((num_samples - FRAME_LEN) / hop_len)) + 1)

    time_stamps: list[float] = []
    f0_list: list[float] = []
    rms_db_list: list[float] = []
    spectral_centroid_list: list[float] = []

    # FFT frequency bins for spectral centroid
    freq_bins = np.fft.rfftfreq(FRAME_LEN, d=1.0 / sr)
    window = np.hanning(FRAME_LEN)

    for i in range(num_frames):
        start_idx = i * hop_len
        end_idx = start_idx + FRAME_LEN

        if end_idx <= num_samples:
            frame = audio[start_idx:end_idx]
        else:
            # Pad with zeros if necessary
            frame = np.zeros(FRAME_LEN, dtype=np.float32)
            remaining = num_samples - start_idx
            if remaining > 0:
                frame[:remaining] = audio[start_idx:num_samples]

        t_sec = round(i * time_step_s, 3)
        time_stamps.append(t_sec)

        # RMS Energy
        frame_rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        frame_db = float(20.0 * np.log10(frame_rms))
        frame_db = max(-80.0, min(0.0, frame_db))
        rms_db_list.append(round(frame_db, 2))

        # Spectral Centroid
        windowed = frame * window
        fft_mag = np.abs(np.fft.rfft(windowed))
        sum_mag = np.sum(fft_mag)
        if sum_mag > 1e-6:
            centroid = float(np.sum(freq_bins * fft_mag) / sum_mag)
        else:
            centroid = 0.0
        spectral_centroid_list.append(round(centroid, 1))

        # F0 Pitch
        if frame_db < SILENCE_ENERGY_DB:
            f0_list.append(0.0)
        else:
            f0 = compute_yin_f0(frame, sr)
            f0_list.append(round(f0, 2))

    # Detect pause intervals
    pauses: list[PauseInterval] = []
    in_pause = False
    pause_start = 0.0

    for i, db in enumerate(rms_db_list):
        t = time_stamps[i]
        is_silent = (db <= SILENCE_ENERGY_DB)

        if is_silent and not in_pause:
            in_pause = True
            pause_start = t
        elif not is_silent and in_pause:
            in_pause = False
            pause_dur = round(t - pause_start, 3)
            if pause_dur >= MIN_PAUSE_DURATION_S:
                pauses.append(
                    PauseInterval(
                        start_s=pause_start,
                        end_s=t,
                        duration_s=pause_dur,
                        is_excessive=(pause_dur >= 1.80)
                    )
                )

    if in_pause:
        last_t = time_stamps[-1]
        pause_dur = round(last_t - pause_start, 3)
        if pause_dur >= MIN_PAUSE_DURATION_S:
            pauses.append(
                PauseInterval(
                    start_s=pause_start,
                    end_s=last_t,
                    duration_s=pause_dur,
                    is_excessive=(pause_dur >= 1.80)
                )
            )

    # Compute speaker median F0 for semitone conversion
    voiced_f0 = [f for f in f0_list if f > 0]
    median_f0 = float(np.median(voiced_f0)) if len(voiced_f0) >= 3 else 150.0

    # Semitone contour relative to median F0: 12 * log2(F0 / median_F0)
    f0_semitones: list[float] = []
    for f in f0_list:
        if f > 0:
            st = 12.0 * np.log2(f / median_f0)
            f0_semitones.append(round(float(st), 2))
        else:
            f0_semitones.append(0.0)

    # Compute initial local speech rate contour (WPM)
    # Estimated via vocal activity pulses and syllabic peak detection
    speech_rate_wpm = estimate_local_wpm(
        rms_db_list, f0_list, time_stamps, time_step_s
    )

    time_series = TimeSeriesData(
        time_step_s=time_step_s,
        time_stamps=time_stamps,
        f0_contour=f0_list,
        f0_semitones=f0_semitones,
        rms_energy_db=rms_db_list,
        speech_rate_wpm=speech_rate_wpm,
        spectral_centroid_hz=spectral_centroid_list
    )

    return time_series, pauses


def estimate_local_wpm(
    rms_db: list[float],
    f0_contour: list[float],
    time_stamps: list[float],
    time_step_s: float,
    window_s: float = 1.0
) -> list[float]:
    """
    Estimates local speech rate (WPM) over sliding windows using syllabic energy
    peaks within voiced segments combined with inter-peak instantaneous rate.
    """
    n_frames = len(rms_db)
    wpm_contour = [0.0] * n_frames

    # Find syllable nucleus candidates: local energy peaks in voiced regions
    peaks = []
    for i in range(1, n_frames - 1):
        if f0_contour[i] > 0 and rms_db[i] > SILENCE_ENERGY_DB + 6.0:
            if rms_db[i] >= rms_db[i - 1] and rms_db[i] >= rms_db[i + 1]:
                peaks.append(i)

    # Filter out spurious peaks closer than 100ms
    filt_peaks = []
    for pk in peaks:
        if not filt_peaks or time_stamps[pk] - time_stamps[filt_peaks[-1]] >= 0.10:
            filt_peaks.append(pk)
        elif rms_db[pk] > rms_db[filt_peaks[-1]]:
            filt_peaks[-1] = pk

    peak_times = [time_stamps[idx] for idx in filt_peaks]

    for i in range(n_frames):
        t = time_stamps[i]

        # If current frame is silent/pause, local rate decays to 0
        if f0_contour[i] == 0 and rms_db[i] <= SILENCE_ENERGY_DB:
            wpm_contour[i] = 0.0
            continue

        # Count peaks in [t - window_s/2, t + window_s/2]
        win_start = max(0.0, t - window_s / 2.0)
        win_end = min(time_stamps[-1], t + window_s / 2.0)
        win_dur = max(0.2, win_end - win_start)

        cnt = sum(1 for pt in peak_times if win_start <= pt <= win_end)
        dens_wpm = (cnt / win_dur) * 60.0

        # Inter-peak instantaneous rate
        prev_p = [pt for pt in peak_times if pt <= t]
        next_p = [pt for pt in peak_times if pt > t]
        inst_wpm = 135.0
        if prev_p and next_p:
            delta = next_p[0] - prev_p[-1]
            if delta < 0.70:
                inst_wpm = 60.0 / delta

        blended = max(dens_wpm, inst_wpm) if inst_wpm >= 170.0 else dens_wpm
        wpm_contour[i] = round(float(np.clip(blended, 60.0, 320.0)), 1)

    return wpm_contour


def summarize_features(
    time_series: TimeSeriesData,
    pauses: list[PauseInterval]
) -> FeaturesSummary:
    """Computes summary statistics across all acoustic features."""
    voiced_f0 = [f for f in time_series.f0_contour if f > 0]
    voiced_st = [st for st, f in zip(time_series.f0_semitones, time_series.f0_contour) if f > 0]
    voiced_rms = [db for db in time_series.rms_energy_db if db > SILENCE_ENERGY_DB]
    active_wpm = [w for w in time_series.speech_rate_wpm if w > 0]

    pitch_mean = float(np.mean(voiced_f0)) if voiced_f0 else 0.0
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0
    pitch_median = float(np.median(voiced_f0)) if voiced_f0 else 0.0
    pitch_semitone_std = float(np.std(voiced_st)) if len(voiced_st) > 1 else 0.0

    energy_mean = float(np.mean(voiced_rms)) if voiced_rms else -50.0
    energy_std = float(np.std(voiced_rms)) if len(voiced_rms) > 1 else 0.0

    avg_wpm = float(np.mean(active_wpm)) if active_wpm else 140.0
    pause_count = len(pauses)
    total_pause_dur = sum(p.duration_s for p in pauses)
    spectral_centroid_mean = float(np.mean(time_series.spectral_centroid_hz)) if time_series.spectral_centroid_hz else 1800.0

    return FeaturesSummary(
        average_wpm=round(avg_wpm, 1),
        pitch_mean_hz=round(pitch_mean, 1),
        pitch_std_hz=round(pitch_std, 1),
        pitch_median_hz=round(pitch_median, 1),
        pitch_semitone_std=round(pitch_semitone_std, 2),
        energy_mean_db=round(energy_mean, 1),
        energy_std_db=round(energy_std, 1),
        pause_count=pause_count,
        total_pause_duration_s=round(total_pause_dur, 2),
        spectral_centroid_mean_hz=round(spectral_centroid_mean, 1)
    )
