from __future__ import annotations

import re
import numpy as np
from core.schemas import AlignedWord, AlignmentStatus, TimeSeriesData, PauseInterval

MIN_CONFIDENCE_THRESHOLD = 0.40


def clean_transcript(text: str) -> list[str]:
    """Tokenizes raw transcript into cleaned words, stripping punctuation."""
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
    return tokens


def find_voiced_segments(
    time_series: TimeSeriesData,
    silence_db_thresh: float = -38.0
) -> list[tuple[float, float]]:
    """
    Finds contiguous speech segments (speech phrases) from energy and F0 contour.
    Merges segments with short pauses (< 250ms).
    """
    time_stamps = time_series.time_stamps
    rms_db = time_series.rms_energy_db
    f0 = time_series.f0_contour

    segments: list[tuple[float, float]] = []
    in_seg = False
    seg_start = 0.0

    for i, (t, db, pitch) in enumerate(zip(time_stamps, rms_db, f0)):
        is_speech = (db > silence_db_thresh or pitch > 0)
        if is_speech and not in_seg:
            in_seg = True
            seg_start = t
        elif not is_speech and in_seg:
            in_seg = False
            seg_end = t
            if seg_end - seg_start >= 0.15:
                segments.append((seg_start, seg_end))

    if in_seg:
        seg_end = time_stamps[-1]
        if seg_end - seg_start >= 0.15:
            segments.append((seg_start, seg_end))

    # Merge closely spaced segments (< 250ms gap)
    merged: list[tuple[float, float]] = []
    for s_start, s_end in segments:
        if not merged:
            merged.append((s_start, s_end))
        else:
            prev_start, prev_end = merged[-1]
            if s_start - prev_end <= 0.25:
                merged[-1] = (prev_start, s_end)
            else:
                merged.append((s_start, s_end))

    return merged


def align_transcript_to_audio(
    transcript: str,
    time_series: TimeSeriesData,
    pauses: list[PauseInterval],
    audio_duration_s: float
) -> tuple[AlignmentStatus, TimeSeriesData]:
    """
    Performs forced alignment of transcript words to acoustic speech segments.
    Explicitly flags ALIGNMENT_DEGRADED or FAILED when confidence is low.
    """
    words = clean_transcript(transcript)
    word_count = len(words)

    # Empty transcript case
    if word_count == 0:
        return (
            AlignmentStatus(
                status="FAILED",
                confidence=0.0,
                word_count=0,
                aligned_words=[],
                failure_reason="Empty or whitespace-only transcript provided."
            ),
            time_series
        )

    speech_segments = find_voiced_segments(time_series)
    total_speech_duration = sum(end - start for start, end in speech_segments)

    # Check for extreme mismatch
    # E.g., 50 words in 1 second, or 1 word in 40 seconds of continuous speech
    expected_wpm = (word_count / (total_speech_duration / 60.0)) if total_speech_duration > 0 else 0.0

    if total_speech_duration < 0.2:
        return (
            AlignmentStatus(
                status="FAILED",
                confidence=0.0,
                word_count=word_count,
                aligned_words=[],
                failure_reason="No voiced speech segments detected in audio signal."
            ),
            time_series
        )

    # Check alignment confidence based on plausible WPM bounds
    # English conversational speech is rarely < 40 WPM or > 350 WPM
    confidence = 1.0
    status_str = "SUCCESS"
    reason = None

    if expected_wpm < 45.0:
        confidence = max(0.20, expected_wpm / 100.0)
        status_str = "DEGRADED"
        reason = f"Transcript word density too sparse ({expected_wpm:.1f} WPM) for active speech duration."
    elif expected_wpm > 320.0:
        confidence = max(0.15, 300.0 / expected_wpm)
        status_str = "DEGRADED"
        reason = f"Transcript word density exceeds plausible human articulation limit ({expected_wpm:.1f} WPM)."

    if confidence < MIN_CONFIDENCE_THRESHOLD:
        status_str = "DEGRADED"

    # Distribute words across speech segments weighted by segment duration and character lengths
    char_lengths = [max(1, len(w)) for w in words]
    total_chars = sum(char_lengths)

    t_arr = np.array(time_series.time_stamps)
    rms_arr = np.array(time_series.rms_energy_db)
    f0_arr = np.array(time_series.f0_contour)

    # 1. Identify speech segments between significant pauses (>= 0.45s)
    sig_pauses = sorted([p for p in pauses if p.duration_s >= 0.45], key=lambda x: x.start_s)
    seg_bounds = []
    curr = 0.0
    for p in sig_pauses:
        if p.start_s - curr >= 0.20:
            seg_bounds.append((curr, p.start_s))
        curr = p.end_s
    if audio_duration_s - curr >= 0.20:
        seg_bounds.append((curr, audio_duration_s))
    if not seg_bounds:
        seg_bounds = [(0.0, audio_duration_s)]

    # Trim leading/trailing unvoiced silence from each segment
    trimmed_segs = []
    for s_start, s_end in seg_bounds:
        v_idx = [i for i, ts_val in enumerate(t_arr) if s_start <= ts_val <= s_end and (f0_arr[i] > 0 or rms_arr[i] > -35.0)]
        if v_idx:
            t_s = max(s_start, t_arr[v_idx[0]] - 0.05)
            t_e = min(s_end, t_arr[v_idx[-1]] + 0.05)
            if t_e - t_s >= 0.15:
                trimmed_segs.append((round(float(t_s), 3), round(float(t_e), 3)))
            else:
                trimmed_segs.append((round(float(s_start), 3), round(float(s_end), 3)))
        else:
            trimmed_segs.append((round(float(s_start), 3), round(float(s_end), 3)))

    # Filter acoustic peaks across the audio
    all_peaks = []
    for i in range(1, len(rms_arr) - 1):
        if (f0_arr[i] > 0 or rms_arr[i] > -32.0) and rms_arr[i] >= rms_arr[i - 1] and rms_arr[i] >= rms_arr[i + 1]:
            all_peaks.append(i)
    filt_all_peaks = []
    for pk in all_peaks:
        if not filt_all_peaks or t_arr[pk] - t_arr[filt_all_peaks[-1]] >= 0.10:
            filt_all_peaks.append(pk)
        elif rms_arr[pk] > rms_arr[filt_all_peaks[-1]]:
            filt_all_peaks[-1] = pk

    seg_peaks = [sum(1 for pk in filt_all_peaks if s <= t_arr[pk] <= e) for s, e in trimmed_segs]
    total_p = sum(seg_peaks) or 1
    total_dur = sum(e - s for s, e in trimmed_segs) or 1.0

    seg_word_counts = []
    assigned = 0
    for idx, (s, e) in enumerate(trimmed_segs):
        p_cnt = seg_peaks[idx]
        seg_dur = e - s
        if idx == len(trimmed_segs) - 1:
            cnt = word_count - assigned
        else:
            weight = 0.5 * (p_cnt / total_p) + 0.5 * (seg_dur / total_dur)
            cnt = int(round(word_count * weight))
            cnt = max(1, min(word_count - assigned - (len(trimmed_segs) - 1 - idx), cnt))
        seg_word_counts.append(cnt)
        assigned += cnt

    aligned_words: list[AlignedWord] = []
    w_idx = 0
    word_conf = round(confidence * (0.95 if status_str == "SUCCESS" else 0.55), 2)

    for seg_idx, (s_start, s_end) in enumerate(trimmed_segs):
        k_words = seg_word_counts[seg_idx]
        if k_words <= 0:
            continue
        sub_words = words[w_idx : w_idx + k_words]
        sub_char_lens = char_lengths[w_idx : w_idx + k_words]
        w_idx += k_words

        # Sub-peaks in this segment
        sub_peaks = []
        for i in range(1, len(rms_arr) - 1):
            if s_start <= t_arr[i] <= s_end and (f0_arr[i] > 0 or rms_arr[i] > -32.0):
                if rms_arr[i] >= rms_arr[i - 1] and rms_arr[i] >= rms_arr[i + 1]:
                    sub_peaks.append(i)

        filt_peaks = []
        for pk in sub_peaks:
            if not filt_peaks or t_arr[pk] - t_arr[filt_peaks[-1]] >= 0.10:
                filt_peaks.append(pk)
            elif rms_arr[pk] > rms_arr[filt_peaks[-1]]:
                filt_peaks[-1] = pk

        if len(filt_peaks) == k_words:
            valleys = [max(0, filt_peaks[0] - 5)]
            for p1, p2 in zip(filt_peaks[:-1], filt_peaks[1:]):
                v_idx = p1 + np.argmin(rms_arr[p1 : p2 + 1])
                valleys.append(v_idx)
            valleys.append(min(len(t_arr) - 1, filt_peaks[-1] + 5))

            for j in range(k_words):
                ws = round(float(t_arr[valleys[j]]), 3)
                we = round(float(t_arr[valleys[j + 1]]), 3)
                if we <= ws:
                    we = round(ws + 0.15, 3)
                aligned_words.append(
                    AlignedWord(
                        word=sub_words[j],
                        start_s=ws,
                        end_s=we,
                        confidence=word_conf
                    )
                )
        else:
            # Acoustic valley snapping based on character length progression
            sub_total_chars = sum(sub_char_lens)
            cum_chars = np.cumsum([0] + sub_char_lens)
            bounds = [s_start]
            for j in range(1, k_words):
                target_t = s_start + (s_end - s_start) * (cum_chars[j] / sub_total_chars)
                min_a = bounds[-1] + 0.12
                max_a = s_end - (k_words - j) * 0.12
                if min_a >= max_a:
                    b_t = min_a
                else:
                    search_start = max(min_a, target_t - 0.25)
                    search_end = min(max_a, target_t + 0.25)
                    s_indices = [idx for idx, ts_v in enumerate(t_arr) if search_start <= ts_v <= search_end]
                    if s_indices:
                        b_t = t_arr[s_indices[np.argmin(rms_arr[s_indices])]]
                    else:
                        b_t = target_t
                bounds.append(round(float(b_t), 3))
            bounds.append(round(float(s_end), 3))

            for j in range(k_words):
                aligned_words.append(
                    AlignedWord(
                        word=sub_words[j],
                        start_s=bounds[j],
                        end_s=bounds[j + 1],
                        confidence=word_conf
                    )
                )

    # Refine speech_rate_wpm blending acoustic rate and aligned word context
    refined_wpm = refine_speech_rate_with_alignment(
        time_series, aligned_words, audio_duration_s
    )
    updated_time_series = time_series.model_copy(
        update={"speech_rate_wpm": refined_wpm}
    )

    alignment_status = AlignmentStatus(
        status=status_str,
        confidence=round(confidence, 2),
        word_count=word_count,
        aligned_words=aligned_words,
        failure_reason=reason
    )

    return alignment_status, updated_time_series


def refine_speech_rate_with_alignment(
    time_series: TimeSeriesData,
    aligned_words: list[AlignedWord],
    duration_s: float,
    window_s: float = 1.0
) -> list[float]:
    """Refines frame-by-frame speech rate (WPM) using aligned word durations and local context."""
    if not aligned_words:
        return time_series.speech_rate_wpm

    n_words = len(aligned_words)
    word_wpms = []
    for i in range(n_words):
        # 3-word local context
        start_idx = max(0, i - 1)
        end_idx = min(n_words, i + 2)
        ctx_words = aligned_words[start_idx:end_idx]
        ctx_dur = max(0.12, sum(w.end_s - w.start_s for w in ctx_words))
        wpm = (len(ctx_words) / ctx_dur) * 60.0
        # Instantaneous single-word rate
        single_dur = max(0.08, aligned_words[i].end_s - aligned_words[i].start_s)
        single_wpm = (1.0 / single_dur) * 60.0
        blended = max(wpm, single_wpm) if single_wpm >= 170.0 else wpm
        word_wpms.append(float(np.clip(blended, 60.0, 320.0)))

    time_stamps = time_series.time_stamps
    orig_wpm = time_series.speech_rate_wpm
    refined = []

    for idx, t in enumerate(time_stamps):
        # If audio is silent or unvoiced pause, decay to 0
        if time_series.f0_contour[idx] == 0 and time_series.rms_energy_db[idx] <= -38.0:
            refined.append(0.0)
            continue

        active_wpm = orig_wpm[idx] if idx < len(orig_wpm) else 0.0
        for i, w in enumerate(aligned_words):
            w_reach = min(w.end_s + 0.10, aligned_words[i + 1].start_s) if i + 1 < n_words else w.end_s + 0.10
            if w.start_s <= t <= w_reach:
                active_wpm = max(active_wpm, word_wpms[i])
                break
            elif t < w.start_s:
                break
        refined.append(round(active_wpm, 1))

    return refined
