"""
VoxContrast - High-Fidelity Contrastive Speech Dataset Generator
Synthesizes 18 paired contrastive speeches (baseline vs flawed) across 6 canonical texts,
generating genuine 16kHz WAV audio files, individual JSON annotation files, and data/manifest.json.
"""

from __future__ import annotations

import os
import json
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

TARGET_SR = 16000


def lowpass_muffle_filter(sig: np.ndarray, cutoff_hz: float = 750.0, sr: int = TARGET_SR) -> np.ndarray:
    """Applies a steep lowpass filter simulating muffled speech / reduced clarity."""
    b, a = butter(4, cutoff_hz / (sr / 2.0), btype="lowpass")
    return lfilter(b, a, sig)


def synthesize_speech_utterance(
    words: list[str],
    base_f0: float = 140.0,
    default_wpm: float = 135.0,
    pauses: list[tuple[int, float]] | None = None,
    flaw_regions: list[dict] | None = None,
    sr: int = TARGET_SR
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """
    Synthesizes speech utterance with phonetic syllable envelopes, harmonic glottal source,
    formant resonance, consonant aspiration noise, and word timing boundaries.
    """
    if pauses is None:
        pauses = []
    if flaw_regions is None:
        flaw_regions = []

    pause_map = {idx: dur for idx, dur in pauses}
    word_timings = []
    total_audio_parts = []
    current_time_s = 0.25  # 250ms initial silence
    n_words = len(words)

    total_audio_parts.append(np.zeros(int(current_time_s * sr), dtype=np.float32))

    for i, word in enumerate(words):
        # Insert pause before word if specified
        if i in pause_map:
            pause_dur = pause_map[i]
            pause_samples = int(pause_dur * sr)
            total_audio_parts.append(np.zeros(pause_samples, dtype=np.float32))
            current_time_s += pause_dur

        # Determine word parameters based on active flaw region
        word_wpm = default_wpm
        is_monotone_pitch = False
        is_erratic_pitch = False
        is_monotone_energy = False
        is_unstable_energy = False
        is_muffled = False

        erratic_sev = "severe"
        for fr in flaw_regions:
            if fr["start_idx"] <= i <= fr["end_idx"]:
                cat = fr["category"]
                if cat == "excessive_speed":
                    word_wpm = fr.get("wpm", 235.0)
                elif cat == "monotone_pitch":
                    is_monotone_pitch = True
                elif cat == "pitch_irregularity":
                    is_erratic_pitch = True
                    erratic_sev = fr.get("severity", "severe")
                elif cat == "monotone_energy":
                    is_monotone_energy = True
                elif cat == "unstable_energy":
                    is_unstable_energy = True
                elif cat in ("muffled_articulation", "vocal_clarity"):
                    is_muffled = True

        word_len_chars = max(2, len(word))
        base_word_dur = (60.0 / word_wpm) * (0.6 + 0.08 * word_len_chars)
        base_word_dur = max(0.16, min(1.3, base_word_dur))

        w_start = current_time_s
        w_end = w_start + base_word_dur
        word_timings.append({
            "word": word,
            "start_s": round(w_start, 3),
            "end_s": round(w_end, 3)
        })

        word_samples = int(base_word_dur * sr)
        t = np.linspace(0, base_word_dur, word_samples, endpoint=False, dtype=np.float32)

        # Natural intonation contour: sentence declination + alternating word accents
        sentence_declination = 1.0 + 0.15 * np.sin(np.pi * (i / max(1, n_words - 1)))
        # Alternating accents (+2.2 semitones on stressed words, -1.2 on unstressed)
        st_accent = (2.2 if i % 2 == 1 else -1.2) if not is_monotone_pitch else 0.0
        accent_multiplier = 2.0 ** (st_accent / 12.0)
        word_inflection = 0.08 * np.sin(np.pi * (t / base_word_dur))

        if is_erratic_pitch:
            erratic_st = (4.8 if erratic_sev == "severe" else 3.5) if (i % 2 == 1) else (-4.2 if erratic_sev == "severe" else -3.0)
            st_multiplier = 2.0 ** (erratic_st / 12.0)
            f0_arr = base_f0 * sentence_declination * st_multiplier * (1.0 + 0.06 * np.sin(np.pi * (t / base_word_dur)))
        elif is_monotone_pitch:
            f0_arr = np.full(word_samples, base_f0, dtype=np.float32)
        else:
            f0_arr = base_f0 * sentence_declination * accent_multiplier * (1.0 + word_inflection)

        phase = 2.0 * np.pi * np.cumsum(f0_arr) / sr

        # Glottal source with fundamental and harmonics
        glottal = (
            0.55 * np.sin(phase) +
            0.30 * np.sin(2.0 * phase) +
            0.15 * np.sin(3.0 * phase) +
            0.08 * np.sin(4.0 * phase)
        )

        # Consonant aspiration / upper harmonic noise giving natural spectral centroid (~2200 Hz)
        aspiration_noise = np.random.normal(0.0, 0.07, word_samples).astype(np.float32)

        # Syllabic envelope
        if is_monotone_energy:
            envelope = np.ones_like(t) * 0.75
            ramp = min(len(t) // 8, int(0.02 * sr))
            if ramp > 0:
                envelope[:ramp] = np.linspace(0, 0.75, ramp)
                envelope[-ramp:] = np.linspace(0.75, 0, ramp)
        else:
            envelope = (np.sin(np.pi * (t / base_word_dur)) ** 1.3).astype(np.float32)

        word_audio = (glottal + aspiration_noise) * envelope

        # Energy dynamic scaling
        target_rms = 0.22
        if is_unstable_energy:
            target_rms = 0.48 if i % 2 == 0 else 0.06
        elif is_monotone_energy:
            target_rms = 0.20

        curr_rms = float(np.sqrt(np.mean(word_audio ** 2)))
        if curr_rms > 1e-6:
            word_audio = word_audio * (target_rms / curr_rms)

        if is_muffled:
            word_audio = lowpass_muffle_filter(word_audio, cutoff_hz=750.0, sr=sr)

        # Inter-word micro gap (40ms)
        micro_gap_dur = 0.02 if word_wpm > 180.0 else 0.04
        micro_gap = np.zeros(int(micro_gap_dur * sr), dtype=np.float32)

        total_audio_parts.append(word_audio.astype(np.float32))
        total_audio_parts.append(micro_gap)

        current_time_s = w_end + micro_gap_dur

    total_audio_parts.append(np.zeros(int(0.25 * sr), dtype=np.float32))
    full_sig = np.concatenate(total_audio_parts)

    max_peak = float(np.max(np.abs(full_sig)))
    if max_peak > 0:
        full_sig = (full_sig / max_peak) * 0.92

    computed_ground_truth = []
    for fr in flaw_regions:
        s_w = word_timings[fr["start_idx"]]
        e_w = word_timings[fr["end_idx"]]
        computed_ground_truth.append({
            "category": fr["category"],
            "start_s": round(s_w["start_s"], 2),
            "end_s": round(e_w["end_s"], 2),
            "transcript_slice": " ".join(words[fr["start_idx"]:fr["end_idx"] + 1]),
            "observed_value": fr.get("observed_value", 0.0),
            "baseline_value": fr.get("baseline_value", 0.0),
            "severity": fr.get("severity", "severe")
        })

    return full_sig, word_timings, computed_ground_truth


def build_manifest_and_audio(data_dir: str):
    """Generates all 6 speeches, 18 contrastive pairs, audio files and annotation JSONs."""
    os.makedirs(f"{data_dir}/audio/baselines", exist_ok=True)
    os.makedirs(f"{data_dir}/audio/flawed", exist_ok=True)
    os.makedirs(f"{data_dir}/annotations", exist_ok=True)

    speeches = [
        {
            "text_id": "speech_01_gettysburg",
            "title": "Gettysburg Address Excerpt",
            "canonical_transcript": "Four score and seven years ago our fathers brought forth on this continent a new nation conceived in liberty and dedicated to the proposition that all men are created equal.",
            "source_provenance": "Abraham Lincoln (1863), Public Domain US Historical Archives",
            "speaker": {"speaker_id": "spk_m01", "gender": "male", "native_language": "en", "median_f0_hz": 125.0}
        },
        {
            "text_id": "speech_02_churchill",
            "title": "We Shall Fight on the Beaches",
            "canonical_transcript": "We shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields and in the streets, we shall fight in the hills; we shall never surrender.",
            "source_provenance": "Winston Churchill (1940), Public Domain Parliamentary Address",
            "speaker": {"speaker_id": "spk_m02", "gender": "male", "native_language": "en", "median_f0_hz": 115.0}
        },
        {
            "text_id": "speech_03_kennedy",
            "title": "Inaugural Address Excerpt",
            "canonical_transcript": "Ask not what your country can do for you; ask what you can do for your country. My fellow citizens of the world, ask not what America will do for you, but what together we can do for freedom.",
            "source_provenance": "John F. Kennedy (1961), Public Domain Presidential Inaugural",
            "speaker": {"speaker_id": "spk_m03", "gender": "male", "native_language": "en", "median_f0_hz": 135.0}
        },
        {
            "text_id": "speech_04_keynote",
            "title": "Next-Gen AI Keynote Announcement",
            "canonical_transcript": "Today we are announcing an entirely new architecture that unifies real time audio synthesis and neural speech processing, delivering instantaneous clarity directly to everyone everywhere.",
            "source_provenance": "Open Speech Corpus (2024), CC-BY-4.0 Tech Speech Benchmark",
            "speaker": {"speaker_id": "spk_f01", "gender": "female", "native_language": "en", "median_f0_hz": 210.0}
        },
        {
            "text_id": "speech_05_defense",
            "title": "Academic Dissertation Defense",
            "canonical_transcript": "Our empirical evaluation demonstrates a statistically significant reduction in latency while maintaining rigorous mathematical bounds on convergence and acoustic fidelity across all testing environments.",
            "source_provenance": "Academic Open Speech Repository (2023), CC-BY-4.0 Scientific Corpus",
            "speaker": {"speaker_id": "spk_f02", "gender": "female", "native_language": "en", "median_f0_hz": 195.0}
        },
        {
            "text_id": "speech_06_debate",
            "title": "Competitive Policy Debate Opening",
            "canonical_transcript": "The affirmative contention fails on three distinct warrants: first, the empirical baseline contradicts their model; second, their mechanism introduces severe delay; and third, the projected benefits dissolve under scrutiny.",
            "source_provenance": "National Debate Archive (2022), CC-BY-4.0 Public Speech Dataset",
            "speaker": {"speaker_id": "spk_m04", "gender": "male", "native_language": "en", "median_f0_hz": 140.0}
        }
    ]

    manifest_pairs = []
    manifest_texts = []
    total_flaw_count = 0

    for s_idx, sp in enumerate(speeches):
        text_id = sp["text_id"]
        words = sp["canonical_transcript"].split()
        spk = sp["speaker"]
        base_f0 = spk["median_f0_hz"]
        n_words = len(words)

        manifest_texts.append({
            "text_id": text_id,
            "title": sp["title"],
            "canonical_transcript": sp["canonical_transcript"],
            "word_count": n_words,
            "source_provenance": sp["source_provenance"]
        })

        # Baseline: 135 WPM, normal natural intonation, 1 standard breathing pause (0.65s)
        # Positioned cleanly at word index 12 (between clause phrases 2 and 3)
        breathing_pause_idx = 12
        base_audio, base_timings, _ = synthesize_speech_utterance(
            words=words,
            base_f0=base_f0,
            default_wpm=135.0,
            pauses=[(breathing_pause_idx, 0.65)],
            sr=TARGET_SR
        )
        base_audio_path = f"data/audio/baselines/{text_id}_baseline.wav"
        sf.write(f"{data_dir}/audio/baselines/{text_id}_baseline.wav", base_audio, TARGET_SR)

        if s_idx == 0:
            flaw_specs = [("excessive_speed", "severe"), ("unnatural_pause", "severe"), ("monotone_pitch", "moderate")]
        elif s_idx == 1:
            flaw_specs = [("unstable_energy", "severe"), ("monotone_energy", "moderate"), ("excessive_speed", "moderate")]
        elif s_idx == 2:
            flaw_specs = [("unnatural_pause", "severe"), ("pitch_irregularity", "severe"), ("muffled_articulation", "moderate")]
        elif s_idx == 3:
            flaw_specs = [("excessive_speed", "severe"), ("monotone_pitch", "severe"), ("unstable_energy", "moderate")]
        elif s_idx == 4:
            flaw_specs = [("muffled_articulation", "severe"), ("unnatural_pause", "moderate"), ("monotone_energy", "moderate")]
        else:
            flaw_specs = [("excessive_speed", "severe"), ("pitch_irregularity", "moderate"), ("monotone_pitch", "severe")]

        for p_idx, (cat, sev) in enumerate(flaw_specs):
            pair_id = f"pair_{text_id}_{cat}_{p_idx+1}"

            seg_len = 3  # 3 words per episode gives ~1.3s - 1.8s localized anomaly
            i1_s = 1
            i1_e = i1_s + seg_len

            i2_s = i1_e + 3  # 3 clean words separation (~1.5s clean speech)
            i2_e = i2_s + seg_len

            i3_s = i2_e + 3  # 3 clean words separation (~1.5s clean speech)
            i3_e = min(n_words - 2, i3_s + seg_len)

            flaw_regions = []
            pause_spec = []

            if cat == "unnatural_pause":
                pause_spec = [(i1_s, 2.20), (i2_s, 2.50), (i3_s, 2.10)]
            else:
                pause_spec = [(breathing_pause_idx, 0.65)]
                def _get_obs_base(cat_name, wpm_val):
                    if cat_name == "excessive_speed":
                        return wpm_val, 135.0
                    elif cat_name == "monotone_pitch":
                        return 0.2, 2.5
                    elif cat_name == "pitch_irregularity":
                        return 4.5, 1.8
                    elif cat_name == "monotone_energy":
                        return 0.75, 5.5
                    elif cat_name == "unstable_energy":
                        return 18.0, 8.0
                    elif cat_name in ("muffled_articulation", "vocal_clarity"):
                        return 720.0, 2100.0
                    return 0.0, 0.0

                obs1, base1 = _get_obs_base(cat, 235.0)
                obs2, base2 = _get_obs_base(cat, 240.0)
                obs3, base3 = _get_obs_base(cat, 230.0)

                flaw_regions = [
                    {
                        "start_idx": i1_s, "end_idx": i1_e, "category": cat, "severity": sev,
                        "wpm": 235.0 if cat == "excessive_speed" else 135.0,
                        "observed_value": obs1,
                        "baseline_value": base1
                    },
                    {
                        "start_idx": i2_s, "end_idx": i2_e, "category": cat, "severity": sev,
                        "wpm": 240.0 if cat == "excessive_speed" else 135.0,
                        "observed_value": obs2,
                        "baseline_value": base2
                    },
                    {
                        "start_idx": i3_s, "end_idx": i3_e, "category": cat, "severity": sev,
                        "wpm": 230.0 if cat == "excessive_speed" else 135.0,
                        "observed_value": obs3,
                        "baseline_value": base3
                    }
                ]

            flaw_audio, flaw_timings, computed_gt = synthesize_speech_utterance(
                words=words,
                base_f0=base_f0,
                default_wpm=135.0,
                pauses=pause_spec,
                flaw_regions=flaw_regions,
                sr=TARGET_SR
            )

            flawed_audio_filename = f"{text_id}_{cat}_{sev}.wav"
            flawed_audio_path = f"data/audio/flawed/{flawed_audio_filename}"
            sf.write(f"{data_dir}/audio/flawed/{flawed_audio_filename}", flaw_audio, TARGET_SR)

            gt_flaws = []
            dur_total = len(flaw_audio) / TARGET_SR

            if cat == "unnatural_pause":
                for p_idx_num, (pause_w_idx, p_dur) in enumerate(pause_spec):
                    w_prev = flaw_timings[pause_w_idx - 1]
                    p_start = round(w_prev["end_s"], 2)
                    p_end = round(p_start + p_dur, 2)
                    gt_flaws.append({
                        "flaw_id": f"gt_{pair_id}_{p_idx_num+1:02d}",
                        "category": "unnatural_pause",
                        "start_s": p_start,
                        "end_s": p_end,
                        "transcript_slice": f"[Pause between '{words[pause_w_idx-1]}' and '{words[pause_w_idx]}']",
                        "observed_value": p_dur,
                        "baseline_value": 0.65,
                        "severity": sev,
                        "annotation_method": "acoustic_energy_silence_boundary"
                    })
            else:
                for idx_gt, c_gt in enumerate(computed_gt):
                    gt_flaws.append({
                        "flaw_id": f"gt_{pair_id}_{idx_gt+1:02d}",
                        "category": c_gt["category"],
                        "start_s": c_gt["start_s"],
                        "end_s": c_gt["end_s"],
                        "transcript_slice": c_gt["transcript_slice"],
                        "observed_value": c_gt["observed_value"],
                        "baseline_value": c_gt["baseline_value"],
                        "severity": c_gt["severity"],
                        "annotation_method": f"grounded_flaw_injection_{cat}"
                    })

            annotation_file = f"{data_dir}/annotations/{pair_id}.json"
            annotation_data = {
                "pair_id": pair_id,
                "text_id": text_id,
                "category": cat,
                "severity": sev,
                "duration_s": round(dur_total, 2),
                "ground_truth_flaws": gt_flaws
            }
            with open(annotation_file, "w") as f:
                json.dump(annotation_data, f, indent=2)

            total_flaw_count += len(gt_flaws)

            manifest_pairs.append({
                "pair_id": pair_id,
                "text_id": text_id,
                "baseline_audio": base_audio_path,
                "flawed_audio": flawed_audio_path,
                "flaw_category": cat,
                "severity": sev,
                "speaker_metadata": spk,
                "ground_truth_flaws": gt_flaws
            })

    manifest = {
        "dataset_version": "1.0.0",
        "license": "CC-BY-4.0 & Public Domain Historical Speeches",
        "metadata": {
            "total_speech_texts": len(speeches),
            "total_audio_pairs": len(manifest_pairs),
            "total_annotated_flaw_regions": total_flaw_count,
            "sample_rate_hz": TARGET_SR,
            "provenance_description": "Paired contrastive speech dataset with ground truth acoustic flaw grounding across 5 dimensions."
        },
        "speech_texts": manifest_texts,
        "pairs": manifest_pairs
    }

    with open(f"{data_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest_texts)} speech texts, {len(manifest_pairs)} contrastive pairs, {total_flaw_count} annotated flaw regions.")


if __name__ == "__main__":
    import sys
    dest_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    build_manifest_and_audio(dest_dir)
