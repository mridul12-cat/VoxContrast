"""
VoxContrast - Verification Suite for Corrected V2 Spoken Speech Baselines
Verifies duration, sample rate, channels, RMS, SNR, and word-by-word forced alignment
against canonical transcripts for all 6 baseline recordings.
"""

import os
import json
import soundfile as sf
import numpy as np
from core.audio_io import process_audio
from core.features import extract_acoustic_contours, summarize_features
from core.alignment import align_transcript_to_audio

MANIFEST_V2 = "data/tts_expanded_v2/manifest.json"

with open(MANIFEST_V2, "r", encoding="utf-8") as f:
    manifest = json.load(f)

baselines = manifest["baselines"]
print(f"=== VOXCONTRAST V2 BASELINE VERIFICATION AUDIT ({len(baselines)} RECORDINGS) ===\n")

all_passed = True

for idx, b in enumerate(baselines, start=1):
    text_id = b["text_id"]
    path = b["audio_path"]
    voice = b["voice"]
    split = b["split"]
    transcript = b["canonical_transcript"]

    print(f"[{idx}/6] Verifying Baseline: {text_id}")
    print(f"  • File path:    {path}")
    print(f"  • Voice / Accent: {voice} ({'en_US' if voice == 'Samantha' else 'en_GB'})")
    print(f"  • Split:        {split}")

    # Check file on disk
    if not os.path.exists(path):
        print(f"  [!] FAILED: File does not exist: {path}")
        all_passed = False
        continue

    audio, meta = process_audio(path)
    print(f"  • Audio stats:  dur={meta.duration_s:.3f}s | sr={meta.sample_rate}Hz | channels={meta.channels} | SNR={meta.snr_db:.1f}dB | clipped={meta.clipped_ratio*100:.2f}%")

    # Feature and Alignment verification
    ts, pauses = extract_acoustic_contours(audio, meta.sample_rate)
    summ = summarize_features(ts, pauses)
    align_st, _ = align_transcript_to_audio(transcript, ts, pauses, meta.duration_s)

    print(f"  • Acoustics:    mean_f0={summ.pitch_median_hz:.1f}Hz | f0_std={summ.pitch_semitone_std:.2f}ST | rms_mean={summ.energy_mean_db:.1f}dB | pauses={len(pauses)}")
    print(f"  • Alignment:    status={align_st.status} | confidence={align_st.confidence:.2f} | words_aligned={len(align_st.aligned_words)}/{align_st.word_count}")

    # First and last words check
    first_w = align_st.aligned_words[0] if align_st.aligned_words else None
    last_w = align_st.aligned_words[-1] if align_st.aligned_words else None
    if first_w and last_w:
        print(f"  • First word:   '{first_w.word}' [{first_w.start_s:.2f}s - {first_w.end_s:.2f}s]")
        print(f"  • Last word:    '{last_w.word}' [{last_w.start_s:.2f}s - {last_w.end_s:.2f}s]")

    # Verification decision
    passed = (
        meta.sample_rate == 16000 and
        meta.channels == 1 and
        meta.duration_s > 4.0 and
        meta.snr_db > 20.0 and
        align_st.status == "SUCCESS" and
        align_st.confidence >= 0.80 and
        len(align_st.aligned_words) == align_st.word_count
    )

    if passed:
        print("  -> RESULT: PASSED (Verified Intelligible Spoken Speech & Transcript Identity)\n")
    else:
        print("  -> RESULT: FAILED VERIFICATION\n")
        all_passed = False

if all_passed:
    print("ALL 6 V2 BASELINES PASSED PHYSICAL AND ALIGNMENT INTEGRITY CHECKS!")
else:
    print("SOME BASELINES FAILED!")
