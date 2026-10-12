import json
import os
import soundfile as sf
import numpy as np

import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.analyzer import analyze_speech
from tests.evaluate_temporal_iou import compute_interval_iou

def diagnose():
    with open("data/tts_expanded/manifest.json") as f:
        m = json.load(f)

    speech_texts = {s["text_id"]: s["canonical_transcript"] for s in m.get("speech_texts", [])}
    cat_counts = {}

    print("==================================================================")
    print("TTS EXPANDED RECORDING AUDIT & FALSE POSITIVE BREAKDOWN")
    print("==================================================================")

    for p in m["pairs"]:
        pid = p["pair_id"]
        pcat = p["flaw_category"]
        psplit = p.get("split")
        flawed_path = p["flawed_audio"]
        txt = speech_texts[p["text_id"]]

        # Audio file stats
        data, sr = sf.read(flawed_path)
        total_dur = len(data) / sr

        res = analyze_speech(flawed_path, txt)
        dets = res.detected_flaws
        gts = p.get("ground_truth_flaws", [])
        confirmed_gts = [g for g in gts if g.get("validation_status", "CONFIRMED") == "CONFIRMED"]

        # Calculate speech-active duration from time series
        ts = res.time_series
        active_frames = sum(1 for i in range(len(ts.time_stamps)) if ts.f0_contour[i] > 0 or ts.rms_energy_db[i] > -38.0)
        active_dur = active_frames * ts.time_step_s

        print(f"\n--- PAIR: {pid} (split={psplit}, expected_flaw={pcat}) ---")
        print(f"File: {flawed_path} | Total dur: {total_dur:.2f}s | Active speech: {active_dur:.2f}s")
        print(f"Ground Truth ({len(gts)} total, {len(confirmed_gts)} confirmed):")
        for g in gts:
            status = g.get("validation_status", "CONFIRMED")
            print(f"  GT: [{g['category']}] {g['start_s']:.2f}s - {g['end_s']:.2f}s ({status}) | val={g.get('observed_value')}")

        print(f"Detections ({len(dets)}):")
        for d in dets:
            cat_counts[d.category] = cat_counts.get(d.category, 0) + 1
            # Check if matches any GT
            best_iou = 0.0
            matched_g_info = "NO MATCH (FP)"
            for g in confirmed_gts:
                is_cat = (d.category == g["category"] or
                          {d.category, g["category"]} <= {"muffled_articulation", "vocal_clarity"} or
                          {d.category, g["category"]} <= {"slow_speech", "slow_delivery"})
                if is_cat:
                    iou = compute_interval_iou(d.start_s, d.end_s, g["start_s"], g["end_s"])
                    if iou > best_iou:
                        best_iou = iou
            if best_iou >= 0.50:
                matched_g_info = f"MATCHED GT (IoU={best_iou:.3f}, TP)"
            elif best_iou > 0.0:
                matched_g_info = f"PARTIAL OVERLAP (IoU={best_iou:.3f}, FP)"

            print(f"  DET: [{d.category:18s}] {d.start_s:5.2f}s - {d.end_s:5.2f}s (dur={d.end_s-d.start_s:.2f}s, obs={d.observed_value:5.1f} {d.unit:4s}, th={d.threshold_value:5.1f}) -> {matched_g_info}")
            print(f"       words: \"{d.transcript_slice[:50]}\"")

    print("\n==================================================================")
    print("TOTAL DETECTIONS ACROSS ALL 12 FLAWED TTS RECORDINGS:")
    print("==================================================================")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:22s}: {cnt:3d}")

    # Also check baselines in TTS expanded!
    print("\n==================================================================")
    print("CLEAN BASELINE TTS RECORDINGS FALSE POSITIVE AUDIT:")
    print("==================================================================")
    baseline_cat_counts = {}
    total_baseline_dur = 0.0
    total_baseline_active = 0.0
    for s in m.get("speech_texts", []):
        tid = s["text_id"]
        bpath = f"data/tts_expanded/audio/baselines/{tid}_baseline.wav"
        if not os.path.exists(bpath):
            continue
        data, sr = sf.read(bpath)
        dur = len(data) / sr
        total_baseline_dur += dur
        txt = s["canonical_transcript"]
        res = analyze_speech(bpath, txt)
        ts = res.time_series
        active_f = sum(1 for i in range(len(ts.time_stamps)) if ts.f0_contour[i] > 0 or ts.rms_energy_db[i] > -38.0)
        active_d = active_f * ts.time_step_s
        total_baseline_active += active_d

        print(f"\nBaseline: {tid} ({bpath}) | Dur: {dur:.2f}s | Active: {active_d:.2f}s")
        print(f"Detections ({len(res.detected_flaws)}):")
        for d in res.detected_flaws:
            baseline_cat_counts[d.category] = baseline_cat_counts.get(d.category, 0) + 1
            print(f"  DET: [{d.category:18s}] {d.start_s:5.2f}s - {d.end_s:5.2f}s (dur={d.end_s-d.start_s:.2f}s, obs={d.observed_value:5.1f} {d.unit:4s}, th={d.threshold_value:5.1f})")

    print("\n--- BASELINE FP SUMMARY ---")
    total_fps = sum(baseline_cat_counts.values())
    fps_per_min = (total_fps / (total_baseline_dur / 60.0)) if total_baseline_dur > 0 else 0.0
    fps_per_active_min = (total_fps / (total_baseline_active / 60.0)) if total_baseline_active > 0 else 0.0
    print(f"Total baseline clean audio duration: {total_baseline_dur:.2f}s ({total_baseline_dur/60.0:.2f} min)")
    print(f"Total baseline clean speech active duration: {total_baseline_active:.2f}s ({total_baseline_active/60.0:.2f} min)")
    print(f"Total False Positives on Clean Baselines: {total_fps}")
    print(f"FP rate per minute (total duration): {fps_per_min:.2f} FP/min")
    print(f"FP rate per minute (speech active duration): {fps_per_active_min:.2f} FP/min")
    for cat, cnt in sorted(baseline_cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:22s}: {cnt:3d}")

if __name__ == "__main__":
    diagnose()
