import json
import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.analyzer import analyze_speech
from tests.evaluate_temporal_iou import compute_interval_iou

def main():
    with open("data/manifest.json") as f: m_frozen = json.load(f)
    with open("data/tts_expanded/manifest.json") as f: m_tts = json.load(f)

    texts_frozen = {s["text_id"]: s["canonical_transcript"] for s in m_frozen["speech_texts"]}
    texts_tts = {s["text_id"]: s["canonical_transcript"] for s in m_tts["speech_texts"]}

    print("Pre-analyzing...")
    frozen_res = []
    for p in m_frozen["pairs"]:
        res = analyze_speech(p["flawed_audio"], texts_frozen[p["text_id"]])
        frozen_res.append((p, res))

    val_res = []
    for p in m_tts["pairs"]:
        if p.get("split") == "validation":
            gts = [g for g in p["ground_truth_flaws"] if g.get("validation_status", "CONFIRMED") == "CONFIRMED"]
            if gts:
                res = analyze_speech(p["flawed_audio"], texts_tts[p["text_id"]])
                val_res.append((p, res, gts))

    tts_baselines = [
        ("data/tts_expanded/audio/baselines/speech_07_roosevelt_baseline.wav", texts_tts["speech_07_roosevelt"]),
        ("data/tts_expanded/audio/baselines/speech_08_curie_baseline.wav", texts_tts["speech_08_curie"])
    ]
    val_b_res = [analyze_speech(b[0], b[1]) for b in tts_baselines]

    print(f"Loaded {len(frozen_res)} frozen, {len(val_res)} val pairs, {len(val_b_res)} val baselines.")

    # 1. Test muffled threshold
    print("\n--- MUFFLED ARTICULATION CALIBRATION ---")
    for thresh in [550.0, 500.0, 460.0, 440.0, 420.0]:
        for min_d in [0.40, 0.50, 0.60]:
            val_muff = 0
            for p, res, gts in val_res:
                ts = res.time_series
                dt = ts.time_step_s
                frames = [(ts.time_stamps[i], ts.time_stamps[i] + dt)
                          for i in range(len(ts.time_stamps))
                          if ts.f0_contour[i] > 0 and ts.rms_energy_db[i] > -32.0 and ts.spectral_centroid_hz[i] < thresh]
                merged = []
                for s, e in frames:
                    if not merged: merged.append([s, e])
                    elif s - merged[-1][1] <= 0.35: merged[-1][1] = max(merged[-1][1], e)
                    else: merged.append([s, e])
                filt = [m for m in merged if (m[1] - m[0]) >= min_d]
                val_muff += len(filt)

            # Check frozen muffled IoU
            f_ious = []
            for p, res in frozen_res:
                if p["flaw_category"] == "muffled_articulation":
                    ts = res.time_series
                    dt = ts.time_step_s
                    frames = [(ts.time_stamps[i], ts.time_stamps[i] + dt)
                              for i in range(len(ts.time_stamps))
                              if ts.f0_contour[i] > 0 and ts.rms_energy_db[i] > -32.0 and ts.spectral_centroid_hz[i] < thresh]
                    merged = []
                    for s, e in frames:
                        if not merged: merged.append([s, e])
                        elif s - merged[-1][1] <= 0.35: merged[-1][1] = max(merged[-1][1], e)
                        else: merged.append([s, e])
                    filt = [m for m in merged if (m[1] - m[0]) >= min_d]
                    for g in p["ground_truth_flaws"]:
                        best_iou = max([compute_interval_iou(m[0], m[1], g["start_s"], g["end_s"]) for m in filt] or [0.0])
                        f_ious.append(best_iou)

            print(f"thresh={thresh:.0f}Hz, min_dur={min_d:.2f}s -> Val Muffled Detections={val_muff:2d}, Frozen Muffled IoU={np.mean(f_ious):.4f}")

if __name__ == "__main__":
    main()
