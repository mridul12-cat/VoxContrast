"""
VoxContrast - Calibration Script on Training/Validation Data Only
Does NOT use or touch the held_out_test split during tuning.
Evaluates on:
1. Original frozen benchmark (data/manifest.json)
2. Validation split (data/tts_expanded/manifest.json, split='validation')
3. Clean baselines (original + validation)
"""
import sys
import copy
import os
import sys
import json
import copy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.audio_io import process_audio
from core.features import extract_acoustic_contours, summarize_features
from core.alignment import align_transcript_to_audio
import core.detector as detector
from tests.evaluate_temporal_iou import (
    match_ground_truth_and_predictions,
    compute_interval_iou,
    is_category_match
)

def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

orig_manifest = load_manifest("data/manifest.json")
tts_manifest = load_manifest("data/tts_expanded/manifest.json")

orig_texts = {st["text_id"]: st["canonical_transcript"] for st in orig_manifest.get("speech_texts", [])}
tts_texts = {st["text_id"]: st["canonical_transcript"] for st in tts_manifest.get("speech_texts", [])}

orig_pairs = orig_manifest.get("pairs", [])
val_pairs = [p for p in tts_manifest.get("pairs", []) if p.get("split") == "validation"]

orig_baselines = [
    {"audio_path": p["baseline_audio"], "canonical_transcript": orig_texts.get(p["text_id"], "")}
    for p in orig_pairs[:6]
]
val_baselines = [
    {"audio_path": p["baseline_audio"], "canonical_transcript": tts_texts.get(p["text_id"], "")}
    for p in val_pairs[:2]
]

print(f"Original pairs: {len(orig_pairs)} | Val pairs: {len(val_pairs)}")
print(f"Original baselines: {len(orig_baselines)} | Val baselines: {len(val_baselines)}")

# Pre-extract features to make parameter search blazing fast
def preextract(pairs_list, texts_dict):
    cached = []
    for pair in pairs_list:
        flaw_path = pair.get("flawed_audio")
        transcript = texts_dict.get(pair.get("text_id"), "")
        gt_flaws = pair.get("ground_truth_flaws", [])
        if not flaw_path or not os.path.exists(flaw_path):
            continue
        audio, _ = process_audio(flaw_path)
        ts, pauses = extract_acoustic_contours(audio)
        dur = len(audio) / 16000.0
        if transcript:
            align_st, ts = align_transcript_to_audio(transcript, ts, pauses, dur)
        else:
            from core.schemas import AlignmentStatus
            align_st = AlignmentStatus(status="FAILED", confidence=0.0, word_count=0, aligned_words=[])
        summ = summarize_features(ts, pauses)
        cached.append({
            "pair_id": pair.get("pair_id"),
            "audio": audio,
            "ts": ts,
            "pauses": pauses,
            "align_st": align_st,
            "summ": summ,
            "gt_flaws": gt_flaws,
            "dur": dur
        })
    return cached

def preextract_baselines(baselines_list):
    cached = []
    for b in baselines_list:
        b_path = b.get("audio_path")
        transcript = b.get("canonical_transcript", "")
        if not b_path or not os.path.exists(b_path):
            continue
        audio, _ = process_audio(b_path)
        ts, pauses = extract_acoustic_contours(audio)
        dur = len(audio) / 16000.0
        if transcript:
            align_st, ts = align_transcript_to_audio(transcript, ts, pauses, dur)
        else:
            from core.schemas import AlignmentStatus
            align_st = AlignmentStatus(status="FAILED", confidence=0.0, word_count=0, aligned_words=[])
        summ = summarize_features(ts, pauses)
        cached.append({
            "audio": audio,
            "ts": ts,
            "pauses": pauses,
            "align_st": align_st,
            "summ": summ,
            "dur": dur
        })
    return cached

print("Pre-extracting features for fast calibration...")
orig_cached = preextract(orig_pairs, orig_texts)
val_cached = preextract(val_pairs, tts_texts)

orig_base_cached = preextract_baselines(orig_baselines)
val_base_cached = preextract_baselines(val_baselines)
print("Pre-extraction complete.")

def evaluate_cached(cached_pairs, cached_baselines):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ious = []
    for item in cached_pairs:
        flaws = detector.detect_temporal_flaws(
            item["ts"], item["pauses"], item["align_st"], item["summ"]
        )
        gt_list = [f for f in item["gt_flaws"] if f.get("validation_status", "CONFIRMED") != "UNCERTAIN"]
        res = match_ground_truth_and_predictions(gt_list, flaws)
        total_tp += res["true_positives"]
        total_fp += res["false_positives"]
        total_fn += res["false_negatives"]
        ious.extend(res["gt_ious"])

    macro_iou = sum(ious) / len(ious) if ious else 0.0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Baseline FP count
    base_fps = 0
    base_dur = sum(b["dur"] for b in cached_baselines)
    for b in cached_baselines:
        det = detector.detect_temporal_flaws(b["ts"], b["pauses"], b["align_st"], b["summ"])
        base_fps += len(det)
    base_fp_rate = (base_fps / (base_dur / 60.0)) if base_dur > 0 else 0.0

    return {
        "macro_iou": macro_iou,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "base_fps": base_fps,
        "base_fp_rate": base_fp_rate
    }

print("\n--- BASELINE METRICS (Current Detector) ---")
m_orig = evaluate_cached(orig_cached, orig_base_cached)
print(f"Original Benchmark: IoU={m_orig['macro_iou']:.4f}, Prec={m_orig['prec']:.4f}, Rec={m_orig['rec']:.4f}, F1={m_orig['f1']:.4f}, FP={m_orig['fp']}, Base_FP={m_orig['base_fps']}")

m_val = evaluate_cached(val_cached, val_base_cached)
print(f"Validation Split  : IoU={m_val['macro_iou']:.4f}, Prec={m_val['prec']:.4f}, Rec={m_val['rec']:.4f}, F1={m_val['f1']:.4f}, FP={m_val['fp']}, Base_FP={m_val['base_fps']}")

print("\n--- RUNNING PARAMETER SWEEP ON TRAIN/VAL (NO HELD-OUT TEST) ---")
best_score = -1.0
best_params = None
results = []

# Sweep excessive_speed thresholds and duration filters
speed_wpm_opts = [180.0, 190.0, 200.0, 210.0]
speed_dur_opts = [0.40, 0.60, 0.80, 1.00]
muffled_cent_opts = [360.0, 380.0, 400.0, 420.0]
muffled_dur_opts = [0.40, 0.55, 0.70]
pitch_mono_opts = [0.45, 0.50, 0.55, 0.60]
energy_mono_opts = [1.20, 1.40, 1.60, 1.85]

# Step 1: Optimize excessive speed parameters while holding others constant
print("Step 1: Calibrating Excessive Speed parameters...")
for wpm in speed_wpm_opts:
    for dur in speed_dur_opts:
        detector.THRESH_FAST_WPM = wpm
        detector.MIN_DUR_FAST_S = dur
        val_res = evaluate_cached(val_cached, val_base_cached)
        orig_res = evaluate_cached(orig_cached, orig_base_cached)
        results.append(("speed", wpm, dur, orig_res, val_res))
        print(f"  [Speed] wpm={wpm}, dur={dur}s -> Orig: IoU={orig_res['macro_iou']:.4f}, F1={orig_res['f1']:.4f}, FP={orig_res['fp']} | Val: IoU={val_res['macro_iou']:.4f}, F1={val_res['f1']:.4f}, FP={val_res['fp']}, BaseFP={val_res['base_fps']}")

# Pick best speed param (preserves orig F1 >= 0.90, minimizes val BaseFP + FP while keeping Rec >= 0.80)
valid_speed = [r for r in results if r[3]['f1'] >= 0.88 and r[4]['rec'] >= 0.80]
if valid_speed:
    # Sort by Val F1 descending, then Val BaseFP ascending
    valid_speed.sort(key=lambda r: (r[4]['f1'], -r[4]['base_fps']), reverse=True)
    best_speed = valid_speed[0]
    print(f"\n=> Best Speed: wpm={best_speed[1]}, dur={best_speed[2]}s (Val F1={best_speed[4]['f1']:.4f}, BaseFP={best_speed[4]['base_fps']})")
    detector.THRESH_FAST_WPM = best_speed[1]
    detector.MIN_DUR_FAST_S = best_speed[2]

# Step 2: Calibrate Muffled Articulation parameters
print("\nStep 2: Calibrating Muffled Articulation parameters...")
muff_results = []
for cent in muffled_cent_opts:
    for dur in muffled_dur_opts:
        detector.THRESH_MUFFLED_CENTROID_HZ = cent
        detector.MIN_DUR_MUFFLED_S = dur
        val_res = evaluate_cached(val_cached, val_base_cached)
        orig_res = evaluate_cached(orig_cached, orig_base_cached)
        muff_results.append((cent, dur, orig_res, val_res))
        print(f"  [Muffled] cent={cent}Hz, dur={dur}s -> Orig: IoU={orig_res['macro_iou']:.4f}, F1={orig_res['f1']:.4f} | Val: F1={val_res['f1']:.4f}, BaseFP={val_res['base_fps']}")

valid_muff = [r for r in muff_results if r[2]['f1'] >= 0.88]
if valid_muff:
    valid_muff.sort(key=lambda r: (r[3]['f1'], -r[3]['base_fps']), reverse=True)
    best_muff = valid_muff[0]
    print(f"\n=> Best Muffled: cent={best_muff[0]}Hz, dur={best_muff[1]}s (Val F1={best_muff[3]['f1']:.4f}, BaseFP={best_muff[3]['base_fps']})")
    detector.THRESH_MUFFLED_CENTROID_HZ = best_muff[0]
    detector.MIN_DUR_MUFFLED_S = best_muff[1]

# Step 3: Calibrate Monotone Pitch & Energy parameters
print("\nStep 3: Calibrating Monotone Pitch & Energy parameters...")
pitch_energy_res = []
for p_th in pitch_mono_opts:
    for e_th in energy_mono_opts:
        detector.THRESH_MONOTONE_PITCH_ST = p_th
        detector.THRESH_MONOTONE_ENERGY_STD_DB = e_th
        val_res = evaluate_cached(val_cached, val_base_cached)
        orig_res = evaluate_cached(orig_cached, orig_base_cached)
        pitch_energy_res.append((p_th, e_th, orig_res, val_res))
        print(f"  [Pitch/Energy] pitch_th={p_th}ST, energy_th={e_th}dB -> Orig: IoU={orig_res['macro_iou']:.4f}, F1={orig_res['f1']:.4f} | Val: F1={val_res['f1']:.4f}, BaseFP={val_res['base_fps']}")

valid_pe = [r for r in pitch_energy_res if r[2]['f1'] >= 0.88 and r[3]['rec'] >= 0.80]
if valid_pe:
    valid_pe.sort(key=lambda r: (r[3]['f1'], -r[3]['base_fps']), reverse=True)
    best_pe = valid_pe[0]
    print(f"\n=> Best Pitch/Energy: pitch_th={best_pe[0]}ST, energy_th={best_pe[1]}dB (Val F1={best_pe[3]['f1']:.4f}, BaseFP={best_pe[3]['base_fps']})")
    detector.THRESH_MONOTONE_PITCH_ST = best_pe[0]
    detector.THRESH_MONOTONE_ENERGY_STD_DB = best_pe[1]

print("\n--- FINAL CALIBRATED BENCHMARK ON TRAIN/VAL ---")
final_orig = evaluate_cached(orig_cached, orig_base_cached)
final_val = evaluate_cached(val_cached, val_base_cached)
print(f"Calibrated Original Benchmark: IoU={final_orig['macro_iou']:.4f}, Prec={final_orig['prec']:.4f}, Rec={final_orig['rec']:.4f}, F1={final_orig['f1']:.4f}, BaseFP={final_orig['base_fps']}")
print(f"Calibrated Validation Split  : IoU={final_val['macro_iou']:.4f}, Prec={final_val['prec']:.4f}, Rec={final_val['rec']:.4f}, F1={final_val['f1']:.4f}, BaseFP={final_val['base_fps']}")

