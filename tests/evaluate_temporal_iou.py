"""
VoxContrast - Temporal Flaw Grounding IoU Benchmark
Evaluates temporal interval localization accuracy against ground-truth annotations in data/manifest.json.
Calculates Macro IoU, Precision@0.5, Recall@0.5, and Mean Onset/Offset errors.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on Python module search path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import numpy as np

from core.analyzer import analyze_speech
from backend.config import MANIFEST_PATH, DATA_DIR


def compute_interval_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculates temporal Intersection over Union between two intervals."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_len = max(0.0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_len = max(1e-6, union_end - union_start)

    return float(intersection_len / union_len)


def run_temporal_grounding_benchmark() -> dict:
    """Executes benchmark over all 18 contrastive pairs and 54 annotations."""
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    speech_texts = {s["text_id"]: s["canonical_transcript"] for s in manifest["speech_texts"]}

    category_stats: dict[str, dict] = {}
    total_ious = []
    total_onset_errs = []
    total_offset_errs = []

    true_positives_05 = 0
    false_positives_05 = 0
    false_negatives_05 = 0

    base_dir = os.path.dirname(DATA_DIR)

    for pair in manifest["pairs"]:
        pair_id = pair["pair_id"]
        cat = pair["flaw_category"]
        text_id = pair["text_id"]
        transcript = speech_texts.get(text_id, "")
        flawed_audio_path = os.path.join(base_dir, pair["flawed_audio"])
        gt_flaws = pair["ground_truth_flaws"]

        if not os.path.exists(flawed_audio_path):
            continue

        analysis = analyze_speech(flawed_audio_path, transcript)
        detected = analysis.detected_flaws

        if cat not in category_stats:
            category_stats[cat] = {
                "ious": [],
                "onset_errs": [],
                "offset_errs": [],
                "gt_count": 0,
                "detected_count": 0
            }

        category_stats[cat]["gt_count"] += len(gt_flaws)
        category_stats[cat]["detected_count"] += len(detected)

        # Match detected flaws with ground truth
        matched_gt = set()
        matched_det = set()

        for gt_idx, gt in enumerate(gt_flaws):
            best_iou = 0.0
            best_det_idx = None
            best_onset_err = 0.0
            best_offset_err = 0.0

            for d_idx, d in enumerate(detected):
                # Category match
                is_cat_match = (
                    d.category == gt["category"] or
                    {d.category, gt["category"]} <= {"muffled_articulation", "vocal_clarity"}
                )
                if not is_cat_match:
                    continue

                # Calculate IoU
                iou = compute_interval_iou(d.start_s, d.end_s, gt["start_s"], gt["end_s"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx
                    best_onset_err = abs(d.start_s - gt["start_s"])
                    best_offset_err = abs(d.end_s - gt["end_s"])

            total_ious.append(best_iou)
            category_stats[cat]["ious"].append(best_iou)
            if best_iou > 0.0:
                total_onset_errs.append(best_onset_err)
                total_offset_errs.append(best_offset_err)
                category_stats[cat]["onset_errs"].append(best_onset_err)
                category_stats[cat]["offset_errs"].append(best_offset_err)

            if best_iou >= 0.50:
                true_positives_05 += 1
                matched_gt.add(gt_idx)
                if best_det_idx is not None:
                    matched_det.add(best_det_idx)
            else:
                false_negatives_05 += 1

        # Any detected flaw not matched with IoU >= 0.5 is a false positive
        false_positives_05 += max(0, len(detected) - len(matched_det))

    macro_iou = float(np.mean(total_ious)) if total_ious else 0.0
    mean_onset_err = float(np.mean(total_onset_errs)) if total_onset_errs else 0.0
    mean_offset_err = float(np.mean(total_offset_errs)) if total_offset_errs else 0.0

    precision_05 = (
        true_positives_05 / (true_positives_05 + false_positives_05)
        if (true_positives_05 + false_positives_05) > 0
        else 0.0
    )
    recall_05 = (
        true_positives_05 / (true_positives_05 + false_negatives_05)
        if (true_positives_05 + false_negatives_05) > 0
        else 0.0
    )
    f1_05 = (
        (2 * precision_05 * recall_05) / (precision_05 + recall_05)
        if (precision_05 + recall_05) > 0
        else 0.0
    )

    results = {
        "macro_iou": round(macro_iou, 4),
        "mean_onset_error_s": round(mean_onset_err, 4),
        "mean_offset_error_s": round(mean_offset_err, 4),
        "precision_at_05_iou": round(precision_05, 4),
        "recall_at_05_iou": round(recall_05, 4),
        "f1_at_05_iou": round(f1_05, 4),
        "total_ground_truth_flaws": sum(s["gt_count"] for s in category_stats.values()),
        "category_breakdown": {}
    }

    for cat, data in category_stats.items():
        results["category_breakdown"][cat] = {
            "mean_iou": round(float(np.mean(data["ious"])), 4) if data["ious"] else 0.0,
            "mean_onset_err_s": round(float(np.mean(data["onset_errs"])), 4) if data["onset_errs"] else 0.0,
            "mean_offset_err_s": round(float(np.mean(data["offset_errs"])), 4) if data["offset_errs"] else 0.0,
            "gt_count": data["gt_count"]
        }

    return results


def test_temporal_grounding_acceptance_gate():
    """Pytest test asserting that Macro IoU and error thresholds meet acceptance gate."""
    benchmark = run_temporal_grounding_benchmark()
    # Acceptance gate: Macro IoU >= 0.65; Onset/Offset error < 0.35s (350ms)
    assert benchmark["macro_iou"] >= 0.65, f"Macro IoU {benchmark['macro_iou']} below 0.65 gate"
    assert benchmark["mean_onset_error_s"] <= 0.35, f"Onset error {benchmark['mean_onset_error_s']}s exceeds 350ms gate"
    assert benchmark["mean_offset_error_s"] <= 0.35, f"Offset error {benchmark['mean_offset_error_s']}s exceeds 350ms gate"


if __name__ == "__main__":
    res = run_temporal_grounding_benchmark()
    print("=" * 60)
    print("VoxContrast Temporal Grounding Benchmark Results")
    print("=" * 60)
    print(f"Macro IoU:              {res['macro_iou']:.4f}")
    print(f"Mean Onset Error:       {res['mean_onset_error_s']*1000:.1f} ms")
    print(f"Mean Offset Error:      {res['mean_offset_error_s']*1000:.1f} ms")
    print(f"Precision @ IoU >= 0.5: {res['precision_at_05_iou']:.4f}")
    print(f"Recall @ IoU >= 0.5:    {res['recall_at_05_iou']:.4f}")
    print(f"F1 @ IoU >= 0.5:        {res['f1_at_05_iou']:.4f}")
    print(f"Total Evaluated Flaws:  {res['total_ground_truth_flaws']}")
    print("-" * 60)
    print("Category Breakdown:")
    for cat, b in res["category_breakdown"].items():
        print(f"  - {cat:22s}: IoU={b['mean_iou']:.4f}, OnsetErr={b['mean_onset_err_s']*1000:.1f}ms, Count={b['gt_count']}")
    print("=" * 60)
