"""
VoxContrast - Temporal Flaw Grounding IoU Benchmark
Evaluates temporal interval localization accuracy against ground-truth annotations.
Supports original frozen benchmark (data/manifest.json) and TTS-assisted expansion splits
(validation, held_out_test in data/tts_expanded/manifest.json).
Calculates Macro IoU, Precision@0.5, Recall@0.5, and Mean Onset/Offset errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure repository root is on Python module search path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from core.analyzer import analyze_speech
from backend.config import MANIFEST_PATH, DATA_DIR

TTS_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "tts_expanded", "manifest.json")


def compute_interval_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculates temporal Intersection over Union between two intervals."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_len = max(0.0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_len = max(1e-6, union_end - union_start)

    return float(intersection_len / union_len)


def run_temporal_grounding_benchmark(
    manifest_path: str = MANIFEST_PATH,
    split: str | None = None,
    confirmed_only: bool = True
) -> dict:
    """
    Executes benchmark over contrastive pairs in the given manifest and optional split.
    
    Parameters:
    - manifest_path: Path to dataset manifest JSON. Defaults to data/manifest.json.
    - split: Specific split ('validation', 'held_out_test') or None for all pairs.
    - confirmed_only: If True, evaluates only flaws marked CONFIRMED (or with no status).
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    speech_texts = {s["text_id"]: s["canonical_transcript"] for s in manifest.get("speech_texts", [])}

    category_stats: dict[str, dict] = {}
    total_ious = []
    total_onset_errs = []
    total_offset_errs = []

    true_positives_05 = 0
    false_positives_05 = 0
    false_negatives_05 = 0

    valid_pair_ids = None
    if split is not None and "splits" in manifest and split in manifest["splits"]:
        valid_pair_ids = set(manifest["splits"][split])

    for pair in manifest.get("pairs", []):
        pair_id = pair["pair_id"]
        pair_split = pair.get("split")

        if split is not None:
            if valid_pair_ids is not None and pair_id not in valid_pair_ids:
                continue
            elif valid_pair_ids is None and pair_split != split:
                continue

        cat = pair["flaw_category"]
        text_id = pair["text_id"]
        transcript = speech_texts.get(text_id, "")

        # Resolve audio path relative to REPO_ROOT or manifest dir
        flawed_rel = pair["flawed_audio"]
        if os.path.isabs(flawed_rel):
            flawed_audio_path = flawed_rel
        elif os.path.exists(os.path.join(REPO_ROOT, flawed_rel)):
            flawed_audio_path = os.path.join(REPO_ROOT, flawed_rel)
        elif os.path.exists(os.path.join(os.path.dirname(manifest_path), flawed_rel)):
            flawed_audio_path = os.path.join(os.path.dirname(manifest_path), flawed_rel)
        else:
            flawed_audio_path = os.path.join(REPO_ROOT, flawed_rel)

        all_gt_flaws = pair.get("ground_truth_flaws", [])
        if confirmed_only:
            gt_flaws = [g for g in all_gt_flaws if g.get("validation_status", "CONFIRMED") == "CONFIRMED"]
        else:
            gt_flaws = all_gt_flaws

        if not gt_flaws:
            continue

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
                    {d.category, gt["category"]} <= {"muffled_articulation", "vocal_clarity"} or
                    {d.category, gt["category"]} <= {"slow_speech", "slow_delivery"}
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

    split_label = split if split else ("all_pairs" if "splits" in manifest else "original_benchmark")
    results = {
        "split": split_label,
        "manifest_path": manifest_path,
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


def print_benchmark_report(res: dict, title: str | None = None) -> None:
    """Prints a formatted evaluation report for a benchmark result."""
    heading = title or f"VoxContrast Temporal Grounding Results [{res.get('split', 'default')}]"
    print("=" * 66)
    print(heading)
    print("=" * 66)
    print(f"Manifest:               {res.get('manifest_path', 'unknown')}")
    print(f"Split:                  {res.get('split', 'all')}")
    print(f"Macro IoU:              {res['macro_iou']:.4f}")
    print(f"Mean Onset Error:       {res['mean_onset_error_s']*1000:.1f} ms")
    print(f"Mean Offset Error:      {res['mean_offset_error_s']*1000:.1f} ms")
    print(f"Precision @ IoU >= 0.5: {res['precision_at_05_iou']:.4f}")
    print(f"Recall @ IoU >= 0.5:    {res['recall_at_05_iou']:.4f}")
    print(f"F1 @ IoU >= 0.5:        {res['f1_at_05_iou']:.4f}")
    print(f"Total Evaluated Flaws:  {res['total_ground_truth_flaws']}")
    print("-" * 66)
    print("Category Breakdown:")
    for cat, b in res["category_breakdown"].items():
        print(f"  - {cat:22s}: IoU={b['mean_iou']:.4f}, OnsetErr={b['mean_onset_err_s']*1000:.1f}ms, Count={b['gt_count']}")
    print("=" * 66)
    print()


def print_comparative_table(results_list: list[dict]) -> None:
    """Prints a comparative markdown table across multiple evaluated splits."""
    print("### Multi-Split Temporal Grounding Benchmark Comparison")
    print()
    print("| Split | Flaws | Macro IoU | Onset Error | Offset Error | Prec@0.5 | Rec@0.5 | F1@0.5 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results_list:
        split_name = r.get("split", "unknown")
        cnt = r["total_ground_truth_flaws"]
        iou = f"{r['macro_iou']:.4f}"
        onset = f"{r['mean_onset_error_s']*1000:.1f} ms"
        offset = f"{r['mean_offset_error_s']*1000:.1f} ms"
        p = f"{r['precision_at_05_iou']:.4f}"
        rec = f"{r['recall_at_05_iou']:.4f}"
        f1 = f"{r['f1_at_05_iou']:.4f}"
        print(f"| **{split_name}** | {cnt} | **{iou}** | {onset} | {offset} | {p} | {rec} | {f1} |")
    print()


def test_temporal_grounding_acceptance_gate():
    """Pytest test asserting that Macro IoU and error thresholds meet acceptance gate on frozen benchmark."""
    benchmark = run_temporal_grounding_benchmark(MANIFEST_PATH)
    # Acceptance gate: Macro IoU >= 0.65; Onset/Offset error < 0.35s (350ms)
    assert benchmark["macro_iou"] >= 0.65, f"Macro IoU {benchmark['macro_iou']} below 0.65 gate"
    assert benchmark["mean_onset_error_s"] <= 0.35, f"Onset error {benchmark['mean_onset_error_s']}s exceeds 350ms gate"
    assert benchmark["mean_offset_error_s"] <= 0.35, f"Offset error {benchmark['mean_offset_error_s']}s exceeds 350ms gate"


def main():
    parser = argparse.ArgumentParser(description="VoxContrast Temporal Grounding Benchmark")
    parser.add_argument("--manifest", type=str, default=MANIFEST_PATH, help="Path to manifest JSON")
    parser.add_argument("--split", type=str, default=None, help="Split name to filter (e.g. validation, held_out_test)")
    parser.add_argument("--all", action="store_true", help="Evaluate across original benchmark, validation, and held_out_test")
    parser.add_argument("--include-uncertain", action="store_true", help="Include unconfirmed/uncertain flaws in evaluation")

    args = parser.parse_args()
    confirmed_only = not args.include_uncertain

    if args.all:
        print("Running comprehensive multi-split evaluation...\n")
        benchmarks = []
        # 1. Original Frozen Benchmark
        res_frozen = run_temporal_grounding_benchmark(
            manifest_path=MANIFEST_PATH,
            split=None,
            confirmed_only=confirmed_only
        )
        res_frozen["split"] = "original_frozen_benchmark"
        print_benchmark_report(res_frozen, "Original Frozen Benchmark (data/manifest.json)")
        benchmarks.append(res_frozen)

        # 2. TTS Validation Split (if exists)
        if os.path.exists(TTS_MANIFEST_PATH):
            res_val = run_temporal_grounding_benchmark(
                manifest_path=TTS_MANIFEST_PATH,
                split="validation",
                confirmed_only=confirmed_only
            )
            print_benchmark_report(res_val, "TTS Expanded - Validation Split (speech_07_roosevelt, speech_08_curie)")
            benchmarks.append(res_val)

            # 3. TTS Held-Out Test Split (if exists)
            res_test = run_temporal_grounding_benchmark(
                manifest_path=TTS_MANIFEST_PATH,
                split="held_out_test",
                confirmed_only=confirmed_only
            )
            print_benchmark_report(res_test, "TTS Expanded - Held-Out Test Split (speech_09_mandela, speech_10_turing)")
            benchmarks.append(res_test)

        print_comparative_table(benchmarks)

    else:
        res = run_temporal_grounding_benchmark(
            manifest_path=args.manifest,
            split=args.split,
            confirmed_only=confirmed_only
        )
        print_benchmark_report(res)


if __name__ == "__main__":
    main()
