"""
VoxContrast - Temporal Flaw Grounding IoU Benchmark
Evaluates temporal interval localization accuracy against ground-truth annotations.
Supports original frozen benchmark (data/manifest.json) and TTS-assisted expansion splits
(validation, held_out_test in data/tts_expanded/manifest.json and versioned expansions).
Calculates Bipartite 1-to-1 Matched Macro IoU, Precision@0.5, Recall@0.5,
Mean Onset/Offset errors, per-category statistics, and clean baseline false-positive rates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Ensure repository root is on Python module search path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import soundfile as sf

from core.analyzer import analyze_speech
from backend.config import MANIFEST_PATH, DATA_DIR

ORIGINAL_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifest.json")
TTS_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "tts_expanded", "manifest.json")
TTS_V2_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "tts_expanded_v2", "manifest.json")


def compute_interval_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculates temporal Intersection over Union between two 1D intervals."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_len = max(0.0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_len = max(1e-6, union_end - union_start)

    return float(intersection_len / union_len)


def is_category_match(det_category: str, gt_category: str) -> bool:
    """
    Checks if a detected category matches a ground-truth flaw category,
    accounting for established standard category aliases.
    """
    if det_category == gt_category:
        return True

    aliases = [
        {"muffled_articulation", "vocal_clarity"},
        {"slow_speech", "slow_delivery"},
    ]
    pair_set = {det_category, gt_category}
    return any(pair_set <= alias_set for alias_set in aliases)


def match_ground_truth_and_predictions(
    gt_flaws: list[dict],
    detected_flaws: list[Any],
    iou_threshold: float = 0.50
) -> dict:
    """
    Performs greedy bipartite 1-to-1 matching between ground-truth flaws and detections.

    Guarantees:
    - 1-to-1 matching: each detection matches at most one GT, and each GT matches at most one detection.
    - Deterministic order: candidate pairs sorted by IoU descending.
    - Conservation: TP + FP = len(detected_flaws); TP + FN = len(gt_flaws).
    - Unmatched predictions count as false positives.
    - Unmatched ground truth count as false negatives.
    - Zero-IoU misses are retained in GT IoU accounting (IoU = 0.0).
    """
    n_gt = len(gt_flaws)
    n_det = len(detected_flaws)

    if n_gt == 0:
        return {
            "matched_pairs": [],
            "true_positives": 0,
            "false_positives": n_det,
            "false_negatives": 0,
            "gt_ious": [],
            "onset_errors": [],
            "offset_errors": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    # Generate candidate matches with compatible categories
    candidates = []
    for g_idx, gt in enumerate(gt_flaws):
        for d_idx, det in enumerate(detected_flaws):
            det_cat = getattr(det, "category", None) or det.get("category")
            gt_cat = gt.get("category")
            if is_category_match(det_cat, gt_cat):
                d_start = getattr(det, "start_s", None) if hasattr(det, "start_s") else det["start_s"]
                d_end = getattr(det, "end_s", None) if hasattr(det, "end_s") else det["end_s"]
                iou = compute_interval_iou(d_start, d_end, gt["start_s"], gt["end_s"])
                candidates.append((iou, g_idx, d_idx))

    # Sort candidates by IoU descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_det = set()
    best_iou_per_gt = {g_idx: 0.0 for g_idx in range(n_gt)}
    best_det_per_gt = {g_idx: None for g_idx in range(n_gt)}
    matched_pairs = []

    for iou, g_idx, d_idx in candidates:
        if iou > best_iou_per_gt[g_idx] and g_idx not in matched_gt:
            best_iou_per_gt[g_idx] = iou
            best_det_per_gt[g_idx] = d_idx

        if iou >= iou_threshold and g_idx not in matched_gt and d_idx not in matched_det:
            matched_gt.add(g_idx)
            matched_det.add(d_idx)
            matched_pairs.append((g_idx, d_idx, iou))

    tp = len(matched_pairs)
    fp = n_det - tp
    fn = n_gt - tp

    prec = float(tp / n_det) if n_det > 0 else 0.0
    rec = float(tp / n_gt) if n_gt > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    gt_ious = [best_iou_per_gt[g_idx] for g_idx in range(n_gt)]

    onset_errors = []
    offset_errors = []
    for g_idx, d_idx, _ in matched_pairs:
        det = detected_flaws[d_idx]
        gt = gt_flaws[g_idx]
        d_start = getattr(det, "start_s", None) if hasattr(det, "start_s") else det["start_s"]
        d_end = getattr(det, "end_s", None) if hasattr(det, "end_s") else det["end_s"]
        onset_errors.append(abs(d_start - gt["start_s"]))
        offset_errors.append(abs(d_end - gt["end_s"]))

    return {
        "matched_pairs": matched_pairs,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "gt_ious": gt_ious,
        "onset_errors": onset_errors,
        "offset_errors": offset_errors,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def run_temporal_grounding_benchmark(
    manifest_path: str = MANIFEST_PATH,
    split: str | None = None,
    confirmed_only: bool = True
) -> dict:
    """
    Executes benchmark over contrastive pairs in the given manifest and optional split
    using strict bipartite 1-to-1 matching.
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

    total_gt_count = 0
    total_det_count = 0
    total_tp_05 = 0
    total_fp_05 = 0
    total_fn_05 = 0

    if split in ("all", "all_pairs"):
        split = None
    elif split in ("held_out_test", "test") and "splits" in manifest and "final_test" in manifest["splits"]:
        split = "final_test"

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

        pair_cat = pair["flaw_category"]
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

        match_res = match_ground_truth_and_predictions(gt_flaws, detected, iou_threshold=0.50)

        total_gt_count += len(gt_flaws)
        total_det_count += len(detected)
        total_tp_05 += match_res["true_positives"]
        total_fp_05 += match_res["false_positives"]
        total_fn_05 += match_res["false_negatives"]

        total_ious.extend(match_res["gt_ious"])
        total_onset_errs.extend(match_res["onset_errors"])
        total_offset_errs.extend(match_res["offset_errors"])

        # Category-wise statistics
        for g_idx, gt in enumerate(gt_flaws):
            c = gt["category"]
            if c not in category_stats:
                category_stats[c] = {
                    "ious": [],
                    "onset_errs": [],
                    "offset_errs": [],
                    "gt_count": 0,
                    "tp_count": 0,
                    "fp_count": 0,
                    "det_count": 0
                }
            category_stats[c]["gt_count"] += 1
            category_stats[c]["ious"].append(match_res["gt_ious"][g_idx])

        # Track per-category TP and Onset/Offset errors
        for g_idx, d_idx, _ in match_res["matched_pairs"]:
            c = gt_flaws[g_idx]["category"]
            category_stats[c]["tp_count"] += 1
            d = detected[d_idx]
            g = gt_flaws[g_idx]
            category_stats[c]["onset_errs"].append(abs(d.start_s - g["start_s"]))
            category_stats[c]["offset_errs"].append(abs(d.end_s - g["end_s"]))

        # Count detections per category for per-category precision
        for d in detected:
            det_c = d.category
            matched_to_stat = False
            for c_stat in category_stats:
                if is_category_match(det_c, c_stat):
                    category_stats[c_stat]["det_count"] += 1
                    matched_to_stat = True
                    break
            if not matched_to_stat:
                if det_c not in category_stats:
                    category_stats[det_c] = {
                        "ious": [],
                        "onset_errs": [],
                        "offset_errs": [],
                        "gt_count": 0,
                        "tp_count": 0,
                        "fp_count": 0,
                        "det_count": 0
                    }
                category_stats[det_c]["det_count"] += 1

    macro_iou = float(np.mean(total_ious)) if total_ious else 0.0
    category_macro_iou = (
        float(np.mean([np.mean(s["ious"]) for s in category_stats.values() if s["ious"]]))
        if category_stats else 0.0
    )
    mean_onset_err = float(np.mean(total_onset_errs)) if total_onset_errs else 0.0
    mean_offset_err = float(np.mean(total_offset_errs)) if total_offset_errs else 0.0

    precision_05 = float(total_tp_05 / total_det_count) if total_det_count > 0 else 0.0
    recall_05 = float(total_tp_05 / total_gt_count) if total_gt_count > 0 else 0.0
    f1_05 = (
        float(2 * precision_05 * recall_05 / (precision_05 + recall_05))
        if (precision_05 + recall_05) > 0
        else 0.0
    )

    split_label = split if split else ("all_pairs" if "splits" in manifest else "original_benchmark")
    results = {
        "split": split_label,
        "manifest_path": manifest_path,
        "macro_iou": round(macro_iou, 4),
        "category_macro_iou": round(category_macro_iou, 4),
        "mean_onset_error_s": round(mean_onset_err, 4),
        "mean_offset_error_s": round(mean_offset_err, 4),
        "precision_at_05_iou": round(precision_05, 4),
        "recall_at_05_iou": round(recall_05, 4),
        "f1_at_05_iou": round(f1_05, 4),
        "true_positives_05": total_tp_05,
        "false_positives_05": total_fp_05,
        "false_negatives_05": total_fn_05,
        "total_detections": total_det_count,
        "total_ground_truth_flaws": total_gt_count,
        "category_breakdown": {}
    }

    for cat, data in category_stats.items():
        c_tp = data["tp_count"]
        c_det = data["det_count"]
        c_gt = data["gt_count"]
        c_p = round(c_tp / c_det, 4) if c_det > 0 else 0.0
        c_r = round(c_tp / c_gt, 4) if c_gt > 0 else 0.0
        c_f1 = round(2 * c_p * c_r / (c_p + c_r), 4) if (c_p + c_r) > 0 else 0.0
        results["category_breakdown"][cat] = {
            "mean_iou": round(float(np.mean(data["ious"])), 4) if data["ious"] else 0.0,
            "mean_onset_err_s": round(float(np.mean(data["onset_errs"])), 4) if data["onset_errs"] else 0.0,
            "mean_offset_err_s": round(float(np.mean(data["offset_errs"])), 4) if data["offset_errs"] else 0.0,
            "gt_count": c_gt,
            "det_count": c_det,
            "tp_count": c_tp,
            "fp_count": max(0, c_det - c_tp),
            "fn_count": max(0, c_gt - c_tp),
            "precision_05": c_p,
            "recall_05": c_r,
            "f1_05": c_f1
        }

    return results


def evaluate_clean_baselines(
    manifest_path: str = MANIFEST_PATH,
    split: str | None = None
) -> dict:
    """
    Evaluates false-positive rate on clean baseline speech recordings where no flaws exist.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    speech_texts = {s["text_id"]: s["canonical_transcript"] for s in manifest.get("speech_texts", [])}

    # Discover unique baseline audio files
    baselines_seen = set()
    baseline_entries = []

    for pair in manifest.get("pairs", []):
        pair_id = pair["pair_id"]
        pair_split = pair.get("split")
        if split is not None:
            if "splits" in manifest and split in manifest["splits"]:
                if pair_id not in manifest["splits"][split]:
                    continue
            elif pair_split != split:
                continue

        b_rel = pair["baseline_audio"]
        if b_rel in baselines_seen:
            continue
        baselines_seen.add(b_rel)

        if os.path.isabs(b_rel):
            b_path = b_rel
        elif os.path.exists(os.path.join(REPO_ROOT, b_rel)):
            b_path = os.path.join(REPO_ROOT, b_rel)
        elif os.path.exists(os.path.join(os.path.dirname(manifest_path), b_rel)):
            b_path = os.path.join(os.path.dirname(manifest_path), b_rel)
        else:
            b_path = os.path.join(REPO_ROOT, b_rel)

        baseline_entries.append((pair["text_id"], b_path))

    total_duration_s = 0.0
    total_active_speech_s = 0.0
    total_false_positives = 0
    fp_by_category: dict[str, int] = {}

    for text_id, b_path in baseline_entries:
        if not os.path.exists(b_path):
            continue

        data, sr = sf.read(b_path, dtype="float32")
        dur = len(data) / sr
        total_duration_s += dur

        transcript = speech_texts.get(text_id, "")
        res = analyze_speech(b_path, transcript)

        ts = res.time_series
        active_frames = sum(1 for i in range(len(ts.time_stamps)) if ts.f0_contour[i] > 0 or ts.rms_energy_db[i] > -38.0)
        active_dur = active_frames * ts.time_step_s
        total_active_speech_s += active_dur

        dets = res.detected_flaws
        total_false_positives += len(dets)
        for d in dets:
            fp_by_category[d.category] = fp_by_category.get(d.category, 0) + 1

    dur_min = total_duration_s / 60.0 if total_duration_s > 0 else 1e-6
    active_min = total_active_speech_s / 60.0 if total_active_speech_s > 0 else 1e-6

    return {
        "manifest_path": manifest_path,
        "split": split or "all",
        "clean_utterances_count": len(baseline_entries),
        "total_duration_s": round(total_duration_s, 2),
        "total_active_speech_s": round(total_active_speech_s, 2),
        "total_false_positives": total_false_positives,
        "fp_per_minute_total_duration": round(total_false_positives / dur_min, 2),
        "fp_per_minute_active_speech": round(total_false_positives / active_min, 2),
        "fp_by_category": fp_by_category
    }


def print_benchmark_report(res: dict, title: str | None = None) -> None:
    """Prints a formatted evaluation report for a benchmark result."""
    heading = title or f"VoxContrast Temporal Grounding Results [{res.get('split', 'default')}]"
    print("=" * 66)
    print(heading)
    print("=" * 66)
    print(f"Manifest:               {res.get('manifest_path', 'unknown')}")
    print(f"Split:                  {res.get('split', 'all')}")
    print(f"Macro IoU (Flaw Mean):  {res['macro_iou']:.4f}")
    if "category_macro_iou" in res:
        print(f"Category Macro IoU:     {res['category_macro_iou']:.4f}")
    print(f"Mean Onset Error:       {res['mean_onset_error_s']*1000:.1f} ms")
    print(f"Mean Offset Error:      {res['mean_offset_error_s']*1000:.1f} ms")
    print(f"Precision @ IoU >= 0.5: {res['precision_at_05_iou']:.4f}")
    print(f"Recall @ IoU >= 0.5:    {res['recall_at_05_iou']:.4f}")
    print(f"F1 @ IoU >= 0.5:        {res['f1_at_05_iou']:.4f}")
    print(f"TP / FP / FN:           {res.get('true_positives_05', 'N/A')} / {res.get('false_positives_05', 'N/A')} / {res.get('false_negatives_05', 'N/A')}")
    print(f"Total Evaluated Flaws:  {res['total_ground_truth_flaws']}")
    print("-" * 66)
    print("Category Breakdown:")
    for cat, b in res["category_breakdown"].items():
        prec_str = f"Prec={b.get('precision_05', 0.0):.2f}"
        rec_str = f"Rec={b.get('recall_05', 0.0):.2f}"
        print(
            f"  - {cat:22s}: IoU={b['mean_iou']:.4f}, OnsetErr={b['mean_onset_err_s']*1000:.1f}ms, "
            f"GT={b['gt_count']}, Det={b.get('det_count', 0)}, {prec_str}, {rec_str}"
        )
    print("=" * 66)
    print()


def print_baseline_fp_report(b_res: dict) -> None:
    """Prints clean baseline false positive report."""
    print("=" * 66)
    print(f"Clean Baseline False-Positive Audit [{b_res.get('split', 'all')}]")
    print("=" * 66)
    print(f"Manifest:                       {b_res['manifest_path']}")
    print(f"Clean Utterances Evaluated:     {b_res['clean_utterances_count']}")
    print(f"Total Clean Audio Duration:     {b_res['total_duration_s']:.2f}s ({b_res['total_duration_s']/60:.2f} min)")
    print(f"Active Speech Duration:         {b_res['total_active_speech_s']:.2f}s ({b_res['total_active_speech_s']/60:.2f} min)")
    print(f"Total False Positives:          {b_res['total_false_positives']}")
    print(f"FP Rate (per total minute):     {b_res['fp_per_minute_total_duration']:.2f} FP/min")
    print(f"FP Rate (per active minute):    {b_res['fp_per_minute_active_speech']:.2f} FP/min")
    print("-" * 66)
    print("FP Count by Category:")
    for cat, cnt in sorted(b_res["fp_by_category"].items(), key=lambda x: -x[1]):
        print(f"  - {cat:24s}: {cnt}")
    print("=" * 66)
    print()


def print_comparative_table(results_list: list[dict]) -> None:
    """Prints a comparative markdown table across multiple evaluated splits."""
    print("### Multi-Split Temporal Grounding Benchmark Comparison")
    print()
    print("| Split | Flaws | Flaw IoU | Cat IoU | Onset Err | Offset Err | Prec@0.5 | Rec@0.5 | F1@0.5 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results_list:
        split_name = r.get("split", "unknown")
        cnt = r["total_ground_truth_flaws"]
        iou = f"{r['macro_iou']:.4f}"
        cat_iou = f"{r.get('category_macro_iou', r['macro_iou']):.4f}"
        onset = f"{r['mean_onset_error_s']*1000:.1f} ms"
        offset = f"{r['mean_offset_error_s']*1000:.1f} ms"
        p = f"{r['precision_at_05_iou']:.4f}"
        rec = f"{r['recall_at_05_iou']:.4f}"
        f1 = f"{r['f1_at_05_iou']:.4f}"
        print(f"| **{split_name}** | {cnt} | **{iou}** | {cat_iou} | {onset} | {offset} | {p} | {rec} | {f1} |")
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
    parser.add_argument("--baselines", action="store_true", help="Evaluate clean baseline false positive rates")

    args = parser.parse_args()
    confirmed_only = not args.include_uncertain

    if args.baselines:
        b_res = evaluate_clean_baselines(args.manifest, split=args.split)
        print_baseline_fp_report(b_res)
        return

    if args.all:
        print("Running comprehensive multi-split evaluation...\n")
        benchmarks = []
        # 1. Original Frozen Benchmark (Historical)
        res_frozen = run_temporal_grounding_benchmark(
            manifest_path=ORIGINAL_MANIFEST_PATH,
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

        # 4. Versioned TTS Expanded V2 (if exists)
        if os.path.exists(TTS_V2_MANIFEST_PATH):
            res_v2_dev = run_temporal_grounding_benchmark(
                manifest_path=TTS_V2_MANIFEST_PATH,
                split="train_dev",
                confirmed_only=confirmed_only
            )
            print_benchmark_report(res_v2_v2_dev := res_v2_dev, "TTS Expanded V2 - Train/Dev Split")
            benchmarks.append(res_v2_dev)

            res_v2_test = run_temporal_grounding_benchmark(
                manifest_path=TTS_V2_MANIFEST_PATH,
                split="final_test",
                confirmed_only=confirmed_only
            )
            print_benchmark_report(res_v2_test, "TTS Expanded V2 - Final Held-Out Evaluation Split")
            benchmarks.append(res_v2_test)

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
