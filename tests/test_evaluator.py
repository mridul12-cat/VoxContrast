"""
VoxContrast - Evaluator Unit & Edge-Case Test Suite
Validates the mathematical correctness, bipartite 1-to-1 matching, and metric conservation
of the temporal flaw grounding benchmark engine.
"""

import pytest
from tests.evaluate_temporal_iou import (
    compute_interval_iou,
    is_category_match,
    match_ground_truth_and_predictions,
)


class MockFlaw:
    def __init__(self, category: str, start_s: float, end_s: float):
        self.category = category
        self.start_s = start_s
        self.end_s = end_s


def test_compute_interval_iou_cases():
    # Identical
    assert compute_interval_iou(1.0, 3.0, 1.0, 3.0) == pytest.approx(1.0)
    # Disjoint
    assert compute_interval_iou(1.0, 2.0, 3.0, 4.0) == 0.0
    # Adjacent
    assert compute_interval_iou(1.0, 2.0, 2.0, 3.0) == 0.0
    # 50% overlap: [1, 3] and [2, 4] -> intersection [2, 3] (len 1), union [1, 4] (len 3) -> 1/3
    assert compute_interval_iou(1.0, 3.0, 2.0, 4.0) == pytest.approx(1.0 / 3.0)
    # Subset: [1, 4] and [2, 3] -> intersection 1, union 3 -> 1/3
    assert compute_interval_iou(1.0, 4.0, 2.0, 3.0) == pytest.approx(1.0 / 3.0)


def test_is_category_match_aliases():
    assert is_category_match("excessive_speed", "excessive_speed") is True
    assert is_category_match("slow_speech", "slow_delivery") is True
    assert is_category_match("slow_delivery", "slow_speech") is True
    assert is_category_match("muffled_articulation", "vocal_clarity") is True
    assert is_category_match("vocal_clarity", "muffled_articulation") is True
    assert is_category_match("monotone_pitch", "monotone_energy") is False
    assert is_category_match("excessive_speed", "unnatural_pause") is False


def test_evaluator_perfect_prediction():
    gts = [{"category": "excessive_speed", "start_s": 1.0, "end_s": 3.0}]
    dets = [MockFlaw("excessive_speed", 1.0, 3.0)]

    res = match_ground_truth_and_predictions(gts, dets, iou_threshold=0.50)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["gt_ious"] == [1.0]
    assert res["onset_errors"] == [0.0]
    assert res["offset_errors"] == [0.0]


def test_evaluator_partial_overlap_above_and_below_threshold():
    # Above threshold: [1.0, 3.0] vs [1.2, 3.0] -> intersection 1.8, union 2.0 -> IoU 0.90
    gts = [{"category": "unnatural_pause", "start_s": 1.0, "end_s": 3.0}]
    dets = [MockFlaw("unnatural_pause", 1.2, 3.0)]
    res = match_ground_truth_and_predictions(gts, dets, iou_threshold=0.50)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0

    # Below threshold: [1.0, 3.0] vs [2.5, 3.5] -> intersection 0.5, union 2.5 -> IoU 0.20
    dets_low = [MockFlaw("unnatural_pause", 2.5, 3.5)]
    res_low = match_ground_truth_and_predictions(gts, dets_low, iou_threshold=0.50)
    assert res_low["true_positives"] == 0
    assert res_low["false_positives"] == 1
    assert res_low["false_negatives"] == 1
    assert res_low["gt_ious"] == [pytest.approx(0.20)]


def test_evaluator_no_predictions():
    gts = [{"category": "monotone_pitch", "start_s": 1.0, "end_s": 3.0}]
    dets = []
    res = match_ground_truth_and_predictions(gts, dets, iou_threshold=0.50)
    assert res["true_positives"] == 0
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 1
    assert res["gt_ious"] == [0.0]
    assert res["recall"] == 0.0


def test_evaluator_no_ground_truth():
    gts = []
    dets = [MockFlaw("monotone_pitch", 1.0, 3.0)]
    res = match_ground_truth_and_predictions(gts, dets, iou_threshold=0.50)
    assert res["true_positives"] == 0
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 0
    assert res["precision"] == 0.0


def test_evaluator_duplicate_predictions_one_to_one():
    # 1 GT, 2 identical predictions -> exactly 1 TP and 1 FP
    gts = [{"category": "excessive_speed", "start_s": 1.0, "end_s": 2.5}]
    dets = [
        MockFlaw("excessive_speed", 1.0, 2.5),
        MockFlaw("excessive_speed", 1.0, 2.5)
    ]
    res = match_ground_truth_and_predictions(gts, dets, iou_threshold=0.50)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 0
    assert res["precision"] == 0.50
    assert res["recall"] == 1.00


def test_evaluator_two_gts_competing_for_single_prediction():
    # 2 GTs, 1 detection that overlaps GT0 better than GT1
    gts = [
        {"category": "unstable_energy", "start_s": 1.0, "end_s": 3.0},
        {"category": "unstable_energy", "start_s": 2.5, "end_s": 4.5}
    ]
    dets = [MockFlaw("unstable_energy", 1.0, 3.0)]  # IoU 1.0 with GT0, IoU 0.5/3.5 = 0.14 with GT1

    res = match_ground_truth_and_predictions(gts, dets, iou_threshold=0.50)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 1
    assert res["gt_ious"][0] == 1.0
    assert res["gt_ious"][1] == pytest.approx(0.5 / 3.5)


def test_evaluator_category_mismatch():
    gts = [{"category": "excessive_speed", "start_s": 1.0, "end_s": 3.0}]
    dets = [MockFlaw("monotone_pitch", 1.0, 3.0)]

    res = match_ground_truth_and_predictions(gts, dets, iou_threshold=0.50)
    assert res["true_positives"] == 0
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["gt_ious"] == [0.0]


def test_evaluator_category_alias_match():
    gts = [{"category": "slow_delivery", "start_s": 1.0, "end_s": 3.0}]
    dets = [MockFlaw("slow_speech", 1.0, 3.0)]

    res = match_ground_truth_and_predictions(gts, dets, iou_threshold=0.50)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["gt_ious"] == [1.0]


def test_evaluator_conservation_laws():
    # Multi-flaw mixed case
    gts = [
        {"category": "excessive_speed", "start_s": 1.0, "end_s": 2.0},
        {"category": "unnatural_pause", "start_s": 3.0, "end_s": 5.0},
        {"category": "monotone_pitch", "start_s": 6.0, "end_s": 8.0}
    ]
    dets = [
        MockFlaw("excessive_speed", 1.0, 2.0),   # TP
        MockFlaw("excessive_speed", 1.5, 2.5),   # FP (duplicate/unmatched)
        MockFlaw("unnatural_pause", 3.2, 4.8),   # TP (IoU 1.6/2.0 = 0.80)
        MockFlaw("monotone_energy", 6.0, 8.0)    # FP (category mismatch)
    ]
    res = match_ground_truth_and_predictions(gts, dets, iou_threshold=0.50)

    # TP + FP must equal total detections
    assert res["true_positives"] + res["false_positives"] == len(dets)
    # TP + FN must equal total ground truth
    assert res["true_positives"] + res["false_negatives"] == len(gts)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 2
    assert res["false_negatives"] == 1
