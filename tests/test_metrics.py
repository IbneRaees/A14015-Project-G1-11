import numpy as np
import pytest

from src.utils.metrics import aggregate_runs, confusion_matrix, summarize


def test_confusion_and_scores_by_hand():
    gt = np.array([[0, 1, 1, 3], [3, 3, 4, 4]], np.uint8)
    pr = np.array([[0, 1, 2, 3], [3, 4, 4, 4]], np.uint8)
    cm = confusion_matrix(gt, pr)
    assert cm.sum() == 8 and cm[1, 1] == 1 and cm[1, 2] == 1 and cm[3, 4] == 1
    s = summarize(cm)
    # class 1: tp=1 fp=0 fn=1 -> IoU 1/2, F1 2/3
    assert s["per_class"]["building-flooded"]["iou"] == pytest.approx(0.5)
    assert s["per_class"]["building-flooded"]["f1"] == pytest.approx(2 / 3)
    # class 3: tp=2 fp=0 fn=1 -> IoU 2/3 ; class 4: tp=2 fp=1 fn=0 -> IoU 2/3
    assert s["tasks"]["road"]["miou"] == pytest.approx(2 / 3)
    # class 2 has no GT pixels but one false positive -> IoU 0, still counted
    assert s["per_class"]["building-non-flooded"]["iou"] == 0.0
    assert s["pixel_accuracy"] == pytest.approx(6 / 8)


def test_absent_classes_are_none_not_zero():
    gt = np.zeros((4, 4), np.uint8)
    s = summarize(confusion_matrix(gt, gt))
    assert s["per_class"]["vehicle"]["iou"] is None
    assert s["miou_all"] == 1.0


def test_perfect_and_empty_prediction():
    gt = np.random.default_rng(0).integers(0, 10, (30, 30)).astype(np.uint8)
    assert summarize(confusion_matrix(gt, gt))["miou_all"] == pytest.approx(1.0)
    assert summarize(confusion_matrix(gt, np.zeros_like(gt)))["miou_foreground"] == 0.0


def test_range_and_shape_checks():
    with pytest.raises(ValueError):
        confusion_matrix(np.zeros((2, 2), np.uint8), np.full((2, 2), 11, np.uint8))
    with pytest.raises(ValueError):
        confusion_matrix(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))


def test_aggregate_mean_std():
    runs = [{"a": {"x": 1.0}, "n": 5}, {"a": {"x": 3.0}, "n": 5}, {"a": {"x": 5.0}, "n": 5}]
    agg = aggregate_runs(runs)
    assert agg["a"]["x"] == {"mean": 3.0, "std": 2.0}
    assert agg["n"]["std"] == 0.0


def test_single_run_has_no_std_and_none_stays_none():
    agg = aggregate_runs([{"x": 1.0, "y": None}])
    assert agg["x"]["std"] is None and agg["y"] is None
