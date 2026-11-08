"""E1 control flow with the model call stubbed out (no torch / ultralytics needed)."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from data.floodnet import SplitError
from experiments import run_baseline as rb
from src.utils.metrics import confusion_matrix, summarize
from tests.conftest import make_floodnet_tree


def fake_run(weights, ds, cfg, save_preds):
    gt = np.array([[3, 3, 4, 4], [1, 2, 2, 0]], np.uint8)
    cm = confusion_matrix(gt, gt)
    return {"weights": str(weights), "n_images": len(ds), "metrics": summarize(cm),
            "latency": {"mean_s": 1.5 + 0.1 * len(str(weights)) % 1, "std_s": 0.1, "median_s": 1.5,
                        "p95_s": 2.0, "fps": 0.6,
                        "ultralytics_ms": {"preprocess": 5.0, "inference": 900.0, "postprocess": 20.0}},
            "vram_peak_mb": None, "confusion_matrix": cm.tolist()}


def test_check_rejects_coco_weights():
    coco = SimpleNamespace(names={i: f"coco{i}" for i in range(80)})
    with pytest.raises(SystemExit, match="9 FloodNet classes"):
        rb.check_is_floodnet_model(coco)
    ok = SimpleNamespace(names={i: n for i, n in enumerate(rb.FLOODNET_YOLO_NAMES)})
    rb.check_is_floodnet_model(ok)


def run_main(monkeypatch, tmp_path, extra):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rb, "evaluate_weights", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_baseline", "--data-root", str(tmp_path / "d"), *extra])
    rb.main()


def test_smoke_run_writes_marked_non_strict_json(monkeypatch, tmp_path, tiny_counts):
    make_floodnet_tree(tmp_path / "d", tiny_counts)
    ws = []
    for i in range(3):
        w = tmp_path / f"w{i}.pt"
        w.write_text("x", encoding="utf-8")
        ws.append(str(w))
    run_main(monkeypatch, tmp_path, ["--weights", *ws, "--smoke-counts", "5,3,4"])
    out = json.loads((tmp_path / "results" / "e1_smoke_metrics.json").read_text(encoding="utf-8"))
    assert out["meta"]["strict_split"] is False and out["meta"]["n_runs"] == 3
    assert out["aggregate_mean_std"]["metrics"]["tasks"]["road"]["f1"]["mean"] == pytest.approx(1.0)
    assert out["aggregate_mean_std"]["metrics"]["tasks"]["road"]["f1"]["std"] == pytest.approx(0.0)
    assert not (tmp_path / "results" / "e1_baseline_metrics.json").exists()


def test_strict_run_refuses_a_wrong_sized_dataset(monkeypatch, tmp_path, tiny_counts):
    make_floodnet_tree(tmp_path / "d", tiny_counts)
    w = tmp_path / "w.pt"
    w.write_text("x", encoding="utf-8")
    with pytest.raises(SplitError):
        run_main(monkeypatch, tmp_path, ["--weights", str(w)])
    assert not (tmp_path / "results").exists()


def test_missing_weights_stop_before_any_output(monkeypatch, tmp_path, tiny_counts):
    make_floodnet_tree(tmp_path / "d", tiny_counts)
    with pytest.raises(SystemExit, match="weights not found"):
        run_main(monkeypatch, tmp_path, ["--weights", str(tmp_path / "nope.pt"), "--smoke-counts", "5,3,4"])
    assert not (tmp_path / "results").exists()
