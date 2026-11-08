import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data.floodnet import FloodNetDataset                                           # noqa: E402
from experiments.run_e3 import (ablation_config, evaluate, fit_operating_points, fit_pipeline,   # noqa: E402
                                run_seed)
from src.components.router import DynamicRouter, RouterConfig                        # noqa: E402
from src.pipeline import AgenticPipeline, PipelineConfig                             # noqa: E402
from src.state import Route                                                          # noqa: E402
from src.utils.cells import CellConfig                                               # noqa: E402
from src.utils.metrics import aggregate_runs                                         # noqa: E402
from tests.conftest import make_floodnet_tree                                        # noqa: E402
from tests.test_pipeline import FakeFastPath, FakeHead, flooded_vlm, scene           # noqa: E402

CFG = {"target_latency_s": 2.0, "max_escalation_rate": None, "target_coverage": 0.8, "fit_tiles": None,
       "calib_tiles": 2}


def test_operating_points_follow_the_measured_times():
    scores = list(np.linspace(0.1, 0.9, 100))
    f = fit_operating_points(scores, [0.2] * 5 + [0.8] * 5, t_fast=0.5, t_extra=3.0, cfg=CFG,
                             router_mode="entropy", has_verifier=True)
    assert f["max_escalation_rate"] == pytest.approx(0.5)
    assert np.mean(np.array(scores) >= f["router_threshold"]) <= 0.5 + 1e-9
    assert f["latency_target_feasible"] is True
    assert f["verifier_threshold"] == pytest.approx(0.8)          # 80% coverage of [0.2]*5 + [0.8]*5


def test_infeasible_latency_target_means_no_escalation():
    scores = list(np.linspace(0.1, 0.9, 50))
    f = fit_operating_points(scores, [], t_fast=2.5, t_extra=3.0, cfg=CFG, router_mode="entropy", has_verifier=False)
    assert f["latency_target_feasible"] is False and f["escalation_rate_used"] == 0.0
    assert f["router_threshold"] > max(scores)


def test_extra_cap_and_missing_measurements():
    scores = list(np.linspace(0, 1, 100))
    f = fit_operating_points(scores, [], 0.1, 1.0, {**CFG, "max_escalation_rate": 0.1}, "entropy", False)
    assert f["escalation_rate_used"] == 0.1 and f["max_escalation_rate"] == 1.0
    with pytest.raises(ValueError):
        fit_operating_points(scores, [], None, None, CFG, "entropy", False)
    g = fit_operating_points([], [0.3, 0.4], None, None, CFG, "always_vlm", True)
    assert g["router_threshold"] is None and g["verifier_threshold"] is not None


def test_ablation_switches():
    base = PipelineConfig()
    assert ablation_config(base, "none") == base and ablation_config(base, None) == base
    assert ablation_config(base, "no_gap1").use_graph is False
    g2 = ablation_config(base, "no_gap2")
    assert g2.uncertainty_source == "instance_conf" and g2.verifier == "instance_conf" and g2.use_graph
    assert ablation_config(base, "no_gap3").router_mode == "always_vlm"
    with pytest.raises(ValueError):
        ablation_config(base, "nope")


@pytest.fixture
def datasets(tmp_path):
    counts = {"train": 2, "val": 3, "test": 4}
    make_floodnet_tree(tmp_path / "d", counts)
    return (FloodNetDataset(tmp_path / "d", "val", expected=counts),
            FloodNetDataset(tmp_path / "d", "test", expected=counts))


def pipe(cfg=None, **kw):
    cfg = cfg or PipelineConfig(imgsz=128, keep_alpha=True)
    head = FakeHead(vacuous_from=32) if cfg.uncertainty_source == "evidential" else None
    return AgenticPipeline(scene(), cfg, head=head, head_meta={"out_stride": 2, "levels": [0, 1]},
                           router=DynamicRouter(RouterConfig(mode=cfg.router_mode, max_regions=8)),
                           vlm=flooded_vlm(), **kw)


def test_fit_then_evaluate_end_to_end(datasets, tmp_path):
    val, test = datasets
    p = pipe()
    assert p.router.cfg.threshold is None and p.verifier.cfg.threshold is None
    run = run_seed(p, val, test, CFG, CellConfig(), tmp_path / "t.jsonl")
    assert p.router.cfg.threshold is not None and p.verifier.cfg.threshold is not None
    assert run["fit"]["n_fit_tiles"] == 3 and run["fit"]["t_fast_s"] > 0 and isinstance(run["fit"]["t_vlm_extra_s"], float)
    assert run["n_images"] == 4 and len(run["per_tile"]["ids"]) == 4
    assert np.array(run["per_tile"]["cell_cm"]).shape == (4, 2, 3, 3)
    assert set(run["latency"]) >= {"mean_s", "std_s", "median_s", "p95_s", "fps"}
    assert 0 <= run["routing"]["escalation_rate"] <= 1
    assert run["verifier"]["coverage"] + run["verifier"]["review_rate"] == pytest.approx(1.0)
    assert 0 <= run["calibration"]["head_ece"] <= 1 and run["metrics"]["pixel_accuracy"] is not None
    assert run["graph"]["n_components"] is not None
    assert len((tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    json.dumps(run, allow_nan=False)                                    # the whole result is JSON-safe
    agg = aggregate_runs([{k: v for k, v in r.items() if k not in ("per_tile", "fit", "seed")} for r in (run, run)])
    assert agg["latency"]["mean_s"]["std"] == pytest.approx(0.0)


def test_calibration_tiles_are_timed_with_forced_paths(datasets):
    val, _ = datasets
    p = pipe()
    fit_pipeline(p, val, CFG)
    # 3 observations + (fast, escalated) per calibration tile, each escalated pass queries the VLM
    assert len(p.vlm.calls) >= CFG["calib_tiles"] * 8


def test_always_vlm_and_instance_conf_ablations_fit_without_a_router_threshold(datasets):
    val, test = datasets
    p = pipe(PipelineConfig(imgsz=128, router_mode="always_vlm", keep_alpha=True))
    run = run_seed(p, val, test, CFG, CellConfig())
    assert run["fit"]["router_threshold"] is None and run["routing"]["escalation_rate"] == 1.0
    assert run["routing"]["mean_vlm_calls"] == 24
    p2 = pipe(PipelineConfig(imgsz=128, uncertainty_source="instance_conf", verifier="instance_conf"))
    run2 = run_seed(p2, val, test, CFG, CellConfig())
    assert run2["calibration"] is None and run2["verifier"]["review_rate"] >= 0


def test_graph_ablation_leaves_graph_stats_empty(datasets):
    val, test = datasets
    run = run_seed(pipe(PipelineConfig(imgsz=128, use_graph=False, keep_alpha=True)), val, test, CFG, CellConfig())
    assert run["graph"]["n_components"] is None


def test_entropy_routing_without_a_vlm_is_refused(datasets):
    val, _ = datasets
    p = AgenticPipeline(scene(), PipelineConfig(imgsz=128), head=FakeHead(), head_meta={"out_stride": 2},
                        router=DynamicRouter(RouterConfig()), vlm=None)
    with pytest.raises(RuntimeError, match="VLM backend"):
        fit_pipeline(p, val, CFG)
