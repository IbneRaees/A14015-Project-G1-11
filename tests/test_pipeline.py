import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from src.components.router import RouterConfig                                    # noqa: E402
from src.pipeline import (AgenticPipeline, FastPathOutput, PipelineConfig, total_latency)   # noqa: E402
from src.state import Disposition, Route                                          # noqa: E402
from src.vlm.backends import MockVLM                                              # noqa: E402

H, W = 240, 320            # original tile; imgsz=128 -> resized 96 x 128
SMALL = (96, 128)


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


class FakeFastPath:
    """Scripted YOLO: fixed instances in resized-image pixels, plus zero neck features."""

    def __init__(self, polys, classes, confs):
        self.polys, self.classes, self.confs = polys, classes, confs

    def run(self, bgr, hw):
        h, w = hw
        feats = [torch.zeros(1, c, h // (8 * 2 ** i), w // (8 * 2 ** i)) for i, c in enumerate((8, 16, 32))]
        return FastPathOutput(list(self.polys), np.array(self.classes, int), np.array(self.confs, float), feats)


class FakeHead(nn.Module):
    """Confident road everywhere except x >= vacuous_from (in head pixels), where it has no evidence."""

    def __init__(self, vacuous_from=None):
        super().__init__()
        self.vacuous_from = vacuous_from

    def forward(self, fused, out_size):
        h, w = out_size
        a = torch.ones(1, 10, h, w)
        a[:, 4] = 50
        if self.vacuous_from is not None:
            a[:, :, :, self.vacuous_from:] = 1.0
        return a


def scene():
    """left dry building, upper-right dry building, a road band across the tile (resized pixels)."""
    return FakeFastPath([rect(10, 10, 40, 30), rect(80, 4, 120, 20), rect(0, 40, 127, 52)], [2, 2, 4], [0.8, 0.8, 0.9])


def image():
    return (np.random.default_rng(0).random((H, W, 3)) * 255).astype(np.uint8)


def flooded_vlm():
    return MockVLM(lambda img, p: '{"buildings": "flooded", "roads": "flooded"}')


def make(fast=None, head=None, vlm=None, cfg=None, router_cfg=None, router_thr=0.6, ver_thr=0.3):
    cfg = cfg or PipelineConfig(imgsz=128)
    head = FakeHead() if head is None and cfg.uncertainty_source == "evidential" else head
    from src.components.router import DynamicRouter
    router = DynamicRouter(router_cfg or RouterConfig(mode=cfg.router_mode, max_regions=64, threshold=router_thr))
    p = AgenticPipeline(fast or scene(), cfg, head=head, head_meta={"out_stride": 2, "levels": [0, 1]},
                        router=router, vlm=vlm)
    p.set_verifier_threshold(ver_thr)
    return p


def test_confident_tile_stays_on_fast_path_and_is_auto_reported():
    vlm = flooded_vlm()
    out = make(vlm=vlm).process("t1", image())
    st = out.state
    assert st.routing.route == Route.FAST and vlm.calls == [] and st.vision.vlm_calls == 0
    assert st.verdict.disposition == Disposition.AUTO_REPORT
    assert out.label_small.shape == SMALL and set(np.unique(out.label_small)) == {0, 2, 4}
    assert st.road_graph.n_edges >= 1
    tools = [s["tool"] for s in st.trajectory if not s.get("nested")]
    assert tools == ["preprocess", "fast_path", "evidential_head", "label_map", "route", "geospatial", "verifier"]


def test_uncertain_right_half_escalates_and_the_vlm_relabels_only_those_regions():
    vlm = flooded_vlm()
    out = make(head=FakeHead(vacuous_from=32), vlm=vlm).process("t2", image())
    st, lab = out.state, out.label_small
    assert st.routing.route == Route.VLM and len(st.routing.regions) == 32 == len(vlm.calls)
    assert all(x0 >= 160 for x0, _, _, _ in st.routing.regions)             # original coordinates
    assert lab[46, 20] == 4 and lab[46, 100] == 3                           # road: left kept, right flooded
    assert lab[20, 20] == 2 and (lab[5:18, 85:118] == 1).all()              # buildings: left kept, right flooded
    assert st.vision.vlm_calls == 32 and st.vision.model.endswith(vlm.name)
    assert any(s["tool"] == "vlm_refine" for s in st.trajectory)


def test_unreadable_vlm_reply_changes_nothing_and_is_counted():
    vlm = MockVLM(lambda img, p: "I cannot tell")
    base = make().process("t", image()).label_small
    out = make(head=FakeHead(vacuous_from=32), vlm=vlm).process("t", image())
    assert (out.label_small == base).all()
    step = next(s for s in out.state.trajectory if s["tool"] == "vlm_refine")
    assert "0 px relabelled" in step["summary"] and "64 unparsed" in step["summary"]


def test_vlm_saying_none_never_erases_a_detection():
    vlm = MockVLM(lambda img, p: '{"buildings": "none", "roads": "none"}')
    base = make().process("t", image()).label_small
    assert (make(head=FakeHead(vacuous_from=32), vlm=vlm).process("t", image()).label_small == base).all()


def test_escalation_without_a_backend_keeps_fast_labels_and_notes_it():
    out = make(head=FakeHead(vacuous_from=32), vlm=None).process("t", image())
    assert out.state.routing.route == Route.VLM and out.state.vision.vlm_calls == 0
    assert any("no VLM backend" in e for e in out.state.errors)


def test_always_vlm_re_judges_the_whole_grid_and_always_fast_never_calls():
    vlm = flooded_vlm()
    make(cfg=PipelineConfig(imgsz=128, router_mode="always_vlm"), vlm=vlm).process("t", image())
    assert len(vlm.calls) == 24                                             # 4 x 6 grid
    vlm2 = flooded_vlm()
    make(cfg=PipelineConfig(imgsz=128, router_mode="always_fast"), vlm=vlm2).process("t", image())
    assert vlm2.calls == []


def test_graph_ablation_removes_graph_and_its_edge_check():
    out = make(cfg=PipelineConfig(imgsz=128, use_graph=False), head=FakeHead(vacuous_from=32)).process("t", image())
    assert out.state.road_graph is None and out.road_result is None
    assert not any("geospatial" in s["tool"] for s in out.state.trajectory)
    assert out.state.uncertainty.uncertain_edges == []


def test_uncertain_road_edge_reaches_the_verdict_with_original_coordinates():
    out = make(head=FakeHead(vacuous_from=32), vlm=None, router_thr=2.0).process("t", image())   # never escalates
    st = out.state
    assert st.routing.route == Route.FAST
    assert st.uncertainty.uncertain_edges and st.verdict.disposition == Disposition.HUMAN_REVIEW
    assert st.uncertainty.flagged_regions and all(b[0] >= 160 and b[2] <= W for b in st.uncertainty.flagged_regions)


def test_instance_confidence_ablation_needs_no_head():
    cfg = PipelineConfig(imgsz=128, uncertainty_source="instance_conf", verifier="instance_conf")
    weak = FakeFastPath([rect(10, 10, 90, 80)], [2], [0.3])
    strong = FakeFastPath([rect(10, 10, 90, 80)], [2], [0.95])
    router_cfg = RouterConfig(top_k=1, threshold=0.6)
    p_weak = make(fast=weak, cfg=cfg, router_cfg=router_cfg, vlm=flooded_vlm(), ver_thr=0.5)
    p_strong = make(fast=strong, cfg=cfg, router_cfg=RouterConfig(top_k=1, threshold=0.6), vlm=flooded_vlm(), ver_thr=0.5)
    assert p_weak.head is None
    ow, os_ = p_weak.process("w", image()), p_strong.process("s", image())
    assert ow.state.routing.route == Route.VLM and os_.state.routing.route == Route.FAST
    assert ow.state.verdict.disposition == Disposition.HUMAN_REVIEW
    assert os_.state.verdict.disposition == Disposition.AUTO_REPORT and os_.state.uncertainty.measure == "instance_conf"


def test_empty_detections_do_not_crash():
    out = make(fast=FakeFastPath([], [], [])).process("t", image())
    assert out.label_small.sum() == 0 and out.state.road_graph.n_edges == 0
    assert out.state.verdict is not None


def test_observe_needs_no_thresholds():
    from src.components.router import DynamicRouter
    p = AgenticPipeline(scene(), PipelineConfig(imgsz=128), head=FakeHead(vacuous_from=32),
                        head_meta={"out_stride": 2, "levels": [0, 1]}, router=DynamicRouter(RouterConfig()))
    obs = p.observe(image())
    assert obs["router_score"] == pytest.approx(1.0, abs=1e-3) and 0 < obs["verifier_uncertainty"] < 1
    assert obs["fast_stage_s"] > 0


def test_latency_counts_only_top_level_steps_and_state_serialises():
    out = make(head=FakeHead(vacuous_from=32), vlm=flooded_vlm()).process("t", image())
    st = out.state
    nested = [s for s in st.trajectory if s.get("nested")]
    assert nested and total_latency(st) == pytest.approx(sum(s["duration_s"] for s in st.trajectory if not s.get("nested")))
    assert total_latency(st) < sum(s["duration_s"] for s in st.trajectory) + 1e-9
    d = json.loads(st.to_json())
    assert d["vision"]["vlm_calls"] == 32 and d["routing"]["route"] == "vlm"
    assert out.alpha is None and "label_small" in d["arrays"]


def test_keep_alpha_and_constructor_validation():
    out = make(cfg=PipelineConfig(imgsz=128, keep_alpha=True)).process("t", image())
    assert out.alpha.shape == (10, 48, 64)
    with pytest.raises(ValueError, match="evidential head"):
        AgenticPipeline(scene(), PipelineConfig(imgsz=128), head=None)
    with pytest.raises(ValueError):
        AgenticPipeline(scene(), PipelineConfig(imgsz=128, verifier="bogus"), head=FakeHead())
