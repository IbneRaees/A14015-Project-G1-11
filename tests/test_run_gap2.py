import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tqdm")
nn = torch.nn

from data.floodnet import FloodNetDataset                                              # noqa: E402
from experiments.run_gap2_calibration import aggregate_systems, check_heads, run_seed   # noqa: E402
from src.eval.gap2 import SYSTEMS                                                       # noqa: E402
from tests.conftest import make_floodnet_tree                                           # noqa: E402
from tests.test_pipeline import FakeFastPath, scene                                     # noqa: E402

COUNTS = {"train": 2, "val": 3, "test": 4}
META = {"head": "evidential", "imgsz": 128, "out_stride": 2, "levels": [0, 1], "in_channels": 24,
        "yolo_weights": "w.pt", "temperature": 1.0}


class ClassHead(nn.Module):
    """Predicts class `cls` everywhere with a chosen strength; mode picks the output type."""

    def __init__(self, cls=4, strength=20.0, mode="alpha"):
        super().__init__()
        self.cls, self.strength, self.mode = cls, strength, mode

    def forward(self, fused, out_size):
        h, w = out_size
        z = torch.zeros(1, 10, h, w)
        z[:, self.cls] = self.strength
        return 1 + z if self.mode == "alpha" else z


@pytest.fixture
def data(tmp_path):
    make_floodnet_tree(tmp_path / "d", COUNTS)
    return (FloodNetDataset(tmp_path / "d", "val", expected=COUNTS), FloodNetDataset(tmp_path / "d", "test", expected=COUNTS))


CFG = {"map_conf": 0.25, "temperature_pixels_per_image": 500, "fit_tiles": None}


def test_run_seed_scores_all_five_systems_with_per_tile_bins(data):
    val, test = data
    run = run_seed(scene(), ClassHead(mode="alpha"), ClassHead(mode="logits"), META, {**META, "head": "softmax", "temperature": 2.5},
                   val, test, CFG, seed=7)
    assert run["temperatures"]["softmax"] == 2.5 and run["temperatures"]["evidential"] > 0
    assert run["tile_ids"] == [s.image_id for s in test.samples]
    assert set(run["systems"]) == set(SYSTEMS)
    for name in SYSTEMS:
        s = run["systems"][name]
        assert np.array(s["per_tile_bins"]).shape == (4, 15, 3) and 0 <= s["ece"] <= 1
    assert run["systems"]["softmax_raw"]["ece"] != run["systems"]["softmax_ts"]["ece"]        # the stored temperature is applied
    json.dumps(run, allow_nan=False)


def test_runs_are_reproducible_and_aggregate_without_per_tile_data(data):
    val, test = data
    kw = dict(ev_head=ClassHead(mode="alpha"), sm_head=ClassHead(mode="logits"))
    mk = lambda seed: run_seed(scene(), kw["ev_head"], kw["sm_head"], META, {**META, "head": "softmax", "temperature": 1.5},
                               val, test, CFG, seed)
    a, b, c = mk(1), mk(1), mk(2)
    assert a["temperatures"] == b["temperatures"] and a["systems"]["evidential"]["ece"] == b["systems"]["evidential"]["ece"]
    agg = aggregate_systems([a, b, c])
    assert set(agg) == set(SYSTEMS) and "per_tile_bins" not in agg["evidential"]
    assert agg["evidential"]["ece"]["std"] is not None and agg["yolo_instance_conf"]["ece"]["mean"] is not None
    json.dumps(agg, allow_nan=False)


def test_heads_with_different_settings_are_refused():
    sm = {**META, "head": "softmax"}
    check_heads(META, sm)
    for k, v in (("levels", [0]), ("out_stride", 4), ("imgsz", 256), ("yolo_weights", "other.pt")):
        with pytest.raises(ValueError, match=k):
            check_heads(META, {**sm, k: v})
