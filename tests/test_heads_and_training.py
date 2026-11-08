import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data.floodnet import NUM_CLASSES                                     # noqa: E402
from experiments.train_uncertainty_head import train                       # noqa: E402
from src.components.evidential import EvidentialSegHead                    # noqa: E402
from src.components.heads import SoftmaxSegHead, build_head, load_head, save_head   # noqa: E402
from tests.conftest import make_floodnet_tree                              # noqa: E402
from tests.test_yolo_hook import FakeDet, _Sum                         # noqa: E402


def test_heads_share_architecture():
    e, s = EvidentialSegHead(24), SoftmaxSegHead(24)
    assert [tuple(p.shape) for p in e.parameters()] == [tuple(p.shape) for p in s.parameters()]
    x = torch.randn(2, 24, 8, 8)
    assert e(x, (16, 16)).min() >= 1.0 and s(x, (16, 16)).shape == (2, NUM_CLASSES, 16, 16)
    with pytest.raises(ValueError):
        build_head("nope", 24)


def test_checkpoint_roundtrip(tmp_path):
    h = build_head("evidential", 24, hidden=16)
    save_head(tmp_path / "c.pt", h, {"head": "evidential", "in_channels": 24, "num_classes": NUM_CLASSES,
                                      "hidden": 16, "temperature": 1.0})
    h2, meta = load_head(tmp_path / "c.pt")
    x = torch.randn(1, 24, 4, 4)
    h.eval()
    assert torch.allclose(h(x), h2(x)) and meta["hidden"] == 16


def smoke_cfg(tmp_path, **kw):
    cfg = {"data_root": str(tmp_path / "d"), "yolo_weights": "unused", "imgsz": 64, "levels": [0, 1],
           "hidden": 16, "out_stride": 2, "epochs": 10, "batch": 2, "lr": 0.02, "weight_decay": 0.0,
           "anneal_epochs": 4, "num_workers": 0, "device": "cpu", "amp": False, "val_max_images": None,
           "temperature_pixels_per_image": 256}
    cfg.update(kw)
    return cfg


@pytest.fixture
def fake_data(tmp_path):
    counts = {"train": 6, "val": 4, "test": 2}
    make_floodnet_tree(tmp_path / "d", counts)
    return counts


@pytest.mark.parametrize("kind", ["softmax", "evidential"])
def test_training_loop_runs_and_writes_artifacts(tmp_path, fake_data, kind):
    out = tmp_path / "run"
    summary = train(smoke_cfg(tmp_path), kind, 7, out, model=FakeDet(_Sum()), expected=fake_data)
    hist = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert len(hist) == 10 and all(np.isfinite(h["train_loss"]) for h in hist)
    assert 0.0 <= hist[-1]["val"]["ece"] <= 1.0 and summary["strict_split"] is False
    head, meta = load_head(out / "best.pt")
    assert meta["head"] == kind and meta["levels"] == [0, 1] and meta["in_channels"] == 24
    if kind == "softmax":                                   # constant-grey images: the head can only learn the label prior
        assert hist[-1]["train_loss"] < hist[0]["train_loss"]
        assert summary["temperature"] > 0 and meta["temperature"] == summary["temperature"]
        assert "val_metrics_after_temperature" in summary
    else:
        assert summary["temperature"] is None and hist[0]["kl_weight"] == pytest.approx(0.25)
        assert summary["val_metrics"]["aurc_vacuity"] is not None


def test_same_seed_gives_same_history(tmp_path, fake_data):
    def run(name):
        torch.manual_seed(0)
        det = FakeDet(_Sum())
        sd = {k: v.clone() for k, v in det.state_dict().items()}
        det.load_state_dict(sd)
        return train(smoke_cfg(tmp_path, epochs=3), "evidential", 5, tmp_path / name, model=det, expected=fake_data)
    a, b = run("a"), run("b")
    assert a["val_metrics"]["loss"] == pytest.approx(b["val_metrics"]["loss"], rel=1e-6)
