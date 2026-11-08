import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
from src.components.yolo_features import YoloNeckFeatures, fuse_features, resize_for_yolo   # noqa: E402


class _NeverRuns(nn.Module):
    def forward(self, x):
        raise AssertionError("head must not run when extract() aborts")


class _Sum(nn.Module):
    def forward(self, x):
        return sum(f.mean() for f in x)


class FakeDet(nn.Module):
    """Mimics an Ultralytics model: .model[-1] is a head that receives [P3, P4, P5]."""

    def __init__(self, head, ch=(8, 16, 32)):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(3, ch[i], 3, stride=8 * 2 ** i, padding=1) for i in range(3)] + [head])

    def forward(self, x):
        return self.model[3]([self.model[i](x) for i in range(3)])


def test_extract_returns_three_levels_and_skips_the_head():
    nf = YoloNeckFeatures(FakeDet(_NeverRuns()))
    feats = nf.extract(torch.rand(2, 3, 64, 96))
    assert [tuple(f.shape) for f in feats] == [(2, 8, 8, 12), (2, 16, 4, 6), (2, 32, 2, 3)]
    assert nf.channels() == [8, 16, 32]
    assert nf._handle is None                     # hook removed afterwards


def test_attach_without_abort_runs_head_and_captures():
    det = FakeDet(_Sum())
    nf = YoloNeckFeatures(det).attach(abort=False)
    out = det(torch.rand(1, 3, 64, 64))
    assert out.ndim == 0 and len(nf.features) == 3
    nf.detach()
    nf.features = None
    det(torch.rand(1, 3, 64, 64))
    assert nf.features is None                    # detached hook does nothing


def test_hook_rejects_a_head_that_does_not_take_a_list():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([nn.Identity()])

        def forward(self, x):
            return self.model[0](x)

    with pytest.raises(TypeError, match="list of neck feature maps"):
        YoloNeckFeatures(Odd()).extract(torch.rand(1, 3, 32, 32))


def test_fuse_features_resizes_to_first_level():
    f = [torch.rand(1, 4, 8, 8), torch.rand(1, 6, 4, 4), torch.rand(1, 8, 2, 2)]
    assert fuse_features(f, (0, 1)).shape == (1, 10, 8, 8)
    assert fuse_features(f, (0, 1, 2)).shape == (1, 18, 8, 8)
    assert fuse_features(f, (2,)).shape == (1, 8, 2, 2)


@pytest.mark.skipif(not os.environ.get("YOLO_SMOKE_WEIGHTS"),
                    reason="set YOLO_SMOKE_WEIGHTS to a yolov8*-seg .pt to run against real Ultralytics")
def test_real_ultralytics_single_pass_gives_results_and_aligned_features():
    from ultralytics import YOLO
    y = YOLO(os.environ["YOLO_SMOKE_WEIGHTS"])
    nf = YoloNeckFeatures.for_predictor(y, device="cpu")
    small, (h, w) = resize_for_yolo((np.random.rand(600, 800, 3) * 255).astype(np.uint8), 256)
    for dev in ("cpu", None):                     # a changed `device` makes Ultralytics rebuild its model
        res, feats = nf.predict_with_features(np.ascontiguousarray(small[..., ::-1]), imgsz=(h, w),
                                              conf=0.01, device=dev or "cpu")
        assert res.orig_shape == (h, w)
        assert [tuple(f.shape[-2:]) for f in feats] == [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
