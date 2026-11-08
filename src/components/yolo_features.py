"""Neck features of a YOLOv8-Seg model, for the uncertainty head (Gap 2).

The segmentation head of an Ultralytics model receives three neck feature maps
(P3, P4, P5 at strides 8, 16, 32) as one list. A forward pre-hook on that last
layer captures the list. `extract()` aborts right after the capture, so the
detection/mask branches are not computed; `attach()` keeps the hook in place so
a normal `yolo.predict(...)` call yields both instances and features from a
single forward pass.

Alignment: feed images that are already resized with `resize_for_yolo` and pass
imgsz=(h, w) to predict(); Ultralytics then adds no letterbox padding and the
features line up with the resized image.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def resize_for_yolo(image: np.ndarray, max_side: int = 1024, stride: int = 32) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize so the long side is ~max_side and both sides are multiples of `stride`.

    Aspect ratio changes by at most a few percent. Returns (image, (h, w)).
    """
    h, w = image.shape[:2]
    s = max_side / max(h, w)
    nh = max(stride, int(round(h * s / stride)) * stride)
    nw = max(stride, int(round(w * s / stride)) * stride)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (nw, nh), interpolation=interp), (nh, nw)


def resize_label(label: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a class-id map, so ids are never blended."""
    return cv2.resize(label, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


def downsample_labels_stride(label: np.ndarray, stride: int) -> np.ndarray:
    """Centre-sampled stride downsampling of (..., H, W) labels (works on numpy or torch)."""
    if stride == 1:
        return label
    o = stride // 2
    return label[..., o::stride, o::stride]


if HAS_TORCH:

    class _Captured(Exception):
        """Raised from the hook to stop the forward pass once features are stored."""

    class YoloNeckFeatures:
        """Hook on the last layer (the Segment/Detect head) of an Ultralytics model.

        `model` is the torch module, i.e. `YOLO(weights).model`. To capture features
        during `yolo.predict()` use `YoloNeckFeatures.for_predictor(yolo)` instead.
        """

        def __init__(self, model: "nn.Module"):
            self.model = model
            self.head = model.model[-1]
            self.yolo = None
            self.features: List["torch.Tensor"] | None = None
            self._handle = None
            self._abort = False

        @classmethod
        def for_predictor(cls, yolo, imgsz=(64, 64), device=None) -> "YoloNeckFeatures":
            """Hook the module that `yolo.predict()` actually runs.

            Ultralytics executes predictions on its own (fused) copy of the model
            inside `yolo.predictor`, so a hook on `yolo.model` never fires there.
            This sets the predictor up with a dummy call if needed.
            """
            if getattr(yolo, "predictor", None) is None or getattr(yolo.predictor, "model", None) is None:
                yolo.predict(np.zeros((*imgsz, 3), np.uint8), imgsz=imgsz, device=device, verbose=False)
            obj = cls(yolo.predictor.model.model)
            obj.yolo = yolo
            return obj

        def _rebind(self) -> None:
            """Point the hook at the module the predictor currently runs."""
            self.model = self.yolo.predictor.model.model
            self.head = self.model.model[-1]

        def predict_with_features(self, image_bgr: np.ndarray, **predict_kwargs):
            """One forward pass that returns (Results, [P3, P4, P5]).

            Ultralytics rebuilds its predictor model when some predict() arguments
            change (for example `device`), which orphans the hook. If that happens
            the hook is rebound to the new module and the call is repeated once.
            """
            if self.yolo is None:
                raise RuntimeError("build with YoloNeckFeatures.for_predictor(yolo)")
            for _ in range(2):
                self._rebind()
                self.features = None
                self.attach(abort=False)
                try:
                    result = self.yolo.predict(image_bgr, verbose=False, **predict_kwargs)[0]
                finally:
                    self.detach()
                if self.features is not None:
                    feats, self.features = self.features, None
                    return result, feats
            raise RuntimeError("neck features were not captured during predict()")

        def _hook(self, module, args):
            x = args[0]
            if not isinstance(x, (list, tuple)):
                raise TypeError("expected the head to receive a list of neck feature maps; "
                                "is this an Ultralytics segmentation/detection model?")
            self.features = list(x)
            if self._abort:
                raise _Captured

        def attach(self, abort: bool = False) -> "YoloNeckFeatures":
            self.detach()
            self._abort = abort
            self._handle = self.head.register_forward_pre_hook(self._hook)
            return self

        def detach(self) -> None:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None

        @torch.no_grad()
        def extract(self, x: "torch.Tensor") -> List["torch.Tensor"]:
            """Neck features [P3, P4, P5] for a float RGB batch in [0, 1]; head is not run."""
            self.model.eval()
            self.attach(abort=True)
            try:
                self.model(x)
            except _Captured:
                pass
            finally:
                self.detach()
            feats, self.features = self.features, None
            if feats is None:
                raise RuntimeError("hook did not fire; no features captured")
            return feats

        def channels(self, device="cpu") -> List[int]:
            """Channel count of each neck level, found with a dummy forward."""
            dummy = torch.zeros(1, 3, 64, 64, device=device)
            return [int(f.shape[1]) for f in self.extract(dummy)]

    def fuse_features(feats: Sequence["torch.Tensor"], levels: Sequence[int] = (0, 1)) -> "torch.Tensor":
        """Bilinearly resize the chosen levels to the first one's resolution and concatenate on channels."""
        base = feats[levels[0]]
        parts = [base] + [F.interpolate(feats[i], size=base.shape[-2:], mode="bilinear", align_corners=False)
                          for i in levels[1:]]
        return torch.cat(parts, dim=1)


def fused_channels(channels: Sequence[int], levels: Sequence[int] = (0, 1)) -> int:
    return int(sum(channels[i] for i in levels))
