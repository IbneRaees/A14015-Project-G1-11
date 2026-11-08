"""Segmentation heads on top of frozen YOLO neck features, plus checkpoint I/O.

`SoftmaxSegHead` has exactly the same body as `EvidentialSegHead`; only the
output differs (logits vs Dirichlet alpha). That keeps the Gap 2 comparison
about the evidential formulation and nothing else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HEAD_KINDS = ("evidential", "softmax")

if HAS_TORCH:
    from .evidential import EvidentialSegHead

    class SoftmaxSegHead(nn.Module):
        """Same layers as EvidentialSegHead, returns class logits (B, K, H, W)."""

        def __init__(self, in_channels: int, num_classes: int = 10, hidden: int = 64):
            super().__init__()
            self.num_classes = num_classes
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, num_classes, 1),
            )

        def forward(self, feats: "torch.Tensor", out_size=None) -> "torch.Tensor":
            logits = self.net(feats)
            if out_size is not None:
                logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
            return logits

    def build_head(kind: str, in_channels: int, num_classes: int = 10, hidden: int = 64) -> "nn.Module":
        if kind == "evidential":
            return EvidentialSegHead(in_channels, num_classes, hidden)
        if kind == "softmax":
            return SoftmaxSegHead(in_channels, num_classes, hidden)
        raise ValueError(f"head kind must be one of {HEAD_KINDS}")

    def save_head(path, head: "nn.Module", meta: dict) -> None:
        """Checkpoint = weights + everything needed to rebuild the head and its input pipeline."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "meta": meta}, path)

    def load_head(path, map_location="cpu") -> Tuple["nn.Module", dict]:
        ck = torch.load(path, map_location=map_location, weights_only=True)
        m = ck["meta"]
        head = build_head(m["head"], m["in_channels"], m["num_classes"], m["hidden"])
        head.load_state_dict(ck["state_dict"])
        return head.eval(), m
