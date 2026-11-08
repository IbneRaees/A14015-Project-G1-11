"""Uncertainty map from YOLO's own instance confidences (no evidential head).

Used as the router's input when Gap 2 is ablated in E5, so removing the
evidential head does not also remove the router's signal. It is a naive
pre-filter on purpose: per pixel, 1 - confidence of the best instance that
covers it. Pixels no instance covers get `uncovered_value`.

Run YOLO with a low `conf` (for example 0.05) when producing the inputs;
at the default 0.25 every covered pixel has uncertainty <= 0.75 and weak
detections never show up.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


def instance_uncertainty_map(polygons: Sequence[np.ndarray], confidences: Sequence[float],
                             shape: Tuple[int, int], uncovered_value: float = 0.5) -> np.ndarray:
    """(H, W) float32 map in [0, 1].

    `uncovered_value` is an uninformative prior for pixels with no detection;
    tune it on the validation split like any other router hyper-parameter.
    """
    best = np.full(shape, -1.0, np.float32)
    for poly, conf in zip(polygons, confidences):
        poly = np.asarray(poly)
        if len(poly) < 3:
            continue
        layer = np.zeros(shape, np.uint8)
        cv2.fillPoly(layer, [np.round(poly).astype(np.int32)], 1)
        np.maximum(best, np.where(layer == 1, np.float32(conf), np.float32(-1.0)), out=best)
    out = np.where(best >= 0, 1.0 - best, uncovered_value).astype(np.float32)
    return np.clip(out, 0.0, 1.0)
