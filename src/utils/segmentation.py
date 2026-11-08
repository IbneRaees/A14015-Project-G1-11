"""Helpers that turn instance predictions into a semantic label map."""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


def instances_to_semantic(polygons: Sequence[np.ndarray], class_ids: Sequence[int],
                          confidences: Sequence[float], shape: Tuple[int, int]) -> np.ndarray:
    """Paint instance polygons into an (H, W) uint8 class-id map.

    Class ids are FloodNet ids (1-9); uncovered pixels stay 0 (background).
    Instances are painted in ascending confidence, so where two overlap the
    more confident one wins.
    """
    out = np.zeros(shape, np.uint8)
    for i in np.argsort(np.asarray(confidences, dtype=np.float64), kind="stable"):
        poly = np.asarray(polygons[i])
        if len(poly) < 3:
            continue
        cv2.fillPoly(out, [np.round(poly).astype(np.int32)], int(class_ids[i]))
    return out


def mask_to_polygons(mask: np.ndarray, min_area: int = 20, eps_frac: float = 0.002):
    """Outer contours of a binary mask as normalised (x, y) polygons in [0, 1].

    Holes are dropped (YOLO polygon labels have no hole support). Contours
    with fewer than 3 points or less than `min_area` pixels are skipped.
    """
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        polys.append(pts)
    return polys


def instances_to_semantic_with_conf(polygons: Sequence[np.ndarray], class_ids: Sequence[int],
                                    confidences: Sequence[float], shape: Tuple[int, int]):
    """Like `instances_to_semantic`, but also returns the winning instance's confidence per pixel.

    Returns (label uint8, conf float32, covered bool). Pixels no instance covers are background
    (label 0) with conf 0 and covered=False: the detector makes no claim there, so calibration of
    the instance-confidence baseline is scored on covered pixels only.
    """
    label = np.zeros(shape, np.uint8)
    conf = np.zeros(shape, np.float32)
    covered = np.zeros(shape, bool)
    for i in np.argsort(np.asarray(confidences, dtype=np.float64), kind="stable"):
        poly = np.asarray(polygons[i])
        if len(poly) < 3:
            continue
        layer = np.zeros(shape, np.uint8)
        cv2.fillPoly(layer, [np.round(poly).astype(np.int32)], 1)
        m = layer == 1
        label[m], conf[m], covered[m] = int(class_ids[i]), float(confidences[i]), True
    return label, conf, covered
