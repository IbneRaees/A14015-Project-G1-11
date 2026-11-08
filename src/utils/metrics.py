"""Segmentation metrics computed from a dataset-level confusion matrix.

Counting pixels over the whole split (not averaging per image) is the usual
protocol for semantic segmentation and keeps rare classes from being
dominated by images where they are absent.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

NUM_CLASSES = 10
CLASS_NAMES = [
    "background", "building-flooded", "building-non-flooded", "road-flooded",
    "road-non-flooded", "water", "tree", "vehicle", "pool", "grass",
]
TASKS = {"building": (1, 2), "road": (3, 4)}


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, n: int = NUM_CLASSES) -> np.ndarray:
    """cm[i, j] = pixels with true class i predicted as j."""
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch {gt.shape} vs {pred.shape}")
    if gt.max(initial=0) >= n or pred.max(initial=0) >= n:
        raise ValueError("class id out of range")
    combo = gt.astype(np.uint16) * n + pred.astype(np.uint16)
    return np.bincount(combo.ravel(), minlength=n * n).reshape(n, n).astype(np.int64)


def _safe(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def per_class(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return {
        "iou": _safe(tp, tp + fp + fn),
        "f1": _safe(2 * tp, 2 * tp + fp + fn),
        "precision": _safe(tp, tp + fp),
        "recall": _safe(tp, tp + fn),
        "support": cm.sum(axis=1),
    }


def _nanmean(x: Sequence[float]) -> Optional[float]:
    arr = np.asarray(x, dtype=np.float64)
    return None if np.isnan(arr).all() else float(np.nanmean(arr))


def _f(x: float) -> Optional[float]:
    return None if (x is None or math.isnan(x)) else float(x)


def summarize(cm: np.ndarray) -> dict:
    """Overall, per-class and task-level (building / road) metrics."""
    pc = per_class(cm)
    out = {
        "pixel_accuracy": float(np.diag(cm).sum() / cm.sum()) if cm.sum() else None,
        "miou_all": _nanmean(pc["iou"]),
        "miou_foreground": _nanmean(pc["iou"][1:]),
        "macro_f1_all": _nanmean(pc["f1"]),
        "macro_f1_foreground": _nanmean(pc["f1"][1:]),
        "per_class": {
            CLASS_NAMES[i]: {k: _f(pc[k][i]) if k != "support" else int(pc[k][i])
                             for k in ("iou", "f1", "precision", "recall", "support")}
            for i in range(len(cm))
        },
        "tasks": {},
    }
    for name, ids in TASKS.items():
        out["tasks"][name] = {
            "miou": _nanmean([pc["iou"][i] for i in ids]),
            "f1": _nanmean([pc["f1"][i] for i in ids]),
        }
    return out


def aggregate_runs(runs: List[dict]) -> dict:
    """Mean and sample std (ddof=1) of every numeric leaf across runs.

    With a single run std is None, so a one-run result can't pass for a
    3-seed result.
    """
    def walk(items):
        first = items[0]
        if isinstance(first, dict):
            return {k: walk([it[k] for it in items]) for k in first}
        vals = [v for v in items if isinstance(v, (int, float)) and v is not None]
        if len(vals) != len(items) or not vals:
            return None
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else None}
    if not runs:
        raise ValueError("no runs to aggregate")
    return walk(runs)
