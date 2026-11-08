"""Grid-cell protocol: one scoring scheme for E1, E2 and E3.

A VLM answers in text, not masks, so pixel mIoU cannot compare it with a
segmentation model. Every system is instead reduced to the same cell labels:
the tile is cut into rows x cols cells, and for each cell and each task
(building, road) the state is

    ABSENT   fewer than `min_frac` of the cell's pixels belong to the task
    DRY      present, and non-flooded pixels are the majority
    FLOODED  present, and flooded pixels are the majority

Ground truth comes from the FloodNet masks, predictions from a predicted label
map (E1, E3) or from the VLM's answers (E2). Cell thresholds are fixed here,
before any result exists, because they define the ground truth.

Scores per task: F1 of the DRY and FLOODED states, their mean (`macro_f1`),
presence F1 (present vs absent) and accuracy. Per-tile 3x3 confusion matrices
are kept so systems can be compared with a paired bootstrap over tiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.utils.metrics import per_class

ABSENT, DRY, FLOODED = 0, 1, 2
STATE_NAMES = ("absent", "dry", "flooded")
TASK_NAMES = ("building", "road")
# FloodNet ids: building-flooded 1, building-non-flooded 2, road-flooded 3, road-non-flooded 4
TASK_IDS = {"building": (1, 2), "road": (3, 4)}      # (flooded id, dry id)


@dataclass(frozen=True)
class CellConfig:
    rows: int = 4
    cols: int = 6
    building_min_frac: float = 0.05
    road_min_frac: float = 0.03

    def min_frac(self, task: str) -> float:
        return self.building_min_frac if task == "building" else self.road_min_frac


def cell_edges(n_pixels: int, n_cells: int) -> np.ndarray:
    return np.linspace(0, n_pixels, n_cells + 1).round().astype(int)


def cell_states(label_map: np.ndarray, cfg: CellConfig = CellConfig()) -> Dict[str, np.ndarray]:
    """{task: (rows, cols) int8 array of ABSENT/DRY/FLOODED} from a class-id map of any resolution."""
    h, w = label_map.shape
    re, ce = cell_edges(h, cfg.rows), cell_edges(w, cfg.cols)
    out = {t: np.zeros((cfg.rows, cfg.cols), np.int8) for t in TASK_NAMES}
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            sub = label_map[re[r]:re[r + 1], ce[c]:ce[c + 1]]
            counts = np.bincount(sub.ravel(), minlength=10)
            area = max(sub.size, 1)
            for task in TASK_NAMES:
                fid, did = TASK_IDS[task]
                nf, nd = counts[fid], counts[did]
                if (nf + nd) / area < cfg.min_frac(task):
                    continue
                out[task][r, c] = FLOODED if nf > nd else DRY
    return out


def cell_boxes_for(shape: Tuple[int, int], cfg: CellConfig = CellConfig()):
    """[x0, y0, x1, y1] of every cell for an image of `shape`, row-major."""
    h, w = shape
    re, ce = cell_edges(h, cfg.rows), cell_edges(w, cfg.cols)
    return [[int(ce[c]), int(re[r]), int(ce[c + 1]), int(re[r + 1])]
            for r in range(cfg.rows) for c in range(cfg.cols)]


def tile_confusions(gt: Dict[str, np.ndarray], pred: Dict[str, np.ndarray]) -> np.ndarray:
    """(2, 3, 3) confusion matrices, task order = TASK_NAMES, rows = ground truth."""
    cms = []
    for t in TASK_NAMES:
        idx = gt[t].astype(np.int64).ravel() * 3 + pred[t].astype(np.int64).ravel()
        cms.append(np.bincount(idx, minlength=9).reshape(3, 3))
    return np.stack(cms)


def tile_error_rate(gt: Dict[str, np.ndarray], pred: Dict[str, np.ndarray]) -> float:
    """Share of wrong cells among cells where either side says the task is present (0 if none)."""
    wrong = relevant = 0
    for t in TASK_NAMES:
        rel = (gt[t] != ABSENT) | (pred[t] != ABSENT)
        relevant += int(rel.sum())
        wrong += int((rel & (gt[t] != pred[t])).sum())
    return wrong / relevant if relevant else 0.0


def _nan_to_none(x: float) -> Optional[float]:
    return None if np.isnan(x) else float(x)


def _mean_or_nan(vals: Sequence[float]) -> float:
    a = np.asarray(vals, np.float64)
    return float("nan") if np.isnan(a).all() else float(np.nanmean(a))


def task_scores(cm: np.ndarray) -> Dict[str, float]:
    """Metrics from one 3x3 confusion matrix (as floats, NaN where undefined)."""
    pc = per_class(cm)
    f1 = pc["f1"]
    present = np.array([[cm[0, 0], cm[0, 1:].sum()], [cm[1:, 0].sum(), cm[1:, 1:].sum()]], np.float64)
    pf1 = (2 * present[1, 1] / (2 * present[1, 1] + present[0, 1] + present[1, 0])
           if (2 * present[1, 1] + present[0, 1] + present[1, 0]) > 0 else float("nan"))
    return {
        "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
        "f1_dry": float(f1[DRY]), "f1_flooded": float(f1[FLOODED]),
        "macro_f1": _mean_or_nan([f1[DRY], f1[FLOODED]]),
        "presence_f1": float(pf1),
    }


def summarize_cells(cms: Dict[str, np.ndarray]) -> dict:
    out: dict = {}
    for t in TASK_NAMES:
        s = task_scores(cms[t])
        out[t] = {k: _nan_to_none(v) for k, v in s.items()}
        out[t]["support"] = {STATE_NAMES[i]: int(cms[t][i].sum()) for i in range(3)}
    out["mean_macro_f1"] = _nan_to_none(_mean_or_nan([task_scores(cms[t])["macro_f1"] for t in TASK_NAMES]))
    return out


class CellAccumulator:
    def __init__(self):
        self.cms = {t: np.zeros((3, 3), np.int64) for t in TASK_NAMES}

    def update(self, gt: Dict[str, np.ndarray], pred: Dict[str, np.ndarray]) -> np.ndarray:
        tile = tile_confusions(gt, pred)
        for i, t in enumerate(TASK_NAMES):
            self.cms[t] += tile[i]
        return tile

    def summary(self) -> dict:
        return summarize_cells(self.cms)


# ---------------------------------------------------------------- bootstrap over tiles
def _metric(cms: np.ndarray, task: str, metric: str) -> float:
    return task_scores(cms[TASK_NAMES.index(task)])[metric]


def bootstrap_ci(per_tile: np.ndarray, task: str = "building", metric: str = "macro_f1",
                 n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean and 95% percentile CI of a cell metric, resampling tiles. per_tile: (N, 2, 3, 3)."""
    per_tile = np.asarray(per_tile)
    rng = np.random.default_rng(seed)
    n = len(per_tile)
    vals = np.array([_metric(per_tile[rng.integers(0, n, n)].sum(0), task, metric) for _ in range(n_boot)])
    vals = vals[~np.isnan(vals)]
    return {"point": _nan_to_none(_metric(per_tile.sum(0), task, metric)),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None}


def paired_bootstrap_diff(per_tile_a: np.ndarray, per_tile_b: np.ndarray, task: str = "building",
                          metric: str = "macro_f1", n_boot: int = 1000, seed: int = 0) -> dict:
    """metric(B) - metric(A) on the SAME resampled tiles; both arrays are (N, 2, 3, 3) over identical tile ids."""
    a, b = np.asarray(per_tile_a), np.asarray(per_tile_b)
    if a.shape != b.shape:
        raise ValueError("systems must be scored on the same tiles")
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = _metric(b[idx].sum(0), task, metric) - _metric(a[idx].sum(0), task, metric)
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    return {"diff_point": _nan_to_none(_metric(b.sum(0), task, metric) - _metric(a.sum(0), task, metric)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_b_better": float((diffs > 0).mean())}
