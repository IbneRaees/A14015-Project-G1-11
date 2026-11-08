"""Dynamic Model Router (Gap 3: latency-accuracy tradeoff).

The fast path (fine-tuned YOLOv8x-Seg) always runs first. The router looks at
an uncertainty map from that pass and decides whether the tile also needs the
heavy VLM, and which regions of it. Because the VLM is a second stage, an
escalated tile pays t_fast + t_vlm:

    E[latency] = t_fast + p_escalate * t_vlm * region_cost_frac

`max_escalation_rate` turns a latency target into the largest escalation rate
the router may use, and `fit_threshold` picks the entropy threshold that stays
inside that rate on the validation split.

Inputs are plain arrays, so the router does not care where the uncertainty
comes from (Dirichlet vacuity or entropy from the evidential head, or any
other per-pixel map in [0, 1]).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from src.state import Route, RoutingDecision
from .grid import cell_boxes, cell_scores

MODES = ("entropy", "always_fast", "always_vlm")


@dataclass
class RouterConfig:
    mode: str = "entropy"
    threshold: Optional[float] = None   # must be fitted on val (see fit_threshold); no default on purpose
    grid: int = 8                       # tile is split into grid x grid cells
    top_k: int = 4                      # tile score = mean of the k most uncertain cells
    max_regions: int = 8                # cap on regions handed to the VLM


def normalized_entropy(probs: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy of class probabilities divided by ln K, so the range is [0, 1]."""
    p = np.clip(np.asarray(probs, np.float32), eps, 1.0)
    k = p.shape[axis]
    return np.clip(-(p * np.log(p)).sum(axis=axis) / math.log(k), 0.0, 1.0)


def fit_threshold(scores: Sequence[float], target_rate: float) -> float:
    """Smallest threshold such that at most `target_rate` of `scores` are >= it.

    Ties are handled by moving the threshold just above the tied value, so the
    achieved rate never exceeds the target.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if len(s) == 0:
        raise ValueError("no scores")
    if not 0.0 <= target_rate <= 1.0:
        raise ValueError("target_rate must be in [0, 1]")
    k = int(math.floor(target_rate * len(s)))
    if k == 0:
        return float(np.nextafter(s[0], np.inf))
    tau = float(s[k - 1])
    if np.count_nonzero(s >= tau) > k:
        tau = float(np.nextafter(tau, np.inf))
    return tau


def achieved_rate(scores: Sequence[float], threshold: float) -> float:
    return float(np.mean(np.asarray(scores) >= threshold))


def expected_latency(t_fast: float, t_vlm: float, p_escalate: float,
                     region_cost_frac: float = 1.0) -> float:
    """Mean seconds per tile for the fast-then-VLM cascade."""
    return t_fast + p_escalate * t_vlm * region_cost_frac


def max_escalation_rate(t_fast: float, t_vlm: float, target_latency: float,
                        region_cost_frac: float = 1.0) -> float:
    """Largest escalation rate that keeps the mean latency at or under the target."""
    if t_fast >= target_latency:
        return 0.0
    cost = t_vlm * region_cost_frac
    if cost <= 0:                      # escalation measured as free (or noise made it negative): no limit
        return 1.0
    return float(min(1.0, max(0.0, (target_latency - t_fast) / cost)))


class DynamicRouter:
    def __init__(self, config: Optional[RouterConfig] = None):
        self.cfg = config or RouterConfig()
        if self.cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

    def score(self, entropy_map: np.ndarray) -> Tuple[float, np.ndarray]:
        """Tile score = mean of the top-k cell entropies, plus the cell grid."""
        cells = cell_scores(entropy_map, self.cfg.grid)
        k = min(self.cfg.top_k, cells.size)
        top = np.sort(cells.ravel())[::-1][:k]
        return float(top.mean()), cells

    def fit(self, val_scores: Sequence[float], target_rate: float) -> float:
        self.cfg.threshold = fit_threshold(val_scores, target_rate)
        return self.cfg.threshold

    def route(self, entropy_map: Optional[np.ndarray] = None,
              orig_shape: Optional[Tuple[int, int]] = None) -> RoutingDecision:
        mode = self.cfg.mode
        if mode == "always_fast":
            return RoutingDecision(Route.FAST, mode, reason="forced fast path (ablation)")
        if mode == "always_vlm":
            return RoutingDecision(Route.VLM, mode, reason="forced VLM on every tile (ablation)")
        if self.cfg.threshold is None:
            raise RuntimeError("router threshold is not set; call fit() on validation scores first")
        if entropy_map is None:
            raise ValueError("entropy mode needs an entropy_map")
        score, cells = self.score(entropy_map)
        thr = self.cfg.threshold
        if score < thr:
            return RoutingDecision(Route.FAST, mode, score, thr,
                                   reason=f"score {score:.3f} < {thr:.3f}")
        regions = cell_boxes(cells, thr, orig_shape or entropy_map.shape, self.cfg.max_regions)
        return RoutingDecision(Route.VLM, mode, score, thr, regions,
                               reason=f"score {score:.3f} >= {thr:.3f}; {len(regions)} region(s)")
