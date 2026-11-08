"""Calibration and selective-prediction metrics (Gap 2).

ECE measures whether confidence matches accuracy. Risk-coverage measures what
abstention buys. They are separate claims: abstaining on uncertain tiles
lowers the error of what is reported, but it does not change the model's ECE.

Both accumulators work on batches, so a full 448-tile test set can be scored
without holding every pixel in memory.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


class CalibrationAccumulator:
    """Streaming Expected Calibration Error with equal-width confidence bins."""

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.count = np.zeros(n_bins, np.int64)
        self.conf_sum = np.zeros(n_bins, np.float64)
        self.correct_sum = np.zeros(n_bins, np.float64)

    def update(self, confidence: np.ndarray, correct: np.ndarray) -> None:
        c = np.asarray(confidence, np.float64).ravel()
        ok = np.asarray(correct, np.float64).ravel()
        if c.shape != ok.shape:
            raise ValueError("confidence and correct must have the same size")
        idx = np.minimum((c * self.n_bins).astype(np.int64), self.n_bins - 1)
        self.count += np.bincount(idx, minlength=self.n_bins)
        self.conf_sum += np.bincount(idx, weights=c, minlength=self.n_bins)
        self.correct_sum += np.bincount(idx, weights=ok, minlength=self.n_bins)

    def ece(self) -> float:
        n = self.count.sum()
        if n == 0:
            raise ValueError("no samples")
        used = self.count > 0
        gap = np.abs(self.correct_sum[used] - self.conf_sum[used]) / n   # = (n_b/N) * |acc_b - conf_b|
        return float(gap.sum())

    def reliability(self) -> Dict[str, list]:
        used = self.count > 0
        return {"bin_count": self.count[used].tolist(),
                "mean_confidence": (self.conf_sum[used] / self.count[used]).tolist(),
                "accuracy": (self.correct_sum[used] / self.count[used]).tolist()}


def risk_coverage_curve(uncertainty: Sequence[float], error: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Exact curve for a modest number of units (e.g. tiles): accept the least uncertain first.

    Returns (coverage, risk) where risk is the mean error among accepted units.
    """
    u = np.asarray(uncertainty, np.float64)
    e = np.asarray(error, np.float64)
    order = np.argsort(u, kind="stable")
    cum_err = np.cumsum(e[order])
    n = np.arange(1, len(u) + 1)
    return n / len(u), cum_err / n


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better), integrated from coverage 0."""
    cov = np.concatenate([[0.0], coverage])
    rsk = np.concatenate([[risk[0]], risk])
    return float(np.sum(np.diff(cov) * (rsk[1:] + rsk[:-1]) / 2))


def risk_at_coverage(coverage: np.ndarray, risk: np.ndarray, target: float) -> float:
    """Risk when the least uncertain `target` fraction is accepted."""
    i = int(np.searchsorted(coverage, target - 1e-12))
    return float(risk[min(i, len(risk) - 1)])


class RiskCoverageAccumulator:
    """Streaming risk-coverage over many pixels, with uncertainty quantised into bins in [0, 1]."""

    def __init__(self, n_bins: int = 1000):
        self.n_bins = n_bins
        self.total = np.zeros(n_bins, np.int64)
        self.errors = np.zeros(n_bins, np.int64)

    def update(self, uncertainty: np.ndarray, error: np.ndarray) -> None:
        u = np.clip(np.asarray(uncertainty, np.float64).ravel(), 0.0, 1.0)
        e = np.asarray(error).ravel().astype(bool)
        idx = np.minimum((u * self.n_bins).astype(np.int64), self.n_bins - 1)
        self.total += np.bincount(idx, minlength=self.n_bins)
        self.errors += np.bincount(idx[e], minlength=self.n_bins)

    def curve(self) -> Tuple[np.ndarray, np.ndarray]:
        used = self.total > 0
        n = np.cumsum(self.total)[used]
        err = np.cumsum(self.errors)[used]
        return n / self.total.sum(), err / n


def fit_threshold_for_coverage(uncertainties: Sequence[float], target_coverage: float) -> float:
    """Threshold t such that accepting u <= t keeps at least `target_coverage` of the units."""
    u = np.sort(np.asarray(uncertainties, np.float64))
    if len(u) == 0:
        raise ValueError("no uncertainties")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    k = max(1, int(math.ceil(target_coverage * len(u))))
    return float(u[k - 1])


# --------------------------------------------------------------------------
# Temperature scaling (post-hoc calibration baseline for the softmax head)
# --------------------------------------------------------------------------
def softmax_np(logits: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, np.float64) / temperature
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    bounds: Tuple[float, float] = (0.05, 20.0)) -> float:
    """Scalar T minimising the negative log-likelihood of softmax(logits / T).

    logits: (N, K) from a validation split, labels: (N,) class ids. Fit on val only.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    z = np.asarray(logits, np.float64)
    y = np.asarray(labels, np.int64)
    if z.ndim != 2 or len(z) != len(y):
        raise ValueError("logits must be (N, K) and labels (N,)")
    idx = np.arange(len(y))

    def nll(log_t: float) -> float:
        s = z / math.exp(log_t)
        return float(-(s[idx, y] - logsumexp(s, axis=1)).mean())

    res = minimize_scalar(nll, bounds=(math.log(bounds[0]), math.log(bounds[1])), method="bounded")
    return float(math.exp(res.x))


# --------------------------------------------------------------------------
# Bin-level ECE, for bootstrapping over tiles
# --------------------------------------------------------------------------
def ece_from_bins(bins: np.ndarray) -> float:
    """ECE from summed bin statistics. bins[..., b, :] = (count, sum of confidence, sum of correct)."""
    b = np.asarray(bins, np.float64).reshape(-1, bins.shape[-2], 3).sum(0)
    n = b[:, 0].sum()
    if n == 0:
        return float("nan")
    return float(np.abs(b[:, 2] - b[:, 1]).sum() / n)


def bootstrap_ece_ci(per_tile_bins: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean and 95% CI of ECE, resampling tiles. per_tile_bins: (N, n_bins, 3)."""
    pt = np.asarray(per_tile_bins, np.float64)
    rng = np.random.default_rng(seed)
    vals = np.array([ece_from_bins(pt[rng.integers(0, len(pt), len(pt))].sum(0)) for _ in range(n_boot)])
    vals = vals[~np.isnan(vals)]
    point = ece_from_bins(pt.sum(0))
    return {"point": None if np.isnan(point) else point,
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None}


def paired_bootstrap_ece(per_tile_a: np.ndarray, per_tile_b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """ECE(B) - ECE(A) on the same resampled tiles. Negative means B is better calibrated."""
    a, b = np.asarray(per_tile_a, np.float64), np.asarray(per_tile_b, np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError("systems must be scored on the same tiles")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        ea, eb = ece_from_bins(a[idx].sum(0)), ece_from_bins(b[idx].sum(0))
        if not (np.isnan(ea) or np.isnan(eb)):
            diffs.append(eb - ea)
    pa, pb = ece_from_bins(a.sum(0)), ece_from_bins(b.sum(0))
    if not diffs or np.isnan(pa) or np.isnan(pb):
        return {"diff_point": None, "ci95": None, "p_b_lower": None}
    d = np.asarray(diffs)
    return {"diff_point": float(pb - pa), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_b_lower": float((d < 0).mean())}
