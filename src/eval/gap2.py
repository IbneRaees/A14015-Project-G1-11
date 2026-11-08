"""Gap 2 calibration experiment: how well do per-pixel confidences match accuracy?

Systems scored on the same test pixels (all from the same E1 backbone):

  yolo_instance_conf  the winning instance's confidence at each pixel (the E1 detector's own score)
  softmax_raw         softmax head, T = 1
  softmax_ts          the same head with the temperature fitted on val
  evidential          Dirichlet head, p = alpha / S
  evidential_ts       the evidential head's probabilities with a temperature fitted on val
                      (p -> softmax(log p / T)); extra control, because evidential heads trained with a
                      KL term tend to be underconfident and a fair reviewer will ask

Two pixel sets:
  covered  pixels some YOLO instance claims (conf >= map_conf). All systems are scored here, so the
           comparison is like for like. This is the primary `ece` used in the gap-resolution table.
  all      every pixel; only the heads (YOLO makes no claim on uncovered pixels). `ece_all`.

Per system: ECE (15 equal-width bins), accuracy, mean confidence, NLL and Brier score (proper scoring rules,
heads only), and the area under the risk-coverage curve using entropy (and vacuity for the evidential
head) as the uncertainty. ECE and risk-coverage are separate claims: ECE says whether confidence matches
accuracy, risk-coverage says whether uncertainty ranks errors.

Per-tile bin statistics are kept so ECE gets a bootstrap CI and paired comparisons over tiles.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from src.components.calibration import (CalibrationAccumulator, RiskCoverageAccumulator, aurc, softmax_np)

SYSTEMS = ("yolo_instance_conf", "softmax_raw", "softmax_ts", "evidential", "evidential_ts")
N_BINS = 15


def _entropy_norm(p: np.ndarray) -> np.ndarray:
    return np.clip(-(p * np.log(np.clip(p, 1e-12, None))).sum(0) / math.log(p.shape[0]), 0, 1)


def head_probabilities(alpha: np.ndarray, logits: np.ndarray, t_softmax: float, t_evidential: float) -> Dict[str, np.ndarray]:
    """(K, H, W) class probabilities of each head system."""
    p_ev = alpha / alpha.sum(0, keepdims=True)
    logp = np.log(np.clip(p_ev, 1e-12, None))
    return {
        "softmax_raw": softmax_np(logits, 1.0, axis=0).astype(np.float32),
        "softmax_ts": softmax_np(logits, t_softmax, axis=0).astype(np.float32),
        "evidential": p_ev.astype(np.float32),
        "evidential_ts": softmax_np(logp, t_evidential, axis=0).astype(np.float32),
    }


class _SystemStats:
    def __init__(self):
        self.cal = {s: CalibrationAccumulator(N_BINS) for s in ("covered", "all")}
        self.rc_ent = {s: RiskCoverageAccumulator() for s in ("covered", "all")}
        self.rc_vac = {s: RiskCoverageAccumulator() for s in ("covered", "all")}
        self.n = {"covered": 0, "all": 0}
        self.correct = {"covered": 0.0, "all": 0.0}
        self.conf = {"covered": 0.0, "all": 0.0}
        self.nll = {"covered": 0.0, "all": 0.0}
        self.brier = {"covered": 0.0, "all": 0.0}
        self.has_dist = False
        self.has_vac = False


class Gap2Scorer:
    """Accumulates calibration statistics over tiles for all five systems."""

    def __init__(self):
        self.stats = {s: _SystemStats() for s in SYSTEMS}

    def update(self, gt: np.ndarray, alpha: np.ndarray, logits: np.ndarray, yolo_label: np.ndarray,
               yolo_conf: np.ndarray, covered: np.ndarray, t_softmax: float, t_evidential: float
               ) -> Dict[str, np.ndarray]:
        """Add one tile. All maps share one grid (the head's output grid). Returns per-tile covered-set bins."""
        probs = head_probabilities(alpha, logits, t_softmax, t_evidential)
        vac = (alpha.shape[0] / alpha.sum(0)).astype(np.float32)
        tile_bins: Dict[str, np.ndarray] = {}
        sets = {"covered": covered, "all": np.ones_like(covered)}

        def add(name: str, pred, conf, unc_ent, p=None, unc_vac=None):
            st = self.stats[name]
            ok = pred == gt
            if p is not None:
                st.has_dist = True
                p_true = np.take_along_axis(p, gt[None].astype(np.int64), axis=0)[0]
                nll = -np.log(np.clip(p_true, 1e-12, None))
                brier = (p ** 2).sum(0) - 2 * p_true + 1
            if unc_vac is not None:
                st.has_vac = True
            for key, mask in sets.items():
                if key == "all" and name == "yolo_instance_conf":
                    continue
                m = mask
                if not m.any():
                    continue
                c, o = conf[m], ok[m]
                st.cal[key].update(c, o)
                st.rc_ent[key].update(unc_ent[m], ~o)
                if unc_vac is not None:
                    st.rc_vac[key].update(unc_vac[m], ~o)
                st.n[key] += int(m.sum())
                st.correct[key] += float(o.sum())
                st.conf[key] += float(c.sum())
                if p is not None:
                    st.nll[key] += float(nll[m].sum())
                    st.brier[key] += float(brier[m].sum())
            tmp = CalibrationAccumulator(N_BINS)
            if covered.any():
                tmp.update(conf[covered], ok[covered])
            tile_bins[name] = np.stack([tmp.count, tmp.conf_sum, tmp.correct_sum], axis=1)

        add("yolo_instance_conf", yolo_label, yolo_conf, 1.0 - yolo_conf)
        for name, p in probs.items():
            add(name, p.argmax(0), p.max(0), _entropy_norm(p), p,
                vac if name == "evidential" else None)
        return tile_bins

    def summarize(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        total_all = max(self.stats["evidential"].n["all"], 1)
        for name, st in self.stats.items():
            def get(key: str) -> Optional[dict]:
                n = st.n[key]
                if n == 0:
                    return None
                return {"ece": st.cal[key].ece(), "accuracy": st.correct[key] / n, "mean_confidence": st.conf[key] / n,
                        "nll": st.nll[key] / n if st.has_dist else None,
                        "brier": st.brier[key] / n if st.has_dist else None,
                        "aurc_entropy": aurc(*st.rc_ent[key].curve()),
                        "aurc_vacuity": aurc(*st.rc_vac[key].curve()) if st.has_vac else None, "n_pixels": n}
            cov, allp = get("covered"), get("all")
            if cov is None:
                out[name] = {"ece": None}
                continue
            out[name] = {**cov, "ece_all": None if allp is None else allp["ece"],
                         "accuracy_all": None if allp is None else allp["accuracy"],
                         "nll_all": None if allp is None else allp["nll"],
                         "coverage_fraction": cov["n_pixels"] / total_all,
                         "reliability": st.cal["covered"].reliability()}
        return out


def plot_reliability(result: dict, path) -> None:
    """Reliability diagrams of the primary (covered-pixel) calibration, one panel per system."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = result["runs"]
    names = [n for n in SYSTEMS if n in runs[0]["systems"] and runs[0]["systems"][n].get("reliability")]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        rel = runs[0]["systems"][name]["reliability"]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.plot(rel["mean_confidence"], rel["accuracy"], "o-", ms=3)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("confidence")
    np.atleast_1d(axes)[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
