"""Coordinating Verifier Agent (Gap 2: uncalibrated visual overconfidence).

Takes the per-pixel Dirichlet parameters of a tile and decides whether the
automatic report can go out or a human has to look. The rules are
deterministic and every decision carries its reasons, so the trajectory logs
show why a tile was escalated.

One threshold on one uncertainty scale is used for all three checks:
  1. tile score: uncertainty averaged over building/road content,
  2. regions: grid cells whose mean uncertainty is above the threshold,
  3. road edges: graph edges (from the geospatial agent) whose centreline sits
     on uncertain pixels, i.e. a passable/blocked verdict resting on weak evidence.
Fit the threshold on validation tiles for a chosen coverage (`fit`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.components.evidential import MEASURES, dirichlet_stats, uncertainty_map
from src.components.calibration import fit_threshold_for_coverage
from src.components.grid import cell_boxes, cell_scores
from src.state import Disposition, TileState, UncertaintyReport, Verdict


@dataclass
class VerifierConfig:
    measure: str = "vacuity"                    # vacuity | entropy | epistemic
    threshold: Optional[float] = None           # fit on val; no default on purpose
    grid: int = 8
    relevant_classes: Tuple[int, ...] = (1, 2, 3, 4)   # FloodNet building and road classes
    max_uncertain_edges: int = 1                # this many uncertain road edges triggers review
    max_regions: Optional[int] = None


class VerifierAgent:
    def __init__(self, config: Optional[VerifierConfig] = None):
        self.cfg = config or VerifierConfig()
        if self.cfg.measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}")

    # -- scoring ---------------------------------------------------------
    def _maps(self, alpha: np.ndarray):
        stats = dirichlet_stats(alpha)
        u = uncertainty_map(stats, self.cfg.measure)
        # soft relevance: probability mass on building/road classes, so pixels the
        # model is vague about still count and confident grass/trees do not
        rel = stats["relevant_p"][list(self.cfg.relevant_classes)].sum(axis=0)
        return stats, u, rel

    @staticmethod
    def _weighted(u: np.ndarray, rel: np.ndarray) -> float:
        total = float(rel.sum())
        return float((u * rel).sum() / total) if total > 1e-9 else 0.0

    def tile_uncertainty(self, alpha: np.ndarray) -> float:
        _, u, rel = self._maps(alpha)
        return self._weighted(u, rel)

    def fit(self, val_tile_uncertainties: Sequence[float], target_coverage: float) -> float:
        """Pick the threshold that lets `target_coverage` of validation tiles go out automatically."""
        self.cfg.threshold = fit_threshold_for_coverage(val_tile_uncertainties, target_coverage)
        return self.cfg.threshold

    # -- decision --------------------------------------------------------
    def assess(self, alpha: np.ndarray, road_result=None,
               orig_shape: Optional[Tuple[int, int]] = None) -> Tuple[UncertaintyReport, Verdict]:
        thr = self.cfg.threshold
        if thr is None:
            raise RuntimeError("verifier threshold is not set; call fit() on validation tiles first")
        stats, u, rel = self._maps(alpha)
        shape = orig_shape or u.shape
        tile_u = self._weighted(u, rel)

        cells = cell_scores(u, self.cfg.grid)
        regions = cell_boxes(cells, thr, shape, self.cfg.max_regions)

        edges: List[dict] = []
        if road_result is not None:
            edges = [e for e in self._edge_uncertainties(road_result, u, shape) if e["uncertainty"] >= thr]

        report = UncertaintyReport(
            measure=self.cfg.measure, threshold=thr, tile_uncertainty=tile_u,
            mean_confidence=float(stats["confidence"].mean()), relevant_mass=float(rel.mean()),
            flagged_regions=regions, n_cells_flagged=int((cells >= thr).sum()), uncertain_edges=edges)

        reasons: List[str] = []
        if tile_u >= thr:
            reasons.append(f"tile uncertainty {tile_u:.3f} >= {thr:.3f}")
        if len(edges) >= self.cfg.max_uncertain_edges:
            n_pass = sum(e["passable"] for e in edges)
            reasons.append(f"{len(edges)} road edge(s) rest on uncertain evidence ({n_pass} marked passable)")
        verdict = Verdict(Disposition.HUMAN_REVIEW if reasons else Disposition.AUTO_REPORT, reasons)
        return report, verdict

    @staticmethod
    def _edge_uncertainties(road_result, u_map: np.ndarray, orig_shape: Tuple[int, int]) -> List[dict]:
        """Mean uncertainty along each road edge. Edge pixels are at processing scale; map them to the u_map grid."""
        sx, sy = road_result.scale
        h, w = orig_shape
        vh, vw = u_map.shape
        out = []
        for a, b, k, d in road_result.graph.edges(keys=True, data=True):
            px = d["pixels"]
            x = np.clip((px[:, 0] / sx * vw / w).astype(int), 0, vw - 1)
            y = np.clip((px[:, 1] / sy * vh / h).astype(int), 0, vh - 1)
            out.append({"u": int(a), "v": int(b), "key": int(k),
                        "uncertainty": float(u_map[y, x].mean()), "passable": bool(d["passable"])})
        return out

    # -- pipeline entry --------------------------------------------------
    def run(self, state: TileState, road_result=None) -> TileState:
        """Reads state.arrays['alpha'] (K, H, W); writes state.uncertainty and state.verdict."""
        with state.step("verifier.assess") as rec:
            report, verdict = self.assess(state.arrays["alpha"], road_result,
                                          (state.tile.height, state.tile.width))
            state.uncertainty, state.verdict = report, verdict
            rec["summary"] = (f"{verdict.disposition.value}; tile_u={report.tile_uncertainty:.3f}; "
                              f"cells={report.n_cells_flagged}; edges={len(report.uncertain_edges)}")
        return state
