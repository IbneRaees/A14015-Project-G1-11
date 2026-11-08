"""Gap 2 ablation verifier: a fixed threshold on YOLO's own instance confidences.

Same interface and same fitting rule as the evidential verifier (threshold chosen
on val for a target coverage), so E5 changes the uncertainty signal and nothing else.

tile uncertainty = 1 - area-weighted mean confidence of the building/road instances.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from src.components.calibration import fit_threshold_for_coverage
from src.state import Disposition, UncertaintyReport, Verdict


class InstanceConfidenceVerifier:
    def __init__(self, threshold: Optional[float] = None, relevant_classes: Tuple[int, ...] = (1, 2, 3, 4)):
        self.threshold = threshold
        self.relevant = tuple(relevant_classes)

    def tile_uncertainty(self, class_ids: Sequence[int], confidences: Sequence[float],
                         areas: Sequence[float]) -> float:
        c, k, a = (np.asarray(x, np.float64) for x in (class_ids, confidences, areas))
        rel = np.isin(c, self.relevant)
        if not rel.any() or a[rel].sum() <= 0:
            return 0.0
        return float(1.0 - np.average(k[rel], weights=a[rel]))

    def fit(self, val_tile_uncertainties: Sequence[float], target_coverage: float) -> float:
        self.threshold = fit_threshold_for_coverage(val_tile_uncertainties, target_coverage)
        return self.threshold

    def assess(self, class_ids: Sequence[int], confidences: Sequence[float], areas: Sequence[float],
               image_area: float) -> Tuple[UncertaintyReport, Verdict]:
        if self.threshold is None:
            raise RuntimeError("verifier threshold is not set; call fit() on validation tiles first")
        u = self.tile_uncertainty(class_ids, confidences, areas)
        c, k, a = (np.asarray(x, np.float64) for x in (class_ids, confidences, areas))
        mean_conf = float(np.average(k, weights=a)) if len(k) and a.sum() > 0 else 0.0
        rel_mass = float(a[np.isin(c, self.relevant)].sum() / image_area) if len(a) and image_area > 0 else 0.0
        report = UncertaintyReport("instance_conf", self.threshold, u, mean_conf, rel_mass)
        reasons = [f"tile uncertainty {u:.3f} >= {self.threshold:.3f}"] if u >= self.threshold else []
        return report, Verdict(Disposition.HUMAN_REVIEW if reasons else Disposition.AUTO_REPORT, reasons)
