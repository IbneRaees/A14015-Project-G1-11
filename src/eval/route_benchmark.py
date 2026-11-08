"""Gap 1 route-query benchmark.

Question asked of every system: "is there a passable route between point A and
point B?" The answer must come from the system's own output, and the ground truth
comes from the FloodNet mask alone.

Ground truth. Take the ground-truth road mask (classes 3 and 4), clean it with the
same preprocessing the geospatial agent uses (downscale to <= 1024 px, close small
gaps, drop specks) and label 8-connected components twice: once over all road pixels
and once over non-flooded road pixels (class 4). The ground truth is pixel
connectivity, not the agent's graph, so the benchmark is not circular.

Queries. Points are drawn from the centreline of the ground-truth dry road, so both
ends have a defined passable component. Three kinds of pair are sampled, up to
`pairs_per_type` per tile, with a fixed seed so every system answers the same set:

    passable      same dry component                                     truth: yes
    blocked       different dry components, same road component         truth: no  (flooding cuts it)
    disconnected  different road components                              truth: no

Systems answer from a label map (an E1 or E3 prediction, or the ground truth as an
oracle) in two ways: `pixel_cc` (connected components of predicted dry road, same
preprocessing) and `graph` (the geospatial agent's road graph). A third system asks a
VLM directly about the image with the two points marked. A point further than
`snap_frac` x the image's long side from any predicted road cannot be answered; it
counts as "no route" and is reported as unanswerable.

Metrics: accuracy / connectivity error, false-passable rate (a blocked or disconnected
pair answered "yes", the dangerous error), false-blocked rate, and per-type accuracy.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.agents.geospatial_agent import (GeospatialConfig, GeospatialReasoningAgent, preprocess_mask,
                                         skeletonize)

DRY_ROAD, FLOODED_ROAD = 4, 3
TYPES = ("passable", "blocked", "disconnected")
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}


@dataclass
class QueryConfig:
    pairs_per_type: int = 4
    min_dist_frac: float = 0.15     # endpoints are at least this fraction of the long side apart
    snap_frac: float = 0.02         # answer tolerance, as a fraction of the long side
    max_attempts: int = 400
    seed: int = 0
    geo: GeospatialConfig = field(default_factory=GeospatialConfig)


def _tile_int(tile_id: str) -> int:
    s = str(tile_id)
    return int(s) if s.isdigit() else zlib.crc32(s.encode())


# --------------------------------------------------------------------- ground truth
def generate_queries(label: np.ndarray, tile_id: str, cfg: QueryConfig) -> List[dict]:
    """Deterministic query set for one tile, from the ground-truth label map only."""
    h, w = label.shape
    dry, sx, sy = preprocess_mask(label == DRY_ROAD, cfg.geo)
    road, _, _ = preprocess_mask(np.isin(label, (DRY_ROAD, FLOODED_ROAD)), cfg.geo)
    if not dry.any():
        return []
    _, dry_lab = cv2.connectedComponents(dry, connectivity=8)
    _, road_lab = cv2.connectedComponents(road, connectivity=8)
    ys, xs = np.nonzero(skeletonize(dry))
    keep = road_lab[ys, xs] > 0
    ys, xs = ys[keep], xs[keep]
    if len(xs) < 2:
        return []
    dl, rl = dry_lab[ys, xs], road_lab[ys, xs]
    dmin = cfg.min_dist_frac * max(dry.shape)
    rng = np.random.default_rng([cfg.seed, _tile_int(tile_id)])
    rules = {
        "passable": lambda i, j: dl[i] == dl[j],
        "blocked": lambda i, j: dl[i] != dl[j] and rl[i] == rl[j],
        "disconnected": lambda i, j: rl[i] != rl[j],
    }
    out: List[dict] = []
    for t in TYPES:
        found, seen = 0, set()
        for _ in range(cfg.max_attempts):
            if found >= cfg.pairs_per_type:
                break
            i, j = (int(v) for v in rng.integers(0, len(xs), 2))
            if i == j or (i, j) in seen:
                continue
            seen.add((i, j))
            if np.hypot(xs[i] - xs[j], ys[i] - ys[j]) < dmin or not rules[t](i, j):
                continue
            out.append({"tile_id": str(tile_id), "idx": len(out), "type": t,
                        "a": [round(float(xs[i]) / sx, 1), round(float(ys[i]) / sy, 1)],
                        "b": [round(float(xs[j]) / sx, 1), round(float(ys[j]) / sy, 1)],
                        "gt": t == "passable"})
            found += 1
    return out


def queries_fingerprint(queries: Sequence[dict]) -> str:
    return hashlib.sha256(json.dumps(list(queries), sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------- answerers
class PixelCCAnswerer:
    """Connected components of predicted dry road. The graph-free baseline."""

    def __init__(self, label: np.ndarray, orig_shape: Tuple[int, int], cfg: QueryConfig):
        clean, _, _ = preprocess_mask(label == DRY_ROAD, cfg.geo)
        _, self.lab = cv2.connectedComponents(clean, connectivity=8)
        H, W = orig_shape
        self.fx, self.fy = clean.shape[1] / W, clean.shape[0] / H
        self.radius = cfg.snap_frac * max(clean.shape)

    def _component(self, p) -> Optional[int]:
        x, y = p[0] * self.fx, p[1] * self.fy
        r = int(np.ceil(self.radius))
        x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
        sub = self.lab[y0:int(y) + r + 1, x0:int(x) + r + 1]
        ys, xs = np.nonzero(sub)
        if len(xs) == 0:
            return None
        d = np.hypot(xs + x0 - x, ys + y0 - y)
        k = int(d.argmin())
        return int(sub[ys[k], xs[k]]) if d[k] <= self.radius else None

    def __call__(self, a, b) -> Tuple[bool, bool]:
        ca, cb = self._component(a), self._component(b)
        if ca is None or cb is None:
            return False, False
        return ca == cb, True


class GraphAnswerer:
    """The geospatial agent's road graph, built from a predicted label map."""

    def __init__(self, label: np.ndarray, orig_shape: Tuple[int, int], tile_id: str, cfg: QueryConfig,
                 agent: Optional[GeospatialReasoningAgent] = None):
        self.res = (agent or GeospatialReasoningAgent(cfg.geo)).analyze_label_map(label, tile_id)
        lh, lw = label.shape
        H, W = orig_shape
        self.fx, self.fy = lw / W, lh / H
        self.res.snap_max_dist_px = cfg.snap_frac * max(lh, lw)

    def __call__(self, a, b) -> Tuple[bool, bool]:
        pa, pb = (a[0] * self.fx, a[1] * self.fy), (b[0] * self.fx, b[1] * self.fy)
        (e1, d1), (e2, d2) = self.res._nearest_edge(pa), self.res._nearest_edge(pb)
        if e1 is None or e2 is None or max(d1, d2) > self.res.snap_max_dist_px:
            return False, False
        return bool(self.res.route_exists(pa, pb, passable_only=True)), True


ROUTE_PROMPT = (
    "This is a post-disaster aerial photograph. A red circle marks point A and a blue circle marks point B, "
    "both on roads. Is there a continuous road route from A to B that is not covered by floodwater and not "
    "blocked by debris? Answer with one word: yes or no."
)
ROUTE_PROMPT_SHA = hashlib.sha256(ROUTE_PROMPT.encode()).hexdigest()[:12]


def parse_yes_no(text) -> Optional[bool]:
    if not isinstance(text, str):
        return None
    t = text.strip().lower()
    first = re.match(r"\W*(yes|no)\b", t)
    if first:
        return first.group(1) == "yes"
    yes, no = re.search(r"\byes\b", t), re.search(r"\bno\b", t)
    if bool(yes) != bool(no):
        return bool(yes)
    return None


def draw_markers(image_rgb: np.ndarray, a, b, max_side: int = 1024) -> np.ndarray:
    """Downscale to max_side and mark A (red) and B (blue) with white-ringed circles."""
    H, W = image_rgb.shape[:2]
    s = min(1.0, max_side / max(H, W))
    img = cv2.resize(image_rgb, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else image_rgb.copy()
    r = max(5, int(0.014 * max(img.shape[:2])))
    for p, colour in ((a, (255, 0, 0)), (b, (0, 0, 255))):
        c = (int(round(p[0] * s)), int(round(p[1] * s)))
        cv2.circle(img, c, r + 2, (255, 255, 255), -1)
        cv2.circle(img, c, r, colour, -1)
    return img


class VLMAnswerer:
    """Direct visual prompting: the base-paper style answer, no graph."""

    def __init__(self, vlm, image_rgb: np.ndarray, max_side: int = 1024, max_new_tokens: int = 8):
        self.vlm, self.image, self.max_side, self.max_new_tokens = vlm, image_rgb, max_side, max_new_tokens
        self.raw: List[str] = []

    def __call__(self, a, b) -> Tuple[bool, bool]:
        text = self.vlm.generate(draw_markers(self.image, a, b, self.max_side), ROUTE_PROMPT, self.max_new_tokens)
        self.raw.append(text)
        ans = parse_yes_no(text)
        return (False, False) if ans is None else (ans, True)


def answer_tile(queries: Sequence[dict], answerer: Callable) -> Tuple[np.ndarray, int]:
    """(3, 2) counts [type, answered_yes] and the number of unanswerable queries."""
    counts, unanswerable = np.zeros((len(TYPES), 2), np.int64), 0
    for q in queries:
        yes, answerable = answerer(q["a"], q["b"])
        counts[TYPE_INDEX[q["type"]], int(bool(yes))] += 1
        unanswerable += int(not answerable)
    return counts, unanswerable


# --------------------------------------------------------------------- metrics
def _rate(num: float, den: float) -> Optional[float]:
    return float(num / den) if den > 0 else None


def route_metrics(counts: np.ndarray) -> dict:
    """counts: (3, 2) = [passable, blocked, disconnected] x [answered no, answered yes]."""
    c = np.asarray(counts, np.float64)
    tp, fn = c[0, 1], c[0, 0]
    fp, tn = c[1, 1] + c[2, 1], c[1, 0] + c[2, 0]
    n = c.sum()
    prec, rec = _rate(tp, tp + fp), _rate(tp, tp + fn)
    return {
        "n_queries": int(n),
        "accuracy": _rate(tp + tn, n),
        "connectivity_error_rate": _rate(fp + fn, n),
        "false_passable_rate": _rate(fp, fp + tn),          # no-route pairs answered "yes"
        "false_blocked_rate": _rate(fn, tp + fn),           # route pairs answered "no"
        "precision": prec, "recall": rec,
        "f1": _rate(2 * tp, 2 * tp + fp + fn),
        "accuracy_passable": _rate(c[0, 1], c[0].sum()),
        "accuracy_blocked": _rate(c[1, 0], c[1].sum()),
        "accuracy_disconnected": _rate(c[2, 0], c[2].sum()),
        "false_passable_rate_blocked": _rate(c[1, 1], c[1].sum()),
        "false_passable_rate_disconnected": _rate(c[2, 1], c[2].sum()),
    }


def _metric_value(counts: np.ndarray, metric: str) -> float:
    v = route_metrics(counts)[metric]
    return float("nan") if v is None else v


def bootstrap_route_ci(per_tile: np.ndarray, metric: str = "connectivity_error_rate", n_boot: int = 1000,
                       seed: int = 0) -> dict:
    """Mean and 95% CI of a route metric, resampling tiles. per_tile: (N, 3, 2)."""
    per_tile = np.asarray(per_tile)
    has = per_tile.reshape(len(per_tile), -1).sum(1) > 0
    pt = per_tile[has]
    rng = np.random.default_rng(seed)
    vals = np.array([_metric_value(pt[rng.integers(0, len(pt), len(pt))].sum(0), metric) for _ in range(n_boot)]
                    if len(pt) else [])
    vals = vals[~np.isnan(vals)]
    point = _metric_value(pt.sum(0), metric) if len(pt) else float("nan")
    return {"point": None if np.isnan(point) else point,
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None}


def paired_bootstrap_route(per_tile_a: np.ndarray, per_tile_b: np.ndarray, metric: str = "false_passable_rate",
                           n_boot: int = 1000, seed: int = 0) -> dict:
    """Same resampled tiles for both systems. For error metrics lower is better.

    Reports diff = metric(B) - metric(A), the relative reduction (A - B) / A of B versus A, and the
    share of resamples in which B has the lower error.
    """
    a, b = np.asarray(per_tile_a), np.asarray(per_tile_b)
    if a.shape != b.shape:
        raise ValueError("systems must be scored on the same tiles and queries")
    has = (a.reshape(len(a), -1).sum(1) > 0) | (b.reshape(len(b), -1).sum(1) > 0)
    a, b = a[has], b[has]
    rng = np.random.default_rng(seed)
    diffs, rels = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        ma, mb = _metric_value(a[idx].sum(0), metric), _metric_value(b[idx].sum(0), metric)
        if np.isnan(ma) or np.isnan(mb):
            continue
        diffs.append(mb - ma)
        if ma > 0:
            rels.append((ma - mb) / ma)
    diffs, rels = np.asarray(diffs), np.asarray(rels)
    ma, mb = _metric_value(a.sum(0), metric), _metric_value(b.sum(0), metric)
    if len(diffs) == 0 or np.isnan(ma) or np.isnan(mb):       # metric undefined on this data
        return {"diff_point": None, "ci95": None, "relative_reduction_point": None,
                "relative_reduction_ci95": None, "p_b_lower_error": None}
    return {"diff_point": mb - ma, "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "relative_reduction_point": (ma - mb) / ma if ma > 0 else None,
            "relative_reduction_ci95": [float(np.percentile(rels, 2.5)), float(np.percentile(rels, 97.5))] if len(rels) else None,
            "p_b_lower_error": float((diffs < 0).mean())}


def config_dict(cfg: QueryConfig) -> dict:
    d = asdict(cfg)
    return d
