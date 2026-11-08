"""Geospatial Reasoning Agent (Gap 1: spatial topographic blindness).

Turns a binary road mask from a UAV tile into an explicit topological graph:

    mask -> preprocess -> skeletonize -> nodes/edges -> prune -> passability

Nodes are road intersections (junctions) and dead ends. Edges are road segments
between them. Each edge carries its length and the fraction of its centreline
that falls on a "blocked" mask (flooded road or debris), so route questions
become graph queries instead of free-text VLM answers.

Everything here is deterministic (no model calls), so it can be unit tested.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional

import cv2
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# FloodNet semantic label ids. Check these against the masks in data/ before
# trusting them; some releases ship colour-coded RGB masks instead of ids.
FLOODNET_ROAD_FLOODED = 3
FLOODNET_ROAD_NON_FLOODED = 4


@dataclass
class GeospatialConfig:
    max_side: int = 1024              # longest side used for processing (px)
    resize_threshold: float = 0.4     # area-fraction cutoff when downscaling a mask
    close_kernel: int = 5             # morphological closing, bridges small gaps (px, processing scale)
    min_component_area: int = 150     # drop road blobs smaller than this (processing px)
    spur_length_px: int = 12          # dead-end branches shorter than this are removed (processing px)
    min_blocked_run_px: int = 20      # an edge is impassable if it holds a contiguous blocked run this long
                                      # (or >= half of the edge, whichever is smaller); processing px
    snap_max_dist_px: float = 60.0    # max distance from a query point to the road (original px)


# --------------------------------------------------------------------------
# Low-level image helpers
# --------------------------------------------------------------------------
def _neighbor_stats(img: np.ndarray):
    """For every pixel of a 0/1 image return (B, A, neighbours).

    B = number of foreground 8-neighbours.
    A = number of 0->1 transitions walking clockwise around the neighbours
        (the crossing number: 1 = endpoint, 2 = path, >=3 = junction).
    neighbours = [p2..p9] clockwise starting from north.
    """
    p = np.pad(img, 1).astype(np.int16)
    n = [
        p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
        p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2],
    ]
    b = sum(n)
    a = sum(((n[i] == 0) & (n[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
    return b, a, n


def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    """Vectorised Zhang-Suen thinning. Fallback when cv2.ximgproc is missing."""
    img = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            b, a, n = _neighbor_stats(img)
            p2, p3, p4, p5, p6, p7, p8, p9 = n
            if step == 0:
                cond = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            remove = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if remove.any():
                img[remove] = 0
                changed = True
    return img


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide skeleton of a binary mask (uint8 0/1)."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    if hasattr(getattr(cv2, "ximgproc", None), "thinning"):
        thin = cv2.ximgproc.thinning(m * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return (thin > 0).astype(np.uint8)
    return zhang_suen_thinning(m)


def _resize_binary(mask: np.ndarray, new_w: int, new_h: int, thr: float) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    return (resized >= thr).astype(np.uint8)


def preprocess_mask(mask: np.ndarray, cfg: GeospatialConfig):
    """Binarise, downscale, close small gaps, drop tiny blobs.

    Returns (clean_mask, scale_x, scale_y) where scale = processing / original.
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    scale = min(1.0, cfg.max_side / max(h, w))
    if scale < 1.0:
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        m = _resize_binary(m, nw, nh, cfg.resize_threshold)
    else:
        nh, nw = h, w
    if cfg.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if cfg.min_component_area > 0 and m.any():
        _, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= cfg.min_component_area
        keep[0] = False
        m = keep[lab].astype(np.uint8)
    return m, nw / w, nh / h


# --------------------------------------------------------------------------
# Skeleton -> graph
# --------------------------------------------------------------------------
def skeleton_to_graph(skel: np.ndarray, blocked: np.ndarray) -> nx.MultiGraph:
    """Build a MultiGraph from a skeleton. Coordinates stay at processing scale.

    Node attrs: pos=(x, y), kind in {"junction", "endpoint"}.
    Edge attrs: pixels (n,2 array of x,y), is_blocked (per pixel), n_px, blocked_px, ambiguous.
    A MultiGraph is used because two roads can legitimately join the same pair
    of intersections (a ring road) and a loop can return to its own node.
    """
    s = skel.astype(np.uint8)
    g = nx.MultiGraph()
    if not s.any():
        return g

    _, a, _ = _neighbor_stats(s)
    endpoint = (s == 1) & (a <= 1)
    junction = (s == 1) & (a >= 3)
    node_mask = (endpoint | junction).astype(np.uint8)
    # Node = 3x3-dilated zone around each endpoint/junction pixel. Without the
    # dilation, the arms of a one-pixel junction touch diagonally once the
    # junction pixel is removed and merge into a single bogus segment.
    zone = cv2.dilate(node_mask, np.ones((3, 3), np.uint8))

    n_lab, node_lab, _, centroids = cv2.connectedComponentsWithStats(zone, connectivity=8)
    has_junction = np.bincount(node_lab[junction], minlength=n_lab) > 0
    for i in range(1, n_lab):
        g.add_node(
            i,
            pos=(float(centroids[i][0]), float(centroids[i][1])),
            kind="junction" if has_junction[i] else "endpoint",
        )

    seg_mask = ((s == 1) & (zone == 0)).astype(np.uint8)
    n_seg, seg_lab, stats, _ = cv2.connectedComponentsWithStats(seg_mask, connectivity=8)
    h, w = s.shape
    synthetic = n_lab
    dil_kernel = np.ones((3, 3), np.uint8)

    for i in range(1, n_seg):
        x, y, bw, bh = (int(stats[i, c]) for c in
                        (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1, y1 = min(w, x + bw + 1), min(h, y + bh + 1)
        sub = (seg_lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ys, xs = np.nonzero(sub)
        pixels = np.stack([xs + x0, ys + y0], axis=1)
        touching = np.unique(node_lab[y0:y1, x0:x1][cv2.dilate(sub, dil_kernel) > 0])
        touching = [int(t) for t in touching if t > 0]
        attrs = dict(
            pixels=pixels,
            n_px=len(pixels),
            is_blocked=blocked[pixels[:, 1], pixels[:, 0]].astype(bool),
            blocked_px=int(blocked[pixels[:, 1], pixels[:, 0]].sum()),
            ambiguous=len(touching) > 2,
        )
        if not touching:  # closed ring with no junction: anchor it on one pixel
            synthetic += 1
            g.add_node(synthetic, pos=(float(pixels[0, 0]), float(pixels[0, 1])), kind="junction")
            g.add_edge(synthetic, synthetic, **attrs)
        elif len(touching) == 1:  # loop hanging off a single node
            g.add_edge(touching[0], touching[0], **attrs)
        else:
            for u, v in combinations(touching, 2):
                g.add_edge(u, v, **attrs)
    return g


def _merge_edge_attrs(d1: dict, d2: dict) -> dict:
    return dict(
        pixels=np.concatenate([d1["pixels"], d2["pixels"]]),
        is_blocked=np.concatenate([d1["is_blocked"], d2["is_blocked"]]),
        n_px=d1["n_px"] + d2["n_px"],
        blocked_px=d1["blocked_px"] + d2["blocked_px"],
        ambiguous=d1["ambiguous"] or d2["ambiguous"],
    )


def contract_degree2(g: nx.MultiGraph) -> None:
    """Merge nodes of degree 2 (pure pass-through points) into single edges."""
    changed = True
    while changed:
        changed = False
        for n in list(g.nodes):
            if n not in g or g.degree(n) != 2 or g.has_edge(n, n):
                continue
            (_, a, _, d1), (_, b, _, d2) = list(g.edges(n, keys=True, data=True))
            g.add_edge(a, b, **_merge_edge_attrs(d1, d2))
            g.remove_node(n)
            changed = True


def prune_spurs(g: nx.MultiGraph, spur_px: int) -> int:
    """Remove short dead-end branches left by skeleton noise. Returns count removed."""
    removed = 0
    changed = True
    while changed:
        changed = False
        for u, v, k, d in list(g.edges(keys=True, data=True)):
            if u == v or u not in g or v not in g or not g.has_edge(u, v, key=k):
                continue
            if d["n_px"] >= spur_px:
                continue
            du, dv = g.degree(u), g.degree(v)
            if (du == 1 and dv >= 3) or (dv == 1 and du >= 3):
                leaf = u if du == 1 else v
                g.remove_edge(u, v, key=k)
                g.remove_node(leaf)
                removed += 1
                changed = True
        if changed:
            contract_degree2(g)
    return removed


def max_blocked_run(pixels: np.ndarray, is_blocked: np.ndarray) -> int:
    """Longest contiguous run of blocked centreline pixels along an edge.

    Blocked pixels are dilated 5x5 first so runs split by the few pixels lost
    at a contracted node are still counted as one run.
    """
    if not is_blocked.any():
        return 0
    pts = pixels[is_blocked]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    crop = np.zeros((y1 - y0 + 5, x1 - x0 + 5), np.uint8)
    crop[pts[:, 1] - y0 + 2, pts[:, 0] - x0 + 2] = 1
    n, lab = cv2.connectedComponents(cv2.dilate(crop, np.ones((5, 5), np.uint8)), connectivity=8)
    return int(np.bincount(lab[crop == 1], minlength=n)[1:].max())


def _order_along_path(pixels: np.ndarray, start_xy) -> np.ndarray:
    """Indices that sort an edge's pixels along the path, starting at the pixel nearest `start_xy`.

    Edge pixels form a one-pixel-wide path, so geodesic (8-connected) distance from one end
    increases monotonically along it.
    """
    n = len(pixels)
    index = {(int(x), int(y)): i for i, (x, y) in enumerate(pixels)}
    start = int(np.argmin(np.hypot(pixels[:, 0] - start_xy[0], pixels[:, 1] - start_xy[1])))
    dist = np.full(n, -1, np.int64)
    dist[start] = 0
    queue = deque([start])
    while queue:
        i = queue.popleft()
        x, y = int(pixels[i, 0]), int(pixels[i, 1])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                j = index.get((x + dx, y + dy))
                if (dx or dy) and j is not None and dist[j] < 0:
                    dist[j] = dist[i] + 1
                    queue.append(j)
    dist[dist < 0] = dist.max() + 1                     # unreachable pixels (should not happen) go last
    return np.argsort(dist, kind="stable")


def _flood_runs(blocked: np.ndarray, limit: int, bridge: int = 4) -> np.ndarray:
    """Smoothed per-pixel flood flags: bridge dry gaps of <= `bridge` pixels, drop runs shorter than `limit`."""
    b = blocked.copy()
    idx = np.flatnonzero(b)
    if len(idx) > 1:
        gaps = np.diff(idx)
        for k in np.flatnonzero((gaps > 1) & (gaps <= bridge + 1)):
            b[idx[k]:idx[k + 1]] = True
    out = np.zeros_like(b)
    bounds = np.flatnonzero(np.diff(np.concatenate([[0], b.astype(np.int8), [0]])))
    for start, end in zip(bounds[::2], bounds[1::2]):
        if end - start >= limit:
            out[start:end] = True
    return out


def split_edges_at_flood_boundaries(g: nx.MultiGraph, min_blocked_run_px: int) -> int:
    """Cut every partly flooded edge where the flood starts and ends.

    Without this a road that is dry, flooded, dry is one edge, so the dry stretches at its
    ends cannot be reached even though they are open. Edges that are fully dry, fully
    flooded or ambiguous are left alone. Returns the number of edges split.
    """
    split = 0
    for u, v, k, d in list(g.edges(keys=True, data=True)):
        if d["ambiguous"] or not d["is_blocked"].any() or d["is_blocked"].all():
            continue
        limit = min(min_blocked_run_px, max(1.0, 0.5 * d["n_px"]))
        order = _order_along_path(d["pixels"], g.nodes[u]["pos"])
        pix, flood = d["pixels"][order], _flood_runs(d["is_blocked"][order], limit)
        if not flood.any() or flood.all():
            continue
        cuts = np.flatnonzero(flood[1:] != flood[:-1]) + 1
        bounds = [0, *[int(c) for c in cuts], len(pix)]
        next_id = max(g.nodes) + 1
        chain = [u]
        for c in bounds[1:-1]:
            g.add_node(next_id, pos=(float(pix[c, 0]), float(pix[c, 1])), kind="split")
            chain.append(next_id)
            next_id += 1
        chain.append(v)
        g.remove_edge(u, v, key=k)
        for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
            g.add_edge(chain[i], chain[i + 1], pixels=pix[s:e], is_blocked=flood[s:e].copy(), n_px=e - s,
                       blocked_px=int(flood[s:e].sum()), ambiguous=False)
        split += 1
    return split


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class RoadGraphResult:
    tile_id: str
    graph: nx.MultiGraph                  # all roads
    passable_graph: nx.MultiGraph         # same nodes, blocked edges removed
    stats: dict
    trajectory: list = field(default_factory=list)
    scale: tuple = (1.0, 1.0)             # processing / original, (x, y)
    snap_max_dist_px: float = 60.0

    def _nearest_edge(self, xy):
        """Closest edge to a point given in original-resolution (x, y)."""
        sx, sy = self.scale
        q = np.array([xy[0] * sx, xy[1] * sy], dtype=np.float64)
        best, best_d = None, np.inf
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            dist = np.sqrt(((d["pixels"] - q) ** 2).sum(axis=1)).min()
            if dist < best_d:
                best, best_d = (u, v, k), dist
        if best is None:
            return None, np.inf
        return best, best_d / ((sx + sy) / 2)

    def route_exists(self, src_xy, dst_xy, passable_only: bool = True) -> bool:
        """True if two road points (original px, x/y) are connected.

        Each point snaps to its nearest road segment. With passable_only the
        segment itself must be passable and the connection must avoid blocked edges.
        """
        e1, d1 = self._nearest_edge(src_xy)
        e2, d2 = self._nearest_edge(dst_xy)
        if e1 is None or e2 is None or max(d1, d2) > self.snap_max_dist_px:
            return False
        if passable_only:
            if not (self.graph.edges[e1]["passable"] and self.graph.edges[e2]["passable"]):
                return False
            return nx.has_path(self.passable_graph, e1[0], e2[0])
        return nx.has_path(self.graph, e1[0], e2[0])

    def to_dict(self) -> dict:
        sx, sy = self.scale
        return {
            "tile_id": self.tile_id,
            "stats": self.stats,
            "nodes": [
                {"id": int(n), "x": d["pos"][0] / sx, "y": d["pos"][1] / sy, "kind": d["kind"]}
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "u": int(u), "v": int(v),
                    "length_px": d["length_px"],
                    "blocked_fraction": d["blocked_fraction"],
                    "passable": d["passable"],
                }
                for u, v, _, d in self.graph.edges(keys=True, data=True)
            ],
            "trajectory": self.trajectory,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class GeospatialReasoningAgent:
    """Builds the road graph and answers connectivity/passability questions.

    Each pipeline stage is recorded in `trajectory` (tool name, duration, a
    one-line summary) and optionally written to logs/ as JSONL, which feeds the
    trajectory-based failure analysis later.
    """

    def __init__(self, config: Optional[GeospatialConfig] = None, log_dir: Optional[str] = None):
        self.cfg = config or GeospatialConfig()
        self.log_dir = Path(log_dir) if log_dir else None

    # -- convenience entry point for FloodNet label maps ---------------------
    def analyze_label_map(self, label_map: np.ndarray, tile_id: str = "tile") -> RoadGraphResult:
        road = np.isin(label_map, (FLOODNET_ROAD_FLOODED, FLOODNET_ROAD_NON_FLOODED))
        blocked = label_map == FLOODNET_ROAD_FLOODED
        return self.analyze(road, blocked, tile_id)

    def analyze(self, road_mask: np.ndarray, blocked_mask: Optional[np.ndarray] = None,
                tile_id: str = "tile") -> RoadGraphResult:
        cfg = self.cfg
        if road_mask.ndim != 2:
            raise ValueError(f"road_mask must be 2-D, got shape {road_mask.shape}")
        if blocked_mask is not None and blocked_mask.shape != road_mask.shape:
            raise ValueError("blocked_mask must have the same shape as road_mask")
        trajectory: list = []

        def step(name, fn, summarize):
            t0 = time.perf_counter()
            out = fn()
            trajectory.append({
                "step": len(trajectory) + 1, "tool": name,
                "duration_s": round(time.perf_counter() - t0, 4),
                "summary": summarize(out),
            })
            return out

        clean, sx, sy = step(
            "preprocess_mask", lambda: preprocess_mask(road_mask, cfg),
            lambda o: f"{road_mask.shape}->{o[0].shape}, road_px={int(o[0].sum())}")
        blocked_proc = (np.zeros_like(clean) if blocked_mask is None else
                        _resize_binary(blocked_mask > 0, clean.shape[1], clean.shape[0], 0.5))
        skel = step("skeletonize", lambda: skeletonize(clean),
                    lambda o: f"skeleton_px={int(o.sum())}")
        g = step("skeleton_to_graph", lambda: skeleton_to_graph(skel, blocked_proc),
                 lambda o: f"nodes={o.number_of_nodes()}, edges={o.number_of_edges()}")
        pruned = step("prune_graph", lambda: self._prune(g),
                      lambda o: f"spurs_removed={o}, nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")
        passable, stats = step("assess_passability", lambda: self._assess(g, (sx + sy) / 2),
                               lambda o: f"passable_edges={o[1]['n_edges'] - o[1]['n_blocked_edges']}"
                                         f"/{o[1]['n_edges']}")
        stats["spurs_removed"] = pruned
        stats["ambiguous_edges"] = sum(1 for *_, d in g.edges(data=True) if d["ambiguous"])

        result = RoadGraphResult(tile_id, g, passable, stats, trajectory, (sx, sy),
                                 cfg.snap_max_dist_px)
        self._write_log(result)
        return result

    # -- internals -----------------------------------------------------------
    def _prune(self, g: nx.MultiGraph) -> int:
        contract_degree2(g)
        removed = prune_spurs(g, self.cfg.spur_length_px)
        split_edges_at_flood_boundaries(g, self.cfg.min_blocked_run_px)   # last: contraction would undo it
        return removed

    def _assess(self, g: nx.MultiGraph, mean_scale: float):
        for _, _, _, d in g.edges(keys=True, data=True):
            d["length_px"] = d["n_px"] / mean_scale
            d["blocked_fraction"] = d["blocked_px"] / max(d["n_px"], 1)
            d["max_blocked_run_px"] = max_blocked_run(d["pixels"], d["is_blocked"])
            limit = min(self.cfg.min_blocked_run_px, max(1.0, 0.5 * d["n_px"]))
            d["passable"] = d["max_blocked_run_px"] < limit

        passable = g.copy()
        passable.remove_edges_from([(u, v, k) for u, v, k, d in g.edges(keys=True, data=True)
                                    if not d["passable"]])

        def edge_components(graph):
            comps = []
            for nodes in nx.connected_components(graph):
                sub = graph.subgraph(nodes)
                if sub.number_of_edges() > 0:
                    comps.append(sum(d["length_px"] for *_, d in sub.edges(data=True)))
            return comps

        all_comps, pass_comps = edge_components(g), edge_components(passable)
        total = sum(all_comps)
        stats = {
            "n_nodes": g.number_of_nodes(),
            "n_edges": g.number_of_edges(),
            "n_blocked_edges": sum(1 for *_, d in g.edges(data=True) if not d["passable"]),
            "n_components": len(all_comps),
            "n_components_passable": len(pass_comps),
            "total_length_px": total,
            "passable_length_px": sum(pass_comps),
            "passable_length_frac": (sum(pass_comps) / total) if total else 0.0,
            # share of the road network that is one connected, drivable piece
            "network_continuity": (max(pass_comps) / total) if (total and pass_comps) else 0.0,
        }
        return passable, stats

    def _write_log(self, result: RoadGraphResult) -> None:
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "geospatial_agent.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"tile_id": result.tile_id, "stats": result.stats,
                                "trajectory": result.trajectory}) + "\n")
