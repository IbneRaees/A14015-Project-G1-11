"""E3 orchestration: one UAV tile through the full agentic pipeline.

    resize -> YOLO fast path (+ neck features in the same pass) -> evidential head
           -> label map -> Router -> [VLM re-judges the flagged regions]
           -> Geospatial agent (road graph) -> Verifier -> report

Everything user-facing (routing regions, verifier regions) is in ORIGINAL tile
pixel coordinates; internal maps live at the resized resolution (long side
`imgsz`) and are scaled back where they leave the pipeline.

What the heavy path does: the VLM is shown the crop of each flagged region and
asked the cell question (buildings / roads: none, flooded, not flooded). Where
it names a flooded or dry state, every building (or road) pixel of the label map
inside that region gets that state. "none" never erases a detection. A reply that
cannot be parsed leaves the region unchanged and is counted.

Ablations for E5 are config switches, so the same code path serves E3 and E5:
    use_graph=False                                  Gap 1 off
    uncertainty_source/verifier = "instance_conf"    Gap 2 off
    router_mode="always_vlm"                         Gap 3 off
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from src.agents.baseline_verifier import InstanceConfidenceVerifier
from src.agents.geospatial_agent import GeospatialReasoningAgent
from src.agents.verifier_agent import VerifierAgent, VerifierConfig
from src.components.evidential import dirichlet_stats, uncertainty_map
from src.components.fallback_uncertainty import instance_uncertainty_map
from src.components.router import DynamicRouter, RouterConfig
from src.components.yolo_features import resize_for_yolo
from src.state import (Disposition, RoadGraphSummary, Route, TileInput, TileState, VisionOutput, Verdict)
from src.utils.cells import ABSENT, FLOODED, TASK_IDS, TASK_NAMES, CellConfig, cell_boxes_for
from src.utils.segmentation import instances_to_semantic
from src.vlm.prompts import CELL_PROMPT, parse_cell_answer

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ------------------------------------------------------------------ fast path
@dataclass
class FastPathOutput:
    polygons: List[np.ndarray]          # instance outlines, resized-image pixels
    class_ids: np.ndarray               # FloodNet ids 1-9
    confidences: np.ndarray
    features: Optional[list] = None     # [P3, P4, P5] neck features from the same forward pass


class FastPath(Protocol):
    def run(self, bgr: np.ndarray, hw: Tuple[int, int]) -> FastPathOutput: ...


class UltralyticsFastPath:
    """Fine-tuned YOLOv8-Seg. One forward pass gives instances and neck features."""

    def __init__(self, yolo, device: str = "cpu", conf_low: float = 0.05, iou: float = 0.7,
                 want_features: bool = True):
        from src.components.yolo_features import YoloNeckFeatures   # needs torch
        self.yolo, self.device, self.conf_low, self.iou = yolo, device, conf_low, iou
        self.nf = YoloNeckFeatures.for_predictor(yolo, device=device) if want_features else None

    def run(self, bgr: np.ndarray, hw: Tuple[int, int]) -> FastPathOutput:
        kw = dict(imgsz=hw, conf=self.conf_low, iou=self.iou, device=self.device)
        if self.nf is not None:
            res, feats = self.nf.predict_with_features(bgr, **kw)
        else:
            res, feats = self.yolo.predict(bgr, verbose=False, **kw)[0], None
        polys, cls, conf = [], [], []
        if res.masks is not None and len(res.masks.xy):
            ids = res.boxes.cls.cpu().numpy().astype(int) + 1
            for poly, c, k in zip(res.masks.xy, ids, res.boxes.conf.cpu().numpy()):
                if 1 <= c <= 9 and len(poly) >= 3:        # ignore ids outside the FloodNet label set
                    polys.append(np.asarray(poly)); cls.append(c); conf.append(k)
        return FastPathOutput(polys, np.asarray(cls, int), np.asarray(conf, float), feats)


# ------------------------------------------------------------------ config / output
@dataclass
class PipelineConfig:
    imgsz: int = 1024
    map_conf: float = 0.25               # instances below this stay out of the label map (same as E1)
    router_mode: str = "entropy"         # entropy | always_fast | always_vlm   (Gap 3)
    router_measure: str = "entropy"      # evidential map the router reads: vacuity | entropy | epistemic
    uncertainty_source: str = "evidential"   # evidential | instance_conf          (Gap 2)
    verifier: str = "evidential"         # evidential | instance_conf | none     (Gap 2)
    verifier_measure: str = "vacuity"
    use_graph: bool = True               # Gap 1
    uncovered_value: float = 0.5         # fallback map prior for pixels with no detection
    vlm_grid: Tuple[int, int] = (4, 6)   # cells the VLM re-judges when router_mode="always_vlm"
    vlm_crop_max_side: int = 448
    vlm_max_new_tokens: int = 48
    keep_alpha: bool = False


@dataclass
class PipelineOutput:
    state: TileState
    label_small: np.ndarray              # final class-id map at resized resolution
    hw: Tuple[int, int]
    scale: Tuple[float, float]           # (sx, sy) = resized / original
    road_result: Any = None
    alpha: Optional[np.ndarray] = None
    instances: Optional[FastPathOutput] = None


def total_latency(state: TileState) -> float:
    """Seconds from decoded tile to final verdict (top-level steps only)."""
    return float(sum(s["duration_s"] for s in state.trajectory if not s.get("nested")))


def _areas(polys) -> np.ndarray:
    return np.array([cv2.contourArea(np.asarray(p, np.float32)) for p in polys], np.float64)


# ------------------------------------------------------------------ pipeline
class AgenticPipeline:
    def __init__(self, fast_path: FastPath, cfg: Optional[PipelineConfig] = None, head=None,
                 head_meta: Optional[dict] = None, router: Optional[DynamicRouter] = None,
                 verifier=None, geospatial: Optional[GeospatialReasoningAgent] = None, vlm=None,
                 device: str = "cpu"):
        self.cfg = cfg or PipelineConfig()
        c = self.cfg
        self.fast_path, self.head, self.vlm, self.device = fast_path, head, vlm, device
        meta = head_meta or {}
        self.out_stride = meta.get("out_stride", 2)
        self.levels = tuple(meta.get("levels", (0, 1)))
        self.needs_head = c.uncertainty_source == "evidential" or c.verifier == "evidential"
        if self.needs_head and head is None:
            raise ValueError("an evidential uncertainty source or verifier needs a trained evidential head")
        if c.uncertainty_source not in ("evidential", "instance_conf") or c.verifier not in (
                "evidential", "instance_conf", "none"):
            raise ValueError("bad uncertainty_source / verifier")
        self.router = router or DynamicRouter(RouterConfig(mode=c.router_mode))
        if verifier is None and c.verifier == "evidential":
            verifier = VerifierAgent(VerifierConfig(measure=c.verifier_measure))
        elif verifier is None and c.verifier == "instance_conf":
            verifier = InstanceConfidenceVerifier()
        self.verifier = verifier
        self.geo = geospatial or GeospatialReasoningAgent()

    # -- thresholds are fitted on val by the runner -------------------------
    def set_router_threshold(self, thr: Optional[float]) -> None:
        self.router.cfg.threshold = thr

    def set_verifier_threshold(self, thr: Optional[float]) -> None:
        if isinstance(self.verifier, VerifierAgent):
            self.verifier.cfg.threshold = thr
        elif self.verifier is not None:
            self.verifier.threshold = thr

    # -- stages -------------------------------------------------------------
    def _sync(self) -> None:
        if HAS_TORCH and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def _alpha(self, feats, hw) -> np.ndarray:
        from src.components.yolo_features import fuse_features
        h, w = hw
        with torch.inference_mode():
            fused = fuse_features([f.float() for f in feats], self.levels)
            a = self.head(fused, (h // self.out_stride, w // self.out_stride))
        return a[0].float().cpu().numpy()

    def _fast_stage(self, state: TileState, image_rgb: np.ndarray) -> dict:
        c = self.cfg
        with state.step("preprocess") as rec:
            small, (h, w) = resize_for_yolo(image_rgb, c.imgsz)
            bgr = np.ascontiguousarray(small[..., ::-1])
            rec["summary"] = f"{image_rgb.shape[0]}x{image_rgb.shape[1]} -> {h}x{w}"
        with state.step("fast_path") as rec:
            out = self.fast_path.run(bgr, (h, w))
            self._sync()
            rec["summary"] = f"{len(out.class_ids)} instances"
        alpha = None
        if self.needs_head:
            with state.step("evidential_head") as rec:
                alpha = self._alpha(out.features, (h, w))
                self._sync()
                rec["summary"] = f"alpha {tuple(alpha.shape)}"
        with state.step("label_map") as rec:
            keep = out.confidences >= c.map_conf
            label = instances_to_semantic([p for p, k in zip(out.polygons, keep) if k],
                                          out.class_ids[keep], out.confidences[keep], (h, w))
            rec["summary"] = f"{int(keep.sum())} instances painted"
        return {"hw": (h, w), "out": out, "alpha": alpha, "label": label}

    def _router_map(self, alpha, out, hw) -> np.ndarray:
        if self.cfg.uncertainty_source == "evidential":
            return uncertainty_map(dirichlet_stats(alpha), self.cfg.router_measure)
        return instance_uncertainty_map(out.polygons, out.confidences, hw, self.cfg.uncovered_value)

    def _verifier_uncertainty(self, alpha, out) -> Optional[float]:
        if isinstance(self.verifier, VerifierAgent):
            return self.verifier.tile_uncertainty(alpha)
        if isinstance(self.verifier, InstanceConfidenceVerifier):
            return self.verifier.tile_uncertainty(out.class_ids, out.confidences, _areas(out.polygons))
        return None

    def observe(self, image_rgb: np.ndarray) -> Dict[str, float]:
        """Fast stage only, with the scores the thresholds are fitted on. Needs no thresholds."""
        st = TileState(TileInput("observe", *image_rgb.shape[:2]))
        p = self._fast_stage(st, image_rgb)
        score = None
        if self.cfg.router_mode == "entropy":
            score = self.router.score(self._router_map(p["alpha"], p["out"], p["hw"]))[0]
        return {"router_score": score, "verifier_uncertainty": self._verifier_uncertainty(p["alpha"], p["out"]),
                "fast_stage_s": total_latency(st)}

    def _shrink(self, crop: np.ndarray) -> np.ndarray:
        m = self.cfg.vlm_crop_max_side
        s = m / max(crop.shape[:2])
        if s >= 1:
            return crop
        return cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)

    def _vlm_refine(self, image_rgb: np.ndarray, label: np.ndarray, regions: List[List[int]]):
        H, W = image_rgb.shape[:2]
        h, w = label.shape
        sx, sy = w / W, h / H
        calls = fails = changed = 0
        spent = 0.0
        for x0, y0, x1, y1 in regions:
            crop = image_rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            t0 = time.perf_counter()
            text = self.vlm.generate(self._shrink(crop), CELL_PROMPT, max_new_tokens=self.cfg.vlm_max_new_tokens)
            spent += time.perf_counter() - t0
            calls += 1
            ans = parse_cell_answer(text)
            sub = label[int(round(y0 * sy)):int(round(y1 * sy)), int(round(x0 * sx)):int(round(x1 * sx))]
            for task in TASK_NAMES:
                state = ans[task]
                if state is None:
                    fails += 1
                    continue
                if state == ABSENT:                  # "none" never erases a detection
                    continue
                fid, did = TASK_IDS[task]
                new = fid if state == FLOODED else did
                m = (sub == fid) | (sub == did)
                changed += int((m & (sub != new)).sum())
                sub[m] = new                         # `sub` is a view into `label`
        return label, {"calls": calls, "parse_failures": fails, "changed_px": changed, "time_s": spent}

    # -- one tile -----------------------------------------------------------
    def process(self, tile_id: str, image_rgb: np.ndarray, split: Optional[str] = None) -> PipelineOutput:
        c = self.cfg
        H, W = image_rgb.shape[:2]
        state = TileState(TileInput(tile_id, H, W, split=split))
        p = self._fast_stage(state, image_rgb)
        (h, w), out, alpha, label = p["hw"], p["out"], p["alpha"], p["label"]
        sx, sy = w / W, h / H

        with state.step("route") as rec:
            if c.router_mode == "entropy":
                decision = self.router.route(self._router_map(alpha, out, (h, w)), (H, W))
            else:
                decision = self.router.route()
                if decision.route == Route.VLM:
                    decision.regions = cell_boxes_for((H, W), CellConfig(*c.vlm_grid))
            state.routing = decision
            rec["summary"] = f"{decision.route.value}: {decision.reason}"

        stats = {"calls": 0, "parse_failures": 0, "changed_px": 0, "time_s": 0.0}
        if decision.route == Route.VLM and self.vlm is not None:
            with state.step("vlm_refine") as rec:
                label, stats = self._vlm_refine(image_rgb, label, decision.regions)
                rec["summary"] = (f"{stats['calls']} calls, {stats['changed_px']} px relabelled, "
                                  f"{stats['parse_failures']} unparsed")
        elif decision.route == Route.VLM:
            state.errors.append("route=vlm but no VLM backend; fast-path labels kept")

        road_result = None
        if c.use_graph:
            with state.step("geospatial") as rec:
                road_result = self.geo.analyze_label_map(label, tile_id)
                state.road_graph = RoadGraphSummary.from_stats(road_result.stats)
                rec["summary"] = (f"nodes={road_result.stats['n_nodes']}, edges={road_result.stats['n_edges']}, "
                                  f"continuity={road_result.stats['network_continuity']:.2f}")
            state.import_steps(road_result.trajectory, "geospatial")

        if c.verifier == "evidential":
            with state.step("verifier") as rec:
                report, verdict = self.verifier.assess(alpha, road_result, (h, w))
                report.flagged_regions = [[int(round(b[0] / sx)), int(round(b[1] / sy)),
                                           int(round(b[2] / sx)), int(round(b[3] / sy))]
                                          for b in report.flagged_regions]
                state.uncertainty, state.verdict = report, verdict
                rec["summary"] = f"{verdict.disposition.value}; tile_u={report.tile_uncertainty:.3f}"
        elif c.verifier == "instance_conf":
            with state.step("verifier") as rec:
                report, verdict = self.verifier.assess(out.class_ids, out.confidences, _areas(out.polygons), h * w)
                state.uncertainty, state.verdict = report, verdict
                rec["summary"] = f"{verdict.disposition.value}; tile_u={report.tile_uncertainty:.3f}"
        else:
            state.verdict = Verdict(Disposition.AUTO_REPORT, ["verifier disabled"])

        latency = total_latency(state)
        used_vlm = decision.route == Route.VLM and self.vlm is not None
        state.vision = VisionOutput(decision.route, "yolov8x-seg" + (f"+{self.vlm.name}" if used_vlm else ""),
                                    latency, len(out.class_ids), vlm_calls=stats["calls"],
                                    vlm_time_s=round(stats["time_s"], 4))
        if c.keep_alpha:
            state.arrays["alpha"] = alpha
        state.arrays["label_small"] = label
        return PipelineOutput(state, label, (h, w), (sx, sy), road_result, alpha if c.keep_alpha else None, out)
