"""E3: the full agentic pipeline on the FloodNet test split (and E5 ablations via --ablation).

Per seed (E1 backbone + evidential head of that seed):
  1. FIT on val. The fast stage runs over val tiles to collect router scores and verifier
     uncertainties. The fast path and the escalated path are timed on a few tiles to get
     t_fast and the extra VLM cost, which turn the latency target into a maximum escalation
     rate; the router threshold is the one that respects it. The verifier threshold is the
     one that lets `target_coverage` of val tiles through. Everything is written to
     <results>/..._fit.json. Nothing is fitted on test.
  2. EVALUATE on test: pixel metrics, grid-cell metrics, latency, routing and verifier
     behaviour, road-graph statistics, head calibration. Trajectories go to logs/.
Then mean and sample std over seeds.

    python -m experiments.run_e3                       # E3
    python -m experiments.run_e3 --ablation no_gap1    # E5: no road graph
    python -m experiments.run_e3 --ablation no_gap2    # E5: instance-confidence signal, no evidential head
    python -m experiments.run_e3 --ablation no_gap3    # E5: VLM on every cell
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from data.floodnet import EXPECTED_COUNTS, NUM_CLASSES, FloodNetDataset
from experiments.common import git_commit, load_config
from src.components.calibration import (CalibrationAccumulator, aurc, fit_threshold_for_coverage,
                                        risk_coverage_curve)
from src.components.router import fit_threshold, max_escalation_rate
from src.components.yolo_features import downsample_labels_stride, resize_label
from src.pipeline import AgenticPipeline, PipelineConfig, total_latency
from src.state import Disposition, Route
from src.utils.cells import (TASK_NAMES, CellAccumulator, CellConfig, cell_states, tile_error_rate)
from src.utils.metrics import aggregate_runs, confusion_matrix, summarize

ABLATIONS = {
    "no_gap1": dict(use_graph=False),
    "no_gap2": dict(uncertainty_source="instance_conf", verifier="instance_conf"),
    "no_gap3": dict(router_mode="always_vlm"),
}


def ablation_config(base: PipelineConfig, name: Optional[str]) -> PipelineConfig:
    if name in (None, "none"):
        return base
    if name not in ABLATIONS:
        raise ValueError(f"ablation must be one of {sorted(ABLATIONS)}")
    return dataclasses.replace(base, **ABLATIONS[name])


# --------------------------------------------------------------------- fitting
def fit_operating_points(router_scores: List[float], verifier_uncertainties: List[float], t_fast: Optional[float],
                         t_extra: Optional[float], cfg: dict, router_mode: str, has_verifier: bool) -> dict:
    """Pure arithmetic: thresholds from val scores and measured times. Nothing is assumed."""
    out: dict = {"router_threshold": None, "verifier_threshold": None, "t_fast_s": t_fast,
                 "t_vlm_extra_s": t_extra, "max_escalation_rate": None, "escalation_rate_used": None,
                 "latency_target_feasible": None}
    if router_mode == "entropy":
        if t_fast is None or t_extra is None:
            raise ValueError("entropy routing needs measured t_fast and t_vlm_extra")
        p_max = max_escalation_rate(t_fast, t_extra, cfg["target_latency_s"])
        cap = cfg.get("max_escalation_rate")
        p = p_max if cap is None else min(p_max, cap)
        out.update(max_escalation_rate=p_max, escalation_rate_used=p, latency_target_feasible=t_fast < cfg["target_latency_s"],
                   router_threshold=fit_threshold(router_scores, p))
    if has_verifier:
        out["verifier_threshold"] = fit_threshold_for_coverage(verifier_uncertainties, cfg["target_coverage"])
    return out


def fit_pipeline(pipe: AgenticPipeline, ds_val, cfg: dict) -> dict:
    """Fit thresholds on val and install them in the pipeline."""
    mode = pipe.cfg.router_mode
    n = len(ds_val) if cfg.get("fit_tiles") is None else min(cfg["fit_tiles"], len(ds_val))
    obs = [pipe.observe(ds_val[i]["image"]) for i in range(n)]
    scores = [o["router_score"] for o in obs if o["router_score"] is not None]
    vu = [o["verifier_uncertainty"] for o in obs if o["verifier_uncertainty"] is not None]
    has_ver = pipe.verifier is not None
    if has_ver:
        pipe.set_verifier_threshold(fit_threshold_for_coverage(vu, cfg["target_coverage"]))

    t_fast = t_extra = None
    if mode == "entropy":
        if pipe.vlm is None:
            raise RuntimeError("entropy routing needs a VLM backend to time the escalated path")
        k = min(cfg["calib_tiles"], len(ds_val))
        fast_t, esc_t = [], []
        for i in range(k):
            img = ds_val[i]["image"]
            pipe.set_router_threshold(float("inf"))          # never escalate
            fast_t.append(total_latency(pipe.process("calib", img).state))
            pipe.set_router_threshold(float("-inf"))         # always escalate (the top-scoring cells)
            esc_t.append(total_latency(pipe.process("calib", img).state))
        t_fast, t_extra = float(np.mean(fast_t)), float(np.mean(np.array(esc_t) - np.array(fast_t)))
    fit = fit_operating_points(scores, vu, t_fast, t_extra, cfg, mode, has_ver)
    fit.update(n_fit_tiles=n, n_calib_tiles=0 if mode != "entropy" else min(cfg["calib_tiles"], len(ds_val)),
               target_latency_s=cfg["target_latency_s"], target_coverage=cfg["target_coverage"])
    if mode == "entropy":
        pipe.set_router_threshold(fit["router_threshold"])
    return fit


# ------------------------------------------------------------------ evaluation
def _stats(x) -> dict:
    a = np.asarray(x, np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None,
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}


def evaluate(pipe: AgenticPipeline, ds, cell_cfg: CellConfig, traj_path: Optional[Path] = None,
             pred_dir: Optional[Path] = None) -> dict:
    try:
        import torch
        cuda = torch.cuda.is_available() and str(pipe.device).startswith("cuda")
    except ImportError:
        torch, cuda = None, False
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    cm_pix = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    acc, cal = CellAccumulator(), CalibrationAccumulator(15)
    ids, tile_cms, lat, vlm_calls, vlm_time, routes, review, unc, err = [], [], [], [], [], [], [], [], []
    graph = {"n_components": [], "network_continuity": [], "passable_length_frac": [], "n_blocked_edges": []}
    uncertain_edges, parse_fail_steps = [], 0
    if traj_path:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text("", encoding="utf-8")
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    for i in tqdm(range(len(ds)), desc="E3 test", unit="tile"):
        item = ds[i]
        gt, H, W = item["label"], *item["label"].shape
        out = pipe.process(item["id"], item["image"], ds.split)
        st = out.state
        cm_pix += confusion_matrix(gt, cv2.resize(out.label_small, (W, H), interpolation=cv2.INTER_NEAREST))
        g, p = cell_states(gt, cell_cfg), cell_states(out.label_small, cell_cfg)
        tile_cms.append(acc.update(g, p).tolist())
        err.append(tile_error_rate(g, p))
        ids.append(item["id"])
        lat.append(total_latency(st))
        routes.append(st.routing.route == Route.VLM)
        vlm_calls.append(st.vision.vlm_calls)
        vlm_time.append(st.vision.vlm_time_s)
        review.append(st.verdict.disposition == Disposition.HUMAN_REVIEW)
        unc.append(st.uncertainty.tile_uncertainty if st.uncertainty else np.nan)
        if st.road_graph:
            for k in graph:
                graph[k].append(getattr(st.road_graph, k))
        if st.uncertainty:
            uncertain_edges.append(len(st.uncertainty.uncertain_edges))
        if out.alpha is not None:
            probs = out.alpha / out.alpha.sum(0, keepdims=True)
            gt_s = downsample_labels_stride(resize_label(gt, out.hw), pipe.out_stride)
            cal.update(probs.max(0), probs.argmax(0) == gt_s)
        if traj_path:
            st.append_jsonl(str(traj_path))
        if pred_dir:                       # final label map at the resized resolution, for the route benchmark
            cv2.imwrite(str(pred_dir / f"{item['id']}.png"), out.label_small)

    lat_a, routes_a, review_a = np.array(lat), np.array(routes), np.array(review)
    result = {
        "n_images": len(ds),
        "metrics": summarize(cm_pix),
        "cell_metrics": acc.summary(),
        "latency": {**{f"{k}_s": v for k, v in _stats(lat).items()}, "fps": float(1 / lat_a.mean()),
                    "fast_only_mean_s": float(lat_a[~routes_a].mean()) if (~routes_a).any() else None,
                    "escalated_mean_s": float(lat_a[routes_a].mean()) if routes_a.any() else None},
        "routing": {"escalation_rate": float(routes_a.mean()), "mean_vlm_calls": float(np.mean(vlm_calls)),
                    "mean_vlm_time_s": float(np.mean(vlm_time))},
        "graph": {k: (float(np.mean(v)) if v else None) for k, v in graph.items()},
        "vram_peak_mb": float(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
        "per_tile": {"ids": ids, "cell_cm": tile_cms, "latency_s": lat, "tile_error": err,
                     "tile_uncertainty": [None if np.isnan(u) else float(u) for u in unc],
                     "escalated": routes_a.tolist(), "review": review_a.tolist()},
    }
    if pipe.verifier is not None:
        u = np.array(unc)
        cov, risk = risk_coverage_curve(u, err)
        result["verifier"] = {
            "review_rate": float(review_a.mean()), "coverage": float(1 - review_a.mean()),
            "overall_tile_error": float(np.mean(err)),
            "selective_tile_error": float(np.mean(np.array(err)[~review_a])) if (~review_a).any() else None,
            "aurc_tile": aurc(cov, risk),
            "mean_uncertain_edges": float(np.mean(uncertain_edges)) if uncertain_edges else None,
        }
    result["calibration"] = {"head_ece": cal.ece()} if cal.count.sum() else None
    return result


# ------------------------------------------------------------------ assembly
def build_pipeline(cfg: dict, seed: int, vlm, ablation: Optional[str]) -> AgenticPipeline:
    from ultralytics import YOLO
    from src.components.heads import load_head
    from src.components.router import DynamicRouter, RouterConfig
    from src.agents.geospatial_agent import GeospatialConfig, GeospatialReasoningAgent
    from src.pipeline import UltralyticsFastPath

    dev = str(cfg["device"])
    tdev = "cpu" if dev == "cpu" else f"cuda:{dev}"
    pcfg = PipelineConfig(imgsz=cfg["imgsz"], map_conf=cfg["map_conf"], router_measure=cfg["router_measure"],
                          verifier_measure=cfg["verifier_measure"], vlm_crop_max_side=cfg["vlm_crop_max_side"],
                          vlm_max_new_tokens=cfg["vlm_max_new_tokens"], keep_alpha=True)
    pcfg = ablation_config(pcfg, ablation)
    head, meta = None, None
    if pcfg.uncertainty_source == "evidential" or pcfg.verifier == "evidential":
        head, meta = load_head(cfg["head_ckpt"].format(seed=seed), tdev)
        head = head.to(tdev)
        pcfg = dataclasses.replace(pcfg, imgsz=meta["imgsz"])
    yolo = YOLO(cfg["yolo_weights"].format(seed=seed))
    fast = UltralyticsFastPath(yolo, dev, cfg["conf_low"], cfg["iou"], want_features=head is not None)
    router = DynamicRouter(RouterConfig(mode=pcfg.router_mode, **cfg["router"]))
    geo = GeospatialReasoningAgent(GeospatialConfig(min_blocked_run_px=cfg.get("min_blocked_run_px", 20)))
    return AgenticPipeline(fast, pcfg, head=head, head_meta=meta, router=router, geospatial=geo, vlm=vlm, device=tdev)


def run_seed(pipe: AgenticPipeline, ds_val, ds_test, cfg: dict, cell_cfg: CellConfig, traj_path=None,
             pred_dir=None) -> dict:
    fit = fit_pipeline(pipe, ds_val, cfg)
    run = evaluate(pipe, ds_test, cell_cfg, traj_path, pred_dir)
    run["fit"] = fit
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "e3_pipeline.yaml"))
    ap.add_argument("--ablation", choices=["none", *sorted(ABLATIONS)], default="none")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fit-tiles", type=int, default=None)
    ap.add_argument("--calib-tiles", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--save-preds", default=None, help="write final label maps to DIR/seed<N>/<id>.png")
    ap.add_argument("--smoke-counts", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, data_root=a.data_root, device=a.device, fit_tiles=a.fit_tiles,
                      calib_tiles=a.calib_tiles)
    seeds = a.seeds or cfg["seeds"]
    strict = a.smoke_counts is None
    expected = EXPECTED_COUNTS if strict else dict(zip(("train", "val", "test"), map(int, a.smoke_counts.split(","))))
    ds_val = FloodNetDataset(cfg["data_root"], "val", expected=expected)
    ds_eval = FloodNetDataset(cfg["data_root"], a.split, expected=expected)
    official = strict and a.split == "test" and a.fit_tiles is None
    name = "e3_proposed" if a.ablation == "none" else f"e5_{a.ablation}"
    out_path = Path(a.output or (f"results/{name}_metrics.json" if official else "results/e3_smoke_metrics.json"))
    if len(seeds) < 3:
        print(f"WARNING: {len(seeds)} seed(s); the protocol asks for 3+.")

    vlm = None
    mode = ablation_config(PipelineConfig(), a.ablation).router_mode
    if mode != "always_fast":
        from src.vlm.backends import Qwen2VLBackend
        vlm = Qwen2VLBackend(cfg["vlm_model"], "cpu" if str(cfg["device"]) == "cpu" else f"cuda:{cfg['device']}",
                             cfg["vlm_dtype"], cfg["vlm_crop_max_side"] ** 2, cfg["vlm_load_in_4bit"])
    cell_cfg = CellConfig()
    runs = []
    for seed in seeds:
        pipe = build_pipeline(cfg, seed, vlm, None if a.ablation == "none" else a.ablation)
        run = run_seed(pipe, ds_val, ds_eval, cfg, cell_cfg, Path(cfg["log_dir"]) / f"{name}_seed{seed}.jsonl",
                       Path(a.save_preds) / f"seed{seed}" if a.save_preds else None)
        run["seed"] = seed
        runs.append(run)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        torch_v, gpu = torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        torch_v = gpu = None
    result = {
        "experiment": "E3" if a.ablation == "none" else f"E5/{a.ablation}",
        "meta": {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
                 "split": a.split, "strict_split": strict, "official_run": official, "seeds": seeds, "n_runs": len(runs),
                 "pipeline": dataclasses.asdict(ablation_config(PipelineConfig(), a.ablation)),
                 "vlm": None if vlm is None else vlm.name,
                 "cells": dataclasses.asdict(cell_cfg), "target_latency_s": cfg["target_latency_s"],
                 "target_coverage": cfg["target_coverage"], "min_blocked_run_px": cfg.get("min_blocked_run_px", 20), "python": platform.python_version(), "torch": torch_v,
                 "gpu": gpu},
        "aggregate_mean_std": aggregate_runs([{k: v for k, v in r.items() if k not in ("per_tile", "fit", "seed")}
                                              for r in runs]),
        "runs": runs,
    }
    out_path.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    agg = result["aggregate_mean_std"]
    f = lambda x: "n/a" if x is None else f"{x['mean']:.4f}" + ("" if x["std"] is None else f" ± {x['std']:.4f}")
    print(f"\n{result['experiment']} on {a.split}: {len(runs)} seed(s) -> {out_path}")
    for t in TASK_NAMES:
        print(f"  {t:8s} cell macro-F1 {f(agg['cell_metrics'][t]['macro_f1'])}   pixel F1 {f(agg['metrics']['tasks'][t]['f1'])}")
    print(f"  latency/tile {f(agg['latency']['mean_s'])}s   escalation rate {f(agg['routing']['escalation_rate'])}")


if __name__ == "__main__":
    main()
