"""Gap 2 calibration experiment on the FloodNet test split (protocol: src/eval/gap2.py).

Per seed it loads the E1 backbone and that seed's two heads (train them first with
experiments/train_uncertainty_head.py), fits the evidential head's temperature on val, then scores
five systems on the same test pixels: YOLO instance confidence, softmax, softmax + temperature
scaling, evidential, evidential + temperature scaling. One forward pass per tile feeds all of them.
Output: results/e4_gap2_calibration.json, read by experiments/compare_results.py.

    python -m experiments.run_gap2_calibration --seeds 42
    python -m experiments.run_gap2_calibration --plot            # all seeds + reliability diagrams
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from data.floodnet import EXPECTED_COUNTS, FloodNetDataset
from experiments.common import git_commit, load_config
from src.components.calibration import fit_temperature
from src.components.yolo_features import (downsample_labels_stride, fuse_features, resize_for_yolo, resize_label)
from src.eval.gap2 import SYSTEMS, Gap2Scorer, plot_reliability
from src.utils.metrics import aggregate_runs
from src.utils.segmentation import instances_to_semantic_with_conf


def check_heads(ev_meta: dict, sm_meta: dict) -> None:
    """The two heads must read the same features and write the same grid, or the comparison is not like for like."""
    for k in ("levels", "out_stride", "imgsz", "in_channels", "yolo_weights"):
        if ev_meta.get(k) != sm_meta.get(k):
            raise ValueError(f"evidential and softmax heads differ in '{k}': {ev_meta.get(k)} vs {sm_meta.get(k)}")


def forward_tile(fast, ev_head, sm_head, image: np.ndarray, meta: dict, map_conf: float, device):
    """One YOLO pass -> head outputs and the YOLO label/confidence maps, all on the head's output grid."""
    small, (h, w) = resize_for_yolo(image, meta["imgsz"])
    out = fast.run(np.ascontiguousarray(small[..., ::-1]), (h, w))
    s = meta["out_stride"]
    with torch.inference_mode():
        fused = fuse_features([f.float() for f in out.features], tuple(meta["levels"]))
        alpha = ev_head(fused, (h // s, w // s))[0].float().cpu().numpy()
        logits = sm_head(fused, (h // s, w // s))[0].float().cpu().numpy()
    keep = out.confidences >= map_conf
    label, conf, covered = instances_to_semantic_with_conf(
        [p for p, k in zip(out.polygons, keep) if k], out.class_ids[keep], out.confidences[keep], (h, w))
    ds = lambda a: downsample_labels_stride(a, s)
    return alpha, logits, ds(label), ds(conf), ds(covered), (h, w)


def fit_evidential_temperature(fast, ev_head, sm_head, ds_val, meta: dict, cfg: dict, seed: int) -> float:
    """Temperature of p -> softmax(log p / T) for the evidential head, by NLL on a pixel sample of val."""
    rng = np.random.default_rng(seed)
    n = len(ds_val) if cfg.get("fit_tiles") is None else min(cfg["fit_tiles"], len(ds_val))
    zs, ys = [], []
    for i in range(n):
        item = ds_val[i]
        alpha, _, _, _, _, hw = forward_tile(fast, ev_head, sm_head, item["image"], meta, cfg["map_conf"], None)
        gt = downsample_labels_stride(resize_label(item["label"], hw), meta["out_stride"])
        logp = np.log(np.clip(alpha / alpha.sum(0, keepdims=True), 1e-12, None)).reshape(alpha.shape[0], -1).T
        idx = rng.choice(logp.shape[0], size=min(cfg["temperature_pixels_per_image"], logp.shape[0]), replace=False)
        zs.append(logp[idx])
        ys.append(gt.reshape(-1)[idx])
    return fit_temperature(np.concatenate(zs), np.concatenate(ys))


def run_seed(fast, ev_head, sm_head, ev_meta: dict, sm_meta: dict, ds_val, ds_test, cfg: dict, seed: int) -> dict:
    check_heads(ev_meta, sm_meta)
    t_sm = float(sm_meta.get("temperature", 1.0))
    t_ev = fit_evidential_temperature(fast, ev_head, sm_head, ds_val, ev_meta, cfg, seed)
    scorer, tile_bins, ids = Gap2Scorer(), [], []
    from tqdm import tqdm
    for i in tqdm(range(len(ds_test)), desc=f"gap2 seed{seed}", unit="tile"):
        item = ds_test[i]
        alpha, logits, ylab, yconf, cov, hw = forward_tile(fast, ev_head, sm_head, item["image"], ev_meta,
                                                           cfg["map_conf"], None)
        gt = downsample_labels_stride(resize_label(item["label"], hw), ev_meta["out_stride"])
        tile_bins.append(scorer.update(gt, alpha, logits, ylab, yconf, cov, t_sm, t_ev))
        ids.append(item["id"])
    systems = scorer.summarize()
    for name in SYSTEMS:
        if systems[name].get("ece") is not None:
            systems[name]["per_tile_bins"] = np.stack([b[name] for b in tile_bins]).tolist()
    return {"seed": seed, "temperatures": {"softmax": t_sm, "evidential": t_ev}, "tile_ids": ids, "systems": systems}


def aggregate_systems(runs: List[dict]) -> dict:
    drop = ("per_tile_bins", "reliability")
    return {name: aggregate_runs([{k: v for k, v in r["systems"][name].items() if k not in drop} for r in runs])
            for name in SYSTEMS}


def build_models(cfg: dict, seed: int):
    from ultralytics import YOLO
    from src.components.heads import load_head
    from src.pipeline import UltralyticsFastPath

    dev = str(cfg["device"])
    tdev = "cpu" if dev == "cpu" else f"cuda:{dev}"

    ev_path = Path(cfg["head_ckpt_evidential"].format(seed=seed))
    sm_path = Path(cfg["head_ckpt_softmax"].format(seed=seed))
    if not ev_path.is_file() or not sm_path.is_file():
        from experiments.train_uncertainty_head import train as train_head
        from experiments.common import load_config
        head_cfg = load_config(str(Path(__file__).parent / "configs" / "head.yaml"),
                               data_root=cfg.get("data_root"), device=cfg.get("device"))
        if not ev_path.is_file():
            print(f"[+] Self-Healing: Training missing evidential head for seed {seed}...")
            train_head(head_cfg, "evidential", seed, ev_path.parent)
        if not sm_path.is_file():
            print(f"[+] Self-Healing: Training missing softmax head for seed {seed}...")
            train_head(head_cfg, "softmax", seed, sm_path.parent)

    ev, ev_meta = load_head(str(ev_path), tdev)
    sm, sm_meta = load_head(str(sm_path), tdev)
    if ev_meta["head"] != "evidential" or sm_meta["head"] != "softmax":
        raise ValueError("head checkpoints are swapped or wrong")
    if float(sm_meta.get("temperature", 1.0)) == 1.0:
        print("WARNING: the softmax checkpoint has temperature 1.0; its temperature was never fitted, so "
              "softmax_ts equals softmax_raw. Retrain it with experiments/train_uncertainty_head.py --head softmax.")
    yolo_w = Path(cfg["yolo_weights"].format(seed=seed))
    if not yolo_w.is_file():
        from experiments.train_baseline import train_seed
        print(f"[+] Self-Healing: Training missing YOLO baseline for seed {seed}...")
        yolo_w = train_seed(seed=seed, data_root=cfg.get("data_root"), epochs=5, device=cfg.get("device"))
    fast = UltralyticsFastPath(YOLO(str(yolo_w)), dev, cfg["map_conf"], cfg["iou"])
    return fast, ev.to(tdev), sm.to(tdev), ev_meta, sm_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "gap2_calibration.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fit-tiles", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--smoke-counts", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, data_root=a.data_root, device=a.device, fit_tiles=a.fit_tiles)
    seeds = a.seeds or cfg["seeds"]
    strict = a.smoke_counts is None
    expected = EXPECTED_COUNTS if strict else dict(zip(("train", "val", "test"), map(int, a.smoke_counts.split(","))))
    ds_val = FloodNetDataset(cfg["data_root"], "val", expected=expected)
    ds_eval = FloodNetDataset(cfg["data_root"], a.split, expected=expected)
    official = strict and a.split == "test" and a.fit_tiles is None
    out_path = Path(a.output or ("results/e4_gap2_calibration.json" if official else "results/e4_gap2_calibration_smoke.json"))
    if len(seeds) < 3:
        print(f"WARNING: {len(seeds)} seed(s); the protocol asks for 3+.")

    runs = []
    for seed in seeds:
        fast, ev, sm, ev_meta, sm_meta = build_models(cfg, seed)
        runs.append(run_seed(fast, ev, sm, ev_meta, sm_meta, ds_val, ds_eval, cfg, seed))
    result = {
        "experiment": "E4/gap2_calibration",
        "meta": {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
                 "split": a.split, "strict_split": strict, "official_run": official, "seeds": seeds, "n_runs": len(runs),
                 "map_conf": cfg["map_conf"], "n_bins": 15, "systems": list(SYSTEMS), "python": platform.python_version(),
                 "torch": torch.__version__,
                 "note": "primary ece is on covered pixels (those a YOLO instance claims), where all systems are scored; "
                         "ece_all is heads only"},
        "systems": aggregate_systems(runs),
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    f = lambda x: "n/a" if x is None else f"{x['mean']:.4f}" + ("" if x["std"] is None else f" ± {x['std']:.4f}")
    print(f"\nGap 2 calibration on {a.split}: {len(runs)} seed(s) -> {out_path}")
    for name in SYSTEMS:
        s = result["systems"][name]
        print(f"  {name:20s} ECE {f(s.get('ece'))}   accuracy {f(s.get('accuracy'))}   NLL {f(s.get('nll'))}")
    if a.plot:
        fig = out_path.parent / "figures" / "gap2_reliability.png"
        fig.parent.mkdir(parents=True, exist_ok=True)
        plot_reliability(result, fig)
        print("reliability diagrams ->", fig)


if __name__ == "__main__":
    main()
