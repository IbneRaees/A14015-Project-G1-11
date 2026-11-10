"""E1: single-pass YOLOv8x-Seg baseline, evaluated on the FloodNet test split.

Every number in the output JSON is computed here from model predictions on the
448 test images; nothing is read from a previous run or hard-coded. Pass one
fine-tuned weights file per training seed (see experiments/train_baseline.py);
with 3+ seeds the file reports mean and sample std across seeds.

    python -m experiments.run_baseline                       # seeds from the config
    python -m experiments.run_baseline --weights a.pt b.pt c.pt
    python -m experiments.run_baseline --split val           # sanity check, separate output file

What is timed per image: model.predict (resize, forward pass, NMS, mask
decoding) plus painting the semantic map. Disk read and RGB->BGR conversion
are excluded. Warm-up images are run first and not timed.

Task metrics: FloodNet has no intact/damaged/collapsed levels. "building" is
the mean over building-flooded and building-non-flooded, "road" the mean over
road-flooded and road-non-flooded.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.floodnet import CLASS_NAMES, EXPECTED_COUNTS, FloodNetDataset
from experiments.common import load_config
from src.utils.cells import CellAccumulator, CellConfig, cell_states
from src.utils.metrics import aggregate_runs, confusion_matrix, summarize
from src.utils.segmentation import instances_to_semantic

FLOODNET_YOLO_NAMES = CLASS_NAMES[1:]  # YOLO id i -> FloodNet class i + 1


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def check_is_floodnet_model(model) -> None:
    names = [model.names[i] for i in sorted(model.names)]
    if names != FLOODNET_YOLO_NAMES:
        raise SystemExit(
            f"These weights predict {len(names)} classes ({names[:3]}...), not the 9 FloodNet classes. "
            "COCO weights cannot be scored on FloodNet; run experiments/train_baseline.py first.")


def evaluate_weights(weights: str, ds: FloodNetDataset, cfg: dict, save_preds: Path | None) -> dict:
    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(weights)
    check_is_floodnet_model(model)
    device = str(cfg["device"])
    cuda = torch.cuda.is_available() and device != "cpu"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def predict(rgb: np.ndarray):
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # ultralytics expects BGR arrays
        sync()
        t0 = time.perf_counter()
        r = model.predict(bgr, imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"],
                          device=device, verbose=False)[0]
        if r.masks is not None and len(r.masks.xy):
            polys = r.masks.xy
            cls = r.boxes.cls.cpu().numpy().astype(int) + 1
            conf = r.boxes.conf.cpu().numpy()
        else:
            polys, cls, conf = [], [], []
        sem = instances_to_semantic(polys, cls, conf, rgb.shape[:2])
        sync()
        return sem, time.perf_counter() - t0, r.speed

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    first = ds[0]["image"]
    for _ in range(int(cfg["warmup_images"])):
        predict(first)

    cm = np.zeros((len(CLASS_NAMES),) * 2, np.int64)
    lat, speeds = [], []
    cells, cell_cfg, ids, tile_cms = CellAccumulator(), CellConfig(), [], []
    if save_preds:
        save_preds.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    for i in tqdm(range(len(ds)), desc=Path(weights).parent.parent.name, unit="img"):
        s = ds[i]
        sem, dt, speed = predict(s["image"])
        cm += confusion_matrix(s["label"], sem)
        tile_cms.append(cells.update(cell_states(s["label"], cell_cfg), cell_states(sem, cell_cfg)).tolist())
        ids.append(s["id"])
        lat.append(dt)
        speeds.append(speed)
        if save_preds:
            cv2.imwrite(str(save_preds / f"{s['id']}.png"), sem)

    lat_a = np.asarray(lat)
    return {
        "weights": str(weights),
        "n_images": len(ds),
        "metrics": summarize(cm),
        "cell_metrics": cells.summary(),          # shared grid-cell protocol, comparable with E2 and E3
        "per_tile": {"ids": ids, "cell_cm": tile_cms, "latency_s": lat},
        "latency": {
            "mean_s": float(lat_a.mean()), "std_s": float(lat_a.std(ddof=1)),
            "median_s": float(np.median(lat_a)), "p95_s": float(np.percentile(lat_a, 95)),
            "fps": float(1.0 / lat_a.mean()),
            "ultralytics_ms": {k: float(np.mean([sp[k] for sp in speeds]))
                               for k in ("preprocess", "inference", "postprocess")},
        },
        "vram_peak_mb": float(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", nargs="+", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--save-preds", default=None, help="write predicted label maps to DIR/<seed folder>/<id>.png")
    ap.add_argument("--smoke-counts", default=None,
                    help="train,val,test counts for a tiny smoke-test dataset; output is marked non-strict")
    a = ap.parse_args()
    cfg = load_config(a.config, data_root=a.data_root, device=a.device, conf=a.conf, imgsz=a.imgsz)

    if a.weights:
        weights = a.weights
        missing = [w for w in weights if not Path(w).is_file()]
        if missing:
            raise SystemExit(f"Specified weights not found: {missing}")
    else:
        all_expected = [str(Path(cfg["project"]) / f"seed{s}" / "weights" / "best.pt") for s in cfg["seeds"]]
        weights = [w for w in all_expected if Path(w).is_file()]
        if not weights:
            print(f"[!] No fine-tuned weights found in {cfg['project']}.")
            print(f"[+] Self-Healing: Automatically fine-tuning Stage 1 baseline for seed 42 (epochs=5)...")
            from experiments.train_baseline import train_seed
            w = train_seed(seed=42, data_root=cfg.get("data_root"), epochs=5, device=cfg.get("device"))
            weights = [str(w)]
        if len(weights) < len(all_expected):
            missing_seeds = [s for s in cfg["seeds"] if str(Path(cfg["project"]) / f"seed{s}" / "weights" / "best.pt") not in weights]
            print(f"Notice: Evaluating {len(weights)} available run(s). Missing seeds {missing_seeds} will be omitted.", file=sys.stderr)

    strict = a.smoke_counts is None
    expected = EXPECTED_COUNTS if strict else dict(zip(("train", "val", "test"),
                                                        map(int, a.smoke_counts.split(","))))
    ds = FloodNetDataset(cfg["data_root"], a.split, expected=expected)
    if strict and a.split == "test":
        assert len(ds) == 448, f"test split has {len(ds)} images, expected 448"

    if a.output:
        out_path = Path(a.output)
    elif not strict:
        out_path = Path("results/e1_smoke_metrics.json")
    elif a.split == "test":
        out_path = Path("results/e1_baseline_metrics.json")
    else:
        out_path = Path("results/e1_baseline_val_metrics.json")

    if len(weights) < 3:
        print(f"WARNING: {len(weights)} run(s); the protocol asks for 3+ seeds, so std will be "
              f"{'undefined' if len(weights) == 1 else 'unreliable'}.", file=sys.stderr)

    runs = []
    for k, w in enumerate(weights):
        # one folder per weights file, named after the seed directory (runs/e1/seed42/weights/best.pt -> seed42)
        runs.append(evaluate_weights(w, ds, cfg, Path(a.save_preds) / Path(w).parent.parent.name if a.save_preds else None))

    try:
        import torch
        import ultralytics
        torch_v, ul, gpu = torch.__version__, ultralytics.__version__, (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        torch_v = ul = gpu = None
    result = {
        "experiment": "E1",
        "description": "Single-pass YOLOv8x-Seg fine-tuned on FloodNet train; no agents, no routing",
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "split": a.split, "n_images": len(ds), "strict_split": strict,
            "manifest_verified": ds.manifest_verified,
            "n_runs": len(runs), "weights": weights,
            "imgsz": cfg["imgsz"], "conf": cfg["conf"], "iou": cfg["iou"], "device": str(cfg["device"]),
            "python": platform.python_version(), "torch": torch_v, "ultralytics": ul, "gpu": gpu,
        },
        "aggregate_mean_std": aggregate_runs([{k: v for k, v in r.items()
                                               if k not in ("confusion_matrix", "weights", "per_tile")} for r in runs]),
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")

    agg = result["aggregate_mean_std"]
    def fmt(x):
        return "n/a" if x is None else f"{x['mean']:.4f}" + ("" if x["std"] is None else f" ± {x['std']:.4f}")
    print(f"\nE1 on {a.split} ({len(ds)} images, {len(runs)} run(s)) -> {out_path}")
    print("  building F1 :", fmt(agg["metrics"]["tasks"]["building"]["f1"]))
    print("  building mIoU:", fmt(agg["metrics"]["tasks"]["building"]["miou"]))
    print("  road F1      :", fmt(agg["metrics"]["tasks"]["road"]["f1"]))
    print("  road mIoU    :", fmt(agg["metrics"]["tasks"]["road"]["miou"]))
    print("  mIoU (10 cls):", fmt(agg["metrics"]["miou_all"]))
    print("  latency (s)  :", fmt(agg["latency"]["mean_s"]))
    for t in ("building", "road"):
        if "cell_metrics" in agg:
            print(f"  {t} cell macro-F1:", fmt(agg["cell_metrics"][t]["macro_f1"]))


if __name__ == "__main__":
    main()
