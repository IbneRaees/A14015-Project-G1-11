"""E2: zero-shot monolithic VLM baseline (surrogate for SkyEyeGPT).

SkyEyeGPT has no public checkpoint, so Qwen2-VL-7B-Instruct stands in as "one
big VLM does everything, no routing, no graph, no verifier". Each test tile is
cut into the shared grid of cells; the VLM sees each cell crop and answers, in
JSON, whether buildings and roads are absent, flooded or not flooded. Cell
states are scored with src/utils/cells.py exactly like E1 and E3.

Notes for the paper
  * A VLM produces no masks, so pixel mIoU is not defined for E2.
  * Greedy decoding is deterministic: there is a single run, and uncertainty is a
    bootstrap over tiles, not a seed std.
  * One tile costs rows x cols VLM calls (24 with the default grid), so a full
    test pass is thousands of calls. Progress is appended to <log_dir>/tiles.jsonl
    and a re-run resumes from it.

    python -m experiments.run_vlm_zeroshot --limit 5          # time a few tiles first
    python -m experiments.run_vlm_zeroshot                    # full test split, resumable
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from data.floodnet import EXPECTED_COUNTS, FloodNetDataset
from experiments.common import git_commit, load_config
from src.utils.cells import (ABSENT, TASK_NAMES, CellConfig, bootstrap_ci, cell_boxes_for, cell_states,
                             summarize_cells, tile_confusions, tile_error_rate)
from src.vlm.prompts import CELL_PROMPT, PROMPT_SHA, parse_cell_answer


def shrink(crop: np.ndarray, max_side: int) -> np.ndarray:
    s = max_side / max(crop.shape[:2])
    if s >= 1:
        return crop
    return cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)


def answer_tile(vlm, image: np.ndarray, cell_cfg: CellConfig, crop_max_side: int, max_new_tokens: int):
    """VLM states for every cell of one tile. Returns (states, parse_failures, raw_answers)."""
    states = {t: np.zeros((cell_cfg.rows, cell_cfg.cols), np.int8) for t in TASK_NAMES}
    fails, raw = 0, []
    for k, (x0, y0, x1, y1) in enumerate(cell_boxes_for(image.shape[:2], cell_cfg)):
        r, c = divmod(k, cell_cfg.cols)
        text = vlm.generate(shrink(image[y0:y1, x0:x1], crop_max_side), CELL_PROMPT, max_new_tokens=max_new_tokens)
        raw.append(text)
        parsed = parse_cell_answer(text)
        for t in TASK_NAMES:
            if parsed[t] is None:
                fails += 1                     # counted, and treated as "absent"
            else:
                states[t][r, c] = parsed[t]
    return states, fails, raw


def load_done(path: Path) -> dict:
    if not path.is_file():
        return {}
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in recs}


def summarize_records(records: List[dict], cell_cfg: CellConfig) -> dict:
    cms = np.array([r["cell_cm"] for r in records])                       # (N, 2, 3, 3)
    lat = np.array([r["latency_s"] for r in records])
    calls = sum(r["n_calls"] for r in records)
    return {
        "n_tiles": len(records),
        "cell_metrics": summarize_cells({t: cms[:, i].sum(0) for i, t in enumerate(TASK_NAMES)}),
        "bootstrap_95ci": {t: {m: bootstrap_ci(cms, t, m) for m in ("macro_f1", "f1_flooded")}
                           for t in TASK_NAMES},
        "latency": {"mean_s": float(lat.mean()), "std_s": float(lat.std(ddof=1)) if len(lat) > 1 else None,
                    "median_s": float(np.median(lat)), "p95_s": float(np.percentile(lat, 95)),
                    "per_call_mean_s": float(lat.sum() / calls) if calls else None},
        "parse_failure_rate": float(sum(r["parse_failures"] for r in records) / max(2 * calls, 1)),
        "mean_tile_error": float(np.mean([r["tile_error"] for r in records])),
        "per_tile": {"ids": [r["id"] for r in records], "cell_cm": cms.tolist(),
                     "latency_s": [r["latency_s"] for r in records]},
    }


def run_e2(vlm, ds, cell_cfg: CellConfig, cfg: dict, log_dir: Path, limit: Optional[int] = None,
           resume: bool = True) -> List[dict]:
    log_dir.mkdir(parents=True, exist_ok=True)
    tiles_path, answers_path = log_dir / "tiles.jsonl", log_dir / "answers.jsonl"
    done = load_done(tiles_path) if resume else {}
    n = len(ds) if limit is None else min(limit, len(ds))
    from tqdm import tqdm
    for i in tqdm(range(n), desc="E2 tiles", unit="tile"):
        if ds.samples[i].image_id in done:
            continue
        item = ds[i]
        t0 = time.perf_counter()
        states, fails, raw = answer_tile(vlm, item["image"], cell_cfg, cfg["crop_max_side"], cfg["max_new_tokens"])
        latency = time.perf_counter() - t0
        gt = cell_states(item["label"], cell_cfg)
        rec = {"id": item["id"], "cell_cm": tile_confusions(gt, states).tolist(), "latency_s": latency,
               "n_calls": cell_cfg.rows * cell_cfg.cols, "parse_failures": fails,
               "tile_error": tile_error_rate(gt, states)}
        with open(tiles_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        with open(answers_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": item["id"], "answers": raw}) + "\n")
        done[item["id"]] = rec
    return [done[ds.samples[i].image_id] for i in range(n) if ds.samples[i].image_id in done]


def make_backend(cfg: dict):
    from src.vlm.backends import Qwen2VLBackend
    dev = cfg["device"] if str(cfg["device"]) in ("cpu", "cuda") or str(cfg["device"]).startswith("cuda") else f"cuda:{cfg['device']}"
    return Qwen2VLBackend(cfg["vlm_model"], dev, cfg["dtype"], max_pixels=cfg["crop_max_side"] ** 2,
                          load_in_4bit=cfg["load_in_4bit"])


def main(backend=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "e2_vlm.yaml"))
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--load-in-4bit", action="store_true", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N tiles only (timing / debugging)")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke-counts", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, data_root=a.data_root, vlm_model=a.vlm_model, device=a.device,
                      dtype=a.dtype, load_in_4bit=a.load_in_4bit, log_dir=a.log_dir)
    strict = a.smoke_counts is None
    expected = EXPECTED_COUNTS if strict else dict(zip(("train", "val", "test"), map(int, a.smoke_counts.split(","))))
    ds = FloodNetDataset(cfg["data_root"], a.split, expected=expected)
    official = strict and a.split == "test" and a.limit is None
    out_path = Path(a.output or ("results/e2_vlm_zeroshot_metrics.json" if official else "results/e2_smoke_metrics.json"))
    log_dir = Path(cfg["log_dir"]) / ("" if official else "partial")

    cell_cfg = CellConfig()
    vlm = backend or make_backend(cfg)
    records = run_e2(vlm, ds, cell_cfg, cfg, log_dir, a.limit, resume=not a.no_resume)
    try:
        import torch
        torch_v = torch.__version__
        vram = float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        torch_v = vram = gpu = None
    result = {
        "experiment": "E2",
        "description": "Zero-shot monolithic VLM (surrogate for SkyEyeGPT); one VLM call per grid cell",
        "meta": {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "git_commit": git_commit(), "split": a.split, "strict_split": strict, "official_run": official,
                 "vlm": vlm.name, "prompt_sha": PROMPT_SHA, "decoding": "greedy",
                 "cells": {"rows": cell_cfg.rows, "cols": cell_cfg.cols,
                           "building_min_frac": cell_cfg.building_min_frac, "road_min_frac": cell_cfg.road_min_frac},
                 "crop_max_side": cfg["crop_max_side"], "dtype": cfg["dtype"], "load_in_4bit": cfg["load_in_4bit"],
                 "python": platform.python_version(), "torch": torch_v, "gpu": gpu,
                 "n_runs": 1, "note": "deterministic decoding: one run; uncertainty is a bootstrap over tiles; "
                                      "pixel mIoU is undefined for a model that outputs text"},
        "vram_peak_mb": vram,
        **summarize_records(records, cell_cfg),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    cm = result["cell_metrics"]
    print(f"\nE2 on {a.split}: {result['n_tiles']} tiles -> {out_path}")
    for t in TASK_NAMES:
        print(f"  {t:8s} macro-F1 {cm[t]['macro_f1']}  flooded-F1 {cm[t]['f1_flooded']}")
    print(f"  latency/tile {result['latency']['mean_s']:.2f}s, parse failures {result['parse_failure_rate']:.1%}")


if __name__ == "__main__":
    main()
