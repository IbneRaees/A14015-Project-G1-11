"""Gap 1 route-query benchmark runner (see src/eval/route_benchmark.py for the protocol).

Systems scored on the same fixed query set:
  oracle/graph, oracle/pixel_cc   ground-truth masks: the floor of the benchmark itself
  <name>/graph, <name>/pixel_cc   predicted label maps from `--preds NAME=DIR`
                                  (E1: run_baseline --save-preds; E3: run_e3 --save-preds)
  vlm/direct                      a VLM asked about the image with A and B marked (`--vlm`)

One seed per name: pass `--preds e3_s42=... --preds e3_s123=... --preds e3_s456=...`; the
comparison script groups names ending in `_s<seed>` into one family.

    python -m experiments.run_gap1_routes --preds e1_s42=results/preds/e1 --preds e3_s42=results/preds/e3/seed42
    python -m experiments.run_gap1_routes --vlm --preds ...        # adds the direct-VLM system (slow, resumable)
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from data.floodnet import EXPECTED_COUNTS, FloodNetDataset, read_label
from experiments.common import git_commit
from src.agents.geospatial_agent import GeospatialConfig
from src.eval.route_benchmark import (ROUTE_PROMPT_SHA, TYPES, GraphAnswerer, PixelCCAnswerer, QueryConfig,
                                      VLMAnswerer, answer_tile, bootstrap_route_ci, config_dict,
                                      generate_queries, queries_fingerprint, route_metrics, TYPE_INDEX)


def load_pred(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read {path}")
    if img.ndim == 3:
        img = img[..., 0]
    if img.max() > 9:
        raise ValueError(f"{path.name}: values up to {int(img.max())}; expected class ids 0-9")
    return img.astype(np.uint8)


def summarize_system(per_tile: np.ndarray, unanswerable: int) -> dict:
    return {"metrics": route_metrics(per_tile.sum(0)), "unanswerable": int(unanswerable),
            "bootstrap_95ci": {m: bootstrap_route_ci(per_tile, m)
                               for m in ("connectivity_error_rate", "false_passable_rate")},
            "per_tile_counts": per_tile.tolist()}


def run_routes(ds, cfg: QueryConfig, sources: Dict[str, Path], vlm=None, limit: Optional[int] = None,
               log_dir: Optional[Path] = None, vlm_max_side: int = 1024) -> dict:
    n = len(ds) if limit is None else min(limit, len(ds))
    labels_queries = []
    for i in range(n):
        s = ds.samples[i]
        label = read_label(s.label_path)
        qs = generate_queries(label, s.image_id, cfg)
        if qs:
            labels_queries.append((i, s.image_id, label, qs))
    all_q = [q for *_, qs in labels_queries for q in qs]
    tile_ids = [t for _, t, _, _ in labels_queries]

    systems: Dict[str, dict] = {}

    def run_label_source(name: str, get_label):
        acc = {"graph": ([], 0), "pixel_cc": ([], 0)}
        counts = {k: [] for k in acc}
        un = {k: 0 for k in acc}
        for i, tid, gt_label, qs in labels_queries:
            lab = get_label(tid, gt_label)
            H, W = gt_label.shape
            for kind, cls in (("graph", GraphAnswerer), ("pixel_cc", PixelCCAnswerer)):
                ans = cls(lab, (H, W), tid, cfg) if kind == "graph" else cls(lab, (H, W), cfg)
                c, u = answer_tile(qs, ans)
                counts[kind].append(c)
                un[kind] += u
        for kind in counts:
            systems[f"{name}/{kind}"] = summarize_system(np.stack(counts[kind]), un[kind])

    run_label_source("oracle", lambda tid, gt: gt)
    for name, d in sources.items():
        run_label_source(name, lambda tid, gt, d=d: load_pred(Path(d) / f"{tid}.png"))

    if vlm is not None:
        done: Dict[tuple, dict] = {}
        log_path = (log_dir / "vlm_route_answers.jsonl") if log_dir else None
        if log_path and log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                done[(r["id"], r["idx"])] = r
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        counts, un = [], 0
        from tqdm import tqdm
        for i, tid, gt_label, qs in tqdm(labels_queries, desc="VLM routes", unit="tile"):
            c = np.zeros((len(TYPES), 2), np.int64)
            todo = [q for q in qs if (tid, q["idx"]) not in done]
            if todo:
                ans = VLMAnswerer(vlm, ds[i]["image"], vlm_max_side)
                for q in todo:
                    yes, ok = ans(q["a"], q["b"])
                    rec = {"id": tid, "idx": q["idx"], "yes": bool(yes), "answerable": bool(ok), "raw": ans.raw[-1]}
                    done[(tid, q["idx"])] = rec
                    if log_path:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
            for q in qs:
                r = done[(tid, q["idx"])]
                c[TYPE_INDEX[q["type"]], int(r["yes"])] += 1
                un += int(not r["answerable"])
            counts.append(c)
        systems["vlm/direct"] = summarize_system(np.stack(counts), un)

    by_type = {t: sum(q["type"] == t for q in all_q) for t in TYPES}
    return {"n_tiles_scanned": n, "tile_ids": tile_ids, "n_queries": len(all_q), "queries_by_type": by_type,
            "queries_fingerprint": queries_fingerprint(all_q), "systems": systems}, all_q


def main(backend=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--preds", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--vlm", action="store_true", help="also score direct VLM prompting (slow)")
    ap.add_argument("--vlm-model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--route-max-pixels", type=int, default=1024 * 768,
                    help="pixel budget the VLM processor may use for the whole tile")
    ap.add_argument("--pairs-per-type", type=int, default=4)
    ap.add_argument("--min-dist-frac", type=float, default=0.15)
    ap.add_argument("--snap-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-blocked-run-px", type=int, default=20,
                    help="geospatial agent: flood run that blocks an edge (sweep this on val)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-dir", default="logs/e4_gap1")
    ap.add_argument("--output", default=None)
    ap.add_argument("--queries-out", default=None)
    ap.add_argument("--smoke-counts", default=None)
    a = ap.parse_args()

    strict = a.smoke_counts is None
    expected = EXPECTED_COUNTS if strict else dict(zip(("train", "val", "test"), map(int, a.smoke_counts.split(","))))
    ds = FloodNetDataset(a.data_root, a.split, expected=expected)
    official = strict and a.split == "test" and a.limit is None
    out = Path(a.output or ("results/e4_gap1_routes.json" if official else "results/e4_gap1_routes_smoke.json"))
    qout = Path(a.queries_out or ("results/e4_gap1_route_queries.json" if official
                                  else "results/e4_gap1_route_queries_smoke.json"))
    sources = dict(p.split("=", 1) for p in a.preds)
    cfg = QueryConfig(a.pairs_per_type, a.min_dist_frac, a.snap_frac, seed=a.seed,
                      geo=GeospatialConfig(min_blocked_run_px=a.min_blocked_run_px))

    vlm = None
    if a.vlm:
        if backend is not None:
            vlm = backend
        else:
            from src.vlm.backends import Qwen2VLBackend
            vlm = Qwen2VLBackend(a.vlm_model, a.device, a.dtype, a.route_max_pixels, a.load_in_4bit)

    res, queries = run_routes(ds, cfg, sources, vlm, a.limit, Path(a.log_dir) if a.vlm else None)
    result = {
        "experiment": "E4/gap1_routes",
        "meta": {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
                 "split": a.split, "strict_split": strict, "official_run": official, "query_config": config_dict(cfg),
                 "route_prompt_sha": ROUTE_PROMPT_SHA if a.vlm else None, "vlm": None if vlm is None else vlm.name,
                 "sources": sources, "python": platform.python_version(),
                 "note": "ground truth = pixel connectivity of ground-truth dry road; no graph is involved in it"},
        **res,
    }
    qout.parent.mkdir(parents=True, exist_ok=True)
    qout.write_text(json.dumps({"fingerprint": res["queries_fingerprint"], "config": config_dict(cfg),
                                "queries": queries}, indent=0), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    print(f"\nE4 gap-1 routes on {a.split}: {res['n_queries']} queries over {len(res['tile_ids'])} tiles "
          f"{res['queries_by_type']} -> {out}")
    for k, s in res["systems"].items():
        m = s["metrics"]
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        print(f"  {k:22s} error {f(m['connectivity_error_rate'])}  false-passable {f(m['false_passable_rate'])}  "
              f"unanswerable {s['unanswerable']}")


if __name__ == "__main__":
    main()
