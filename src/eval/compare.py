"""Assemble the paper tables from the experiment JSONs.

Inputs (all under `results_dir`; anything missing is reported, never invented):
  e1_baseline_metrics.json        E1  single-pass YOLOv8x-Seg          (run_baseline)
  e2_vlm_zeroshot_metrics.json    E2  zero-shot VLM surrogate          (run_vlm_zeroshot)
  e3_proposed_metrics.json        E3  full pipeline                    (run_e3)
  e5_no_gap{1,2,3}_metrics.json   E5  ablations                        (run_e3 --ablation)
  e4_gap1_routes.json             Gap 1 route-query benchmark          (run_gap1_routes)
  e4_gap2_calibration.json        Gap 2 calibration comparison         (not yet written; schema below)

Gap 2 file (run_gap2_calibration): {"systems": {name: {"ece": {"mean", "std"}, ...}}, "runs": [{"tile_ids": [...],
"systems": {name: {"per_tile_bins": ...}}}]} for name in yolo_instance_conf, softmax_raw, softmax_ts, evidential,
evidential_ts. A bare {"systems": {name: {"ece": number}}} also works, without the significance test.

Statistics: mean +/- std over seeds where an experiment has several; significance by paired
bootstrap over tiles (all systems are scored on the same test tiles). Seeds are pooled when
resampling tiles.

"Addressed?" is a suggestion from a fixed rule, written next to the table so it can be audited:
    Fully      every target met AND the improvement over the base is significant
    Partially  the improvement is significant OR at least one target is met (but not Fully)
    Not        neither
For Gap 2, significance is a paired bootstrap of ECE over tiles (needs the per-tile bins in e4_gap2_calibration.json);
without them "improvement" falls back to a lower mean ECE.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.components.calibration import paired_bootstrap_ece
from src.eval.route_benchmark import paired_bootstrap_route, route_metrics
from src.utils.cells import TASK_NAMES, paired_bootstrap_diff

FILES = {
    "E1": "e1_baseline_metrics.json", "E2": "e2_vlm_zeroshot_metrics.json", "E3": "e3_proposed_metrics.json",
    "E5/no_gap1": "e5_no_gap1_metrics.json", "E5/no_gap2": "e5_no_gap2_metrics.json",
    "E5/no_gap3": "e5_no_gap3_metrics.json",
}
LABELS = {"E1": "E1 YOLOv8x-Seg", "E2": "E2 VLM zero-shot", "E3": "E3 proposed", "E5/no_gap1": "E3 w/o Gap 1 (graph)",
          "E5/no_gap2": "E3 w/o Gap 2 (evidential)", "E5/no_gap3": "E3 w/o Gap 3 (router)"}
ROUTES_FILE, GAP2_FILE = "e4_gap1_routes.json", "e4_gap2_calibration.json"

DEFAULT_CRITERIA = {
    "gap1": {"base_system": "vlm/direct", "our_system": "e3/graph", "connectivity_error_max": 0.08,
             "false_passable_reduction_min": 0.15},
    "gap2": {"base_system": "yolo_instance_conf", "strong_baseline": "softmax_ts", "our_system": "evidential",
             "ece_max": 0.05},
    "gap3": {"base": "E2", "ablation": "E5/no_gap3", "ours": "E3", "latency_max_s": 2.0, "building_macro_f1_min": 0.82},
}

MS = Optional[Tuple[float, Optional[float]]]


def ms(x: Any) -> MS:
    if isinstance(x, dict) and "mean" in x:
        return (x["mean"], x.get("std")) if x["mean"] is not None else None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return (float(x), None)
    return None


def dig(d: Any, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(v: MS, nd: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v[0]:.{nd}f}" if v[1] is None else f"{v[0]:.{nd}f} ± {v[1]:.{nd}f}"


def fmt_ci(ci, nd: int = 3) -> str:
    return "—" if ci is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


# ------------------------------------------------------------------ loading
@dataclass
class System:
    key: str
    label: str
    n_runs: int
    official: bool
    cell: Dict[str, Dict[str, MS]]
    pixel: Dict[str, MS]
    latency: MS
    vram: MS
    escalation: MS
    head_ece: MS
    ids: List[str] = field(default_factory=list)
    cms: Optional[np.ndarray] = None            # (N, 2, 3, 3), summed over seeds
    lat_tile: Optional[np.ndarray] = None       # (N,), mean over seeds


def _official(d: dict) -> bool:
    m = d.get("meta", {})
    return bool(m.get("strict_split", False)) and m.get("split") == "test" and m.get("official_run", True) is not False


def _pool(runs: List[dict]):
    per = [r.get("per_tile") for r in runs]
    if not per or any(p is None for p in per):
        return [], None, None
    ids = per[0]["ids"]
    if any(p["ids"] != ids for p in per[1:]):
        raise ValueError("runs of one experiment cover different tiles")
    cms = np.sum([np.array(p["cell_cm"]) for p in per], axis=0)
    lat = np.mean([p["latency_s"] for p in per], axis=0) if all("latency_s" in p for p in per) else None
    return ids, cms, lat


def load_system(results_dir, key: str) -> Optional[System]:
    path = Path(results_dir) / FILES[key]
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    src = d.get("aggregate_mean_std") or d
    runs = d.get("runs") or [d]
    ids, cms, lat = _pool(runs)
    cell = {t: {m: ms(dig(src, "cell_metrics", t, m))
                for m in ("macro_f1", "f1_flooded", "f1_dry", "presence_f1", "accuracy")} for t in TASK_NAMES}
    pixel = {f"{t}_{m}": ms(dig(src, "metrics", "tasks", t, m)) for t in TASK_NAMES for m in ("f1", "miou")}
    pixel["miou_all"] = ms(dig(src, "metrics", "miou_all"))
    return System(key, LABELS[key], int(dig(d, "meta", "n_runs") or 1), _official(d), cell, pixel,
                  ms(dig(src, "latency", "mean_s")), ms(src.get("vram_peak_mb")),
                  ms(dig(src, "routing", "escalation_rate")), ms(dig(src, "calibration", "head_ece")),
                  ids, cms, lat)


def read_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# ------------------------------------------------------------------ statistics
def paired_bootstrap_scalar(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of b minus mean of a over resampled tiles (same tiles for both)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("arrays must cover the same tiles")
    rng = np.random.default_rng(seed)
    d = np.array([b[i].mean() - a[i].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))])
    return {"diff_point": float(b.mean() - a.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_b_lower": float((d < 0).mean())}


def _align(a: System, b: System):
    """Per-tile arrays of both systems restricted to their common tiles."""
    if a.cms is None or b.cms is None:
        return None
    common = sorted(set(a.ids) & set(b.ids))
    if not common:
        return None
    ia, ib = {t: i for i, t in enumerate(a.ids)}, {t: i for i, t in enumerate(b.ids)}
    xa, xb = [ia[t] for t in common], [ib[t] for t in common]
    la = a.lat_tile[xa] if a.lat_tile is not None else None
    lb = b.lat_tile[xb] if b.lat_tile is not None else None
    # pooled seeds make cms sums; rates are ratios, so the scale does not matter for F1
    return a.cms[xa], b.cms[xb], la, lb, len(common)


def cell_diff(a: System, b: System, task: str, metric: str, n_boot: int, seed: int) -> Optional[dict]:
    al = _align(a, b)
    if al is None:
        return None
    return {**paired_bootstrap_diff(al[0], al[1], task, metric, n_boot, seed), "n_tiles": al[4]}


def latency_diff(a: System, b: System, n_boot: int, seed: int) -> Optional[dict]:
    al = _align(a, b)
    if al is None or al[2] is None or al[3] is None:
        return None
    return {**paired_bootstrap_scalar(al[2], al[3], n_boot, seed), "n_tiles": al[4]}


# ------------------------------------------------------------------ route families
def parse_spec(spec: str) -> Tuple[str, str]:
    family, kind = spec.split("/", 1)
    return family, kind


def route_family(routes: Optional[dict], spec: str) -> Optional[dict]:
    """Systems named <family>[_s<seed>]/<kind> pooled into one entry (counts summed over seeds)."""
    if not routes:
        return None
    family, kind = parse_spec(spec)
    pat = re.compile(rf"^{re.escape(family)}(_s\d+)?/{re.escape(kind)}$")
    keys = sorted(k for k in routes["systems"] if pat.match(k))
    if not keys:
        return None
    counts = np.sum([np.array(routes["systems"][k]["per_tile_counts"]) for k in keys], axis=0)
    per_seed = [routes["systems"][k]["metrics"] for k in keys]
    return {"keys": keys, "counts": counts, "per_seed": per_seed, "n_seeds": len(keys),
            "unanswerable": sum(routes["systems"][k]["unanswerable"] for k in keys)}


def route_ms(fam: dict, metric: str) -> MS:
    vals = [m[metric] for m in fam["per_seed"] if m[metric] is not None]
    if not vals:
        return None
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else None)


# ------------------------------------------------------------------ tables
@dataclass
class Table:
    key: str
    title: str
    headers: List[str]
    rows: List[List[str]]
    notes: List[str] = field(default_factory=list)


def status(targets: Sequence[bool], improves: bool) -> str:
    if all(targets) and improves:
        return "Fully"
    return "Partially" if (improves or any(targets)) else "Not"


def main_table(sys_: Dict[str, System]) -> Table:
    rows = []
    for k in ("E1", "E2", "E3"):
        s = sys_.get(k)
        if s is None:
            rows.append([LABELS[k]] + ["NOT RUN"] + ["—"] * 9)
            continue
        rows.append([s.label, str(s.n_runs), fmt(s.cell["building"]["macro_f1"]), fmt(s.cell["building"]["f1_flooded"]),
                     fmt(s.cell["road"]["macro_f1"]), fmt(s.cell["road"]["f1_flooded"]), fmt(s.latency, 3),
                     fmt(s.vram, 0), fmt(s.pixel.get("building_f1")), fmt(s.pixel.get("road_f1"))])
    return Table("main", "Grid-cell and pixel results on the FloodNet test split",
                 ["System", "Runs", "Bldg macro-F1", "Bldg flooded-F1", "Road macro-F1", "Road flooded-F1",
                  "Latency (s/tile)", "VRAM (MB)", "Bldg pixel F1", "Road pixel F1"], rows,
                 ["Cell metrics use the shared 4x6 grid protocol; pixel metrics are undefined for E2 (text output).",
                  "Mean ± sample std over seeds; E2 is a single deterministic run."])


def significance_table(sys_: Dict[str, System], n_boot: int, seed: int) -> Table:
    rows = []
    for a, b in (("E1", "E3"), ("E2", "E3")):
        sa, sb = sys_.get(a), sys_.get(b)
        for task in TASK_NAMES:
            r = cell_diff(sa, sb, task, "macro_f1", n_boot, seed) if sa and sb else None
            rows.append([f"{a} → {b}", f"{task} macro-F1", *(_diff_cells(r))])
        r = latency_diff(sa, sb, n_boot, seed) if sa and sb else None
        rows.append([f"{a} → {b}", "latency (s)", *(_diff_cells(r, lower_better=True))])
    return Table("significance", "Paired bootstrap over tiles (B minus A)",
                 ["Comparison", "Metric", "Δ (B−A)", "95% CI", "P(B better)", "Tiles"], rows,
                 [f"{n_boot} resamples of the test tiles, same tiles for both systems. Seeds are pooled."])


def _diff_cells(r: Optional[dict], lower_better: bool = False) -> List[str]:
    if r is None:
        return ["—", "—", "—", "—"]
    p = r["p_b_lower"] if lower_better else r["p_b_better"]
    return [f"{r['diff_point']:+.4f}", fmt_ci(r["ci95"], 4), f"{p:.3f}", str(r["n_tiles"])]


def ablation_table(sys_: Dict[str, System], n_boot: int, seed: int) -> Table:
    e3 = sys_.get("E3")
    rows = []
    for k in ("E3", "E5/no_gap1", "E5/no_gap2", "E5/no_gap3"):
        s = sys_.get(k)
        if s is None:
            rows.append([LABELS[k], "NOT RUN"] + ["—"] * 7)
            continue
        d = {}
        if e3 and k != "E3":
            d["b"], d["r"] = cell_diff(e3, s, "building", "macro_f1", n_boot, seed), cell_diff(e3, s, "road", "macro_f1", n_boot, seed)
            d["l"] = latency_diff(e3, s, n_boot, seed)
        cd = lambda r: "—" if not r else f"{r['diff_point']:+.4f} {fmt_ci(r['ci95'], 3)}"
        rows.append([s.label, "yes", fmt(s.cell["building"]["macro_f1"]), fmt(s.cell["road"]["macro_f1"]), fmt(s.latency, 3),
                     fmt(s.escalation, 3), fmt(s.head_ece, 3), cd(d.get("b")) if d else "—", cd(d.get("l")) if d else "—"])
    return Table("ablation", "Ablations: removing one gap component at a time",
                 ["Variant", "Ran", "Bldg macro-F1", "Road macro-F1", "Latency (s)", "Escalation", "Head ECE",
                  "Δ bldg F1 vs E3", "Δ latency vs E3"], rows,
                 ["Δ = variant minus E3 with a paired-bootstrap 95% CI; a negative F1 delta means removing the component hurts."])


def gap1_detail_table(routes: Optional[dict]) -> Table:
    headers = ["System", "Seeds", "Conn. error", "False-passable", "FP (blocked)", "FP (disconnected)",
               "Acc. passable", "Unanswerable"]
    if not routes:
        return Table("gap1_detail", "Route-query benchmark", headers, [["NOT RUN"] + ["—"] * 7])
    fams = sorted({re.sub(r"_s\d+(?=/)", "", k) for k in routes["systems"]})
    rows = []
    for spec in fams:
        f = route_family(routes, spec)
        rows.append([spec, str(f["n_seeds"])] + [fmt(route_ms(f, m), 3) for m in (
            "connectivity_error_rate", "false_passable_rate", "false_passable_rate_blocked",
            "false_passable_rate_disconnected", "accuracy_passable")] + [str(f["unanswerable"])])
    q = routes.get("queries_by_type", {})
    return Table("gap1_detail", "Route-query benchmark (ground truth = pixel connectivity of ground-truth dry road)",
                 headers, rows, [f"{routes.get('n_queries')} queries over {len(routes.get('tile_ids', []))} tiles {q}.",
                                 "oracle/* rows score the ground-truth masks themselves: oracle/graph is the error floor of the graph abstraction."])


# ------------------------------------------------------------------ gap resolution
def _gap1(routes, crit, n_boot, seed) -> dict:
    row = {"gap": "Gap 1", "method": "Geospatial graph: route passability from road topology", "missing": []}
    if routes is None:
        row.update(base="NOT RUN", ours="NOT RUN", addressed="Not evaluable", evidence="results/e4_gap1_routes.json missing")
        row["missing"].append(ROUTES_FILE)
        return row
    base, ours = route_family(routes, crit["base_system"]), route_family(routes, crit["our_system"])
    if ours is None:
        row.update(base="—", ours="NOT RUN", addressed="Not evaluable", evidence=f"no '{crit['our_system']}' in routes file")
        return row
    err, fp = route_ms(ours, "connectivity_error_rate"), route_ms(ours, "false_passable_rate")
    row["ours"] = f"conn. error {fmt(err, 3)}, false-passable {fmt(fp, 3)}"
    if base is None:
        row.update(base=f"NOT RUN ({crit['base_system']})", addressed="Not evaluable",
                   evidence=f"base '{crit['base_system']}' missing; rerun run_gap1_routes with --vlm")
        row["missing"].append(crit["base_system"])
        return row
    be, bf = route_ms(base, "connectivity_error_rate"), route_ms(base, "false_passable_rate")
    row["base"] = f"conn. error {fmt(be, 3)}, false-passable {fmt(bf, 3)}"
    pe = paired_bootstrap_route(base["counts"], ours["counts"], "connectivity_error_rate", n_boot, seed)
    pf = paired_bootstrap_route(base["counts"], ours["counts"], "false_passable_rate", n_boot, seed)
    if pe["ci95"] is None or pf["ci95"] is None:
        row.update(addressed="Not evaluable", evidence="route metrics are undefined on these queries "
                                                       "(no blocked/disconnected pairs were sampled)")
        return row
    improves = pe["ci95"][1] < 0
    red = pf["relative_reduction_point"]
    targets = [err[0] <= crit["connectivity_error_max"], red is not None and red >= crit["false_passable_reduction_min"]]
    row["addressed"] = status(targets, improves)
    row["evidence"] = (f"Δerror {pe['diff_point']:+.3f} {fmt_ci(pe['ci95'])}; false-passable reduction "
                       f"{'n/a' if red is None else f'{red:.0%}'} {fmt_ci(pf['relative_reduction_ci95'], 2)}; "
                       f"targets: error<={crit['connectivity_error_max']} {'met' if targets[0] else 'missed'}, "
                       f"reduction>={crit['false_passable_reduction_min']:.0%} {'met' if targets[1] else 'missed'}; "
                       f"improvement {'significant' if improves else 'not significant'}")
    return row


def gap2_bins(gap2: Optional[dict], name: str) -> Optional[np.ndarray]:
    """Per-tile calibration bins of one system, summed over seeds: (N, n_bins, 3), or None if not stored."""
    runs = (gap2 or {}).get("runs") or []
    if not runs:
        return None
    arrs = []
    for r in runs:
        b = dig(r, "systems", name, "per_tile_bins")
        if b is None:
            return None
        arrs.append(np.array(b))
    ids = runs[0].get("tile_ids")
    if any(r.get("tile_ids") != ids for r in runs[1:]):
        raise ValueError("Gap 2 runs cover different tiles")
    return np.sum(arrs, axis=0)


def _gap2(gap2: Optional[dict], e3: Optional[System], crit, n_boot: int = 1000, seed: int = 0) -> dict:
    row = {"gap": "Gap 2", "method": "Evidential (Dirichlet) head + verifier", "missing": []}
    ours = ms(dig(gap2, "systems", crit["our_system"], "ece")) or (e3.head_ece if e3 else None)
    row["ours"] = f"ECE {fmt(ours, 4)}" if ours else "NOT RUN"
    base = ms(dig(gap2, "systems", crit["base_system"], "ece"))
    strong_name = crit["strong_baseline"]
    strong = ms(dig(gap2, "systems", strong_name, "ece"))
    if base is None:
        row.update(base=f"NOT RUN ({crit['base_system']})", addressed="Not evaluable",
                   evidence="results/e4_gap2_calibration.json missing or incomplete; E3's head ECE has no baseline to beat yet")
        row["missing"].append(GAP2_FILE)
        return row
    row["base"] = f"ECE {fmt(base, 4)}"
    if ours is None:
        row.update(addressed="Not evaluable", evidence="no evidential ECE (run E3 or the Gap 2 experiment)")
        return row
    b_base, b_ours, b_strong = (gap2_bins(gap2, n) for n in (crit["base_system"], crit["our_system"], strong_name))
    paired = paired_bootstrap_ece(b_base, b_ours, n_boot, seed) if b_base is not None and b_ours is not None else None
    if paired and paired["ci95"] is not None:
        improves = paired["ci95"][1] < 0
        vs_base = f"Δ ECE vs base {paired['diff_point']:+.4f} {fmt_ci(paired['ci95'], 4)} ({'significant' if improves else 'not significant'})"
    else:
        improves = ours[0] < base[0]
        vs_base = f"lower than base: {improves} (no per-tile data, so no significance test)"
    target = ours[0] <= crit["ece_max"]
    row["addressed"] = status([target], improves)
    ev = f"ECE target <= {crit['ece_max']} {'met' if target else 'missed'}; {vs_base}"
    ps = paired_bootstrap_ece(b_strong, b_ours, n_boot, seed) if b_strong is not None and b_ours is not None else None
    if ps and ps["ci95"] is not None:
        ev += f"; vs {strong_name} ({fmt(strong, 4)}): Δ ECE {ps['diff_point']:+.4f} {fmt_ci(ps['ci95'], 4)}"
    else:
        ev += f"; stronger baseline {strong_name}: {fmt(strong, 4)}"
    row["evidence"] = ev
    return row


def gap2_detail_table(gap2: Optional[dict]) -> Table:
    from src.eval.gap2 import SYSTEMS
    headers = ["System", "ECE (covered px)", "ECE (all px)", "Accuracy", "NLL", "Brier", "AURC (entropy)",
               "AURC (vacuity)", "Temperature"]
    if not gap2:
        return Table("gap2_detail", "Calibration comparison", headers, [["NOT RUN"] + ["—"] * 8])
    temps = {"softmax_ts": "softmax", "softmax_raw": None, "evidential_ts": "evidential"}
    rows = []
    for name in SYSTEMS:
        sy = gap2["systems"].get(name) or {}
        t = None
        if temps.get(name):
            vals = [r["temperatures"][temps[name]] for r in gap2.get("runs", []) if "temperatures" in r]
            t = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else None) if vals else None
        rows.append([name] + [fmt(ms(sy.get(k)), 4) for k in ("ece", "ece_all", "accuracy", "nll", "brier", "aurc_entropy",
                                                              "aurc_vacuity")] + [fmt(t, 3)])
    return Table("gap2_detail", "Calibration comparison on the test split", headers, rows,
                 ["Covered pixels are those a YOLO instance claims (conf >= map_conf); every system is scored there, so the comparison is like for like. "
                  "All-pixel, NLL and Brier are defined for the heads only.",
                  "*_ts rows apply a temperature fitted on the validation split."])


def _gap3(sys_: Dict[str, System], crit, n_boot, seed) -> dict:
    row = {"gap": "Gap 3", "method": "Entropy router: fast path first, VLM only on uncertain regions", "missing": []}
    ours, base, abl = sys_.get(crit["ours"]), sys_.get(crit["base"]), sys_.get(crit.get("ablation", ""))
    row["ours"] = (f"latency {fmt(ours.latency, 3)} s, bldg macro-F1 {fmt(ours.cell['building']['macro_f1'], 4)}"
                   if ours else "NOT RUN")
    if ours is None or base is None:
        miss = [FILES[k] for k, s in ((crit["ours"], ours), (crit["base"], base)) if s is None]
        row.update(base="NOT RUN" if base is None else f"latency {fmt(base.latency, 3)} s", addressed="Not evaluable",
                   evidence="missing: " + ", ".join(miss))
        row["missing"] += miss
        return row
    row["base"] = f"latency {fmt(base.latency, 3)} s, bldg macro-F1 {fmt(base.cell['building']['macro_f1'], 4)}"
    ld, fd = latency_diff(base, ours, n_boot, seed), cell_diff(base, ours, "building", "macro_f1", n_boot, seed)
    improves = bool(ld and ld["ci95"][1] < 0)
    t_lat = ours.latency is not None and ours.latency[0] <= crit["latency_max_s"]
    t_f1 = ours.cell["building"]["macro_f1"] is not None and ours.cell["building"]["macro_f1"][0] >= crit["building_macro_f1_min"]
    row["addressed"] = status([t_lat, t_f1], improves)
    row["evidence"] = (f"Δlatency {'n/a' if not ld else f'{ld['diff_point']:+.3f}s {fmt_ci(ld['ci95'])}'}; "
                       f"Δbldg F1 {'n/a' if not fd else f'{fd['diff_point']:+.4f} {fmt_ci(fd['ci95'], 4)}'}; "
                       f"latency<={crit['latency_max_s']}s {'met' if t_lat else 'missed'}, "
                       f"F1>={crit['building_macro_f1_min']} {'met' if t_f1 else 'missed'}; "
                       f"latency reduction {'significant' if improves else 'not significant'}")
    if abl is not None:                  # internal control: same pipeline with the VLM on every cell
        al, af = latency_diff(abl, ours, n_boot, seed), cell_diff(abl, ours, "building", "macro_f1", n_boot, seed)
        row["evidence"] += (f"; vs {abl.label}: Δlatency {'n/a' if not al else f'{al['diff_point']:+.3f}s {fmt_ci(al['ci95'])}'}, "
                            f"Δbldg F1 {'n/a' if not af else f'{af['diff_point']:+.4f} {fmt_ci(af['ci95'], 4)}'}")
    return row


def gap_resolution_table(rows: List[dict]) -> Table:
    return Table("gap_resolution", "Gap-resolution table",
                 ["Gap", "Method / component", "Base paper result", "Our result", "Addressed? (Fully / Partially / Not)", "Evidence"],
                 [[r["gap"], r["method"], r["base"], r["ours"], r["addressed"], r["evidence"]] for r in rows],
                 ["Base paper = surrogate (zero-shot Qwen2-VL) since SkyEyeGPT has no public checkpoint.",
                  "Addressed? follows a fixed rule: Fully = all targets met and a significant improvement; Partially = a significant "
                  "improvement or at least one target met; Not = neither. Treat it as a suggestion, not a verdict."])


# ------------------------------------------------------------------ report
def merge_criteria(user: Optional[dict]) -> dict:
    out = {k: dict(v) for k, v in DEFAULT_CRITERIA.items()}
    for k, v in (user or {}).items():
        out.setdefault(k, {}).update(v or {})
    return out


def build_report(results_dir, criteria: Optional[dict] = None, n_boot: int = 1000, seed: int = 0) -> dict:
    results_dir = Path(results_dir)
    crit = merge_criteria(criteria)
    sys_ = {k: s for k in FILES if (s := load_system(results_dir, k)) is not None}
    routes, gap2 = read_json(results_dir / ROUTES_FILE), read_json(results_dir / GAP2_FILE)
    missing = [FILES[k] for k in FILES if k not in sys_]
    missing += [f for f, d in ((ROUTES_FILE, routes), (GAP2_FILE, gap2)) if d is None]
    warnings = [f"{FILES[k]} is not an official test run (smoke/limited/val); do not cite its numbers"
                for k, s in sys_.items() if not s.official]
    if routes and not (routes.get("meta", {}).get("official_run", False)):
        warnings.append(f"{ROUTES_FILE} is not an official run")
    gaps = [_gap1(routes, crit["gap1"], n_boot, seed), _gap2(gap2, sys_.get("E3"), crit["gap2"], n_boot, seed),
            _gap3(sys_, crit["gap3"], n_boot, seed)]
    tables = [gap_resolution_table(gaps), main_table(sys_), significance_table(sys_, n_boot, seed),
              ablation_table(sys_, n_boot, seed), gap1_detail_table(routes), gap2_detail_table(gap2)]
    return {"tables": tables, "gaps": gaps, "missing": missing, "warnings": warnings,
            "criteria": crit, "n_boot": n_boot, "seed": seed}


# ------------------------------------------------------------------ rendering
def to_markdown(t: Table) -> str:
    esc = lambda c: str(c).replace("|", "\\|")
    lines = [f"### {t.title}", "", "| " + " | ".join(t.headers) + " |", "|" + "|".join(["---"] * len(t.headers)) + "|"]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in t.rows]
    lines += [""] + [f"*{n}*" for n in t.notes]
    return "\n".join(lines) + "\n"


_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "±": r"$\pm$", "Δ": r"$\Delta$", "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\rightarrow$", "—": "--", "−": "-"}


def tex_escape(s: str) -> str:
    return "".join(_TEX.get(c, c) for c in str(s))


def to_latex(t: Table, drop: Sequence[str] = ()) -> str:
    keep = [i for i, h in enumerate(t.headers) if h not in drop]
    wide = len(keep) > 6
    env = "table*" if wide else "table"
    cols = "l" + "c" * (len(keep) - 1)
    out = [f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize", f"\\caption{{{tex_escape(t.title)}}}",
           f"\\label{{tab:{t.key}}}", f"\\begin{{tabular}}{{{cols}}}", "\\hline",
           " & ".join(tex_escape(t.headers[i]) for i in keep) + " \\\\", "\\hline"]
    out += [" & ".join(tex_escape(r[i]) for i in keep) + " \\\\" for r in t.rows]
    out += ["\\hline", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(out) + "\n"


def write_report(report: dict, out_dir) -> Path:
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    md = ["# Comparison of E1–E5", ""]
    if report["warnings"]:
        md += ["> **Warnings**"] + [f"> - {w}" for w in report["warnings"]] + [""]
    if report["missing"]:
        md += ["> **Not yet available:** " + ", ".join(report["missing"]), ""]
    for t in report["tables"]:
        md.append(to_markdown(t))
        drop = ("Evidence",) if t.key == "gap_resolution" else ()
        (out / "tables" / f"{t.key}.tex").write_text(to_latex(t, drop), encoding="utf-8")
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(
        {**report, "tables": [t.__dict__ for t in report["tables"]]}, indent=1, default=float), encoding="utf-8")
    return out / "comparison.md"
