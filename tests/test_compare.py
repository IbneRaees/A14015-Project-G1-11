import json
import re

import numpy as np
import pytest

from src.eval.compare import (FILES, build_report, merge_criteria, paired_bootstrap_scalar, route_family, status,
                              tex_escape, to_latex, write_report)
from src.eval.route_benchmark import route_metrics
from src.utils.cells import summarize_cells
from src.utils.metrics import aggregate_runs, confusion_matrix, summarize
from tests.test_cells import per_tile_cms
from tests.test_route_benchmark import synthetic_tiles

N = 120
IDS = [str(1000 + i) for i in range(N)]


def make_run(rng, acc, lat_mean, ece=None, esc=None):
    cms = per_tile_cms(rng, N, acc)
    gt = rng.integers(0, 10, (40, 40)).astype(np.uint8)
    run = {"cell_metrics": summarize_cells({t: cms[:, i].sum(0) for i, t in enumerate(("building", "road"))}),
           "metrics": summarize(confusion_matrix(gt, gt)),
           "latency": {"mean_s": lat_mean, "std_s": 0.1}, "vram_peak_mb": 9000.0,
           "per_tile": {"ids": IDS, "cell_cm": cms.tolist(),
                        "latency_s": list(np.clip(rng.normal(lat_mean, 0.05, N), 0.01, None))}}
    if esc is not None:
        run["routing"] = {"escalation_rate": esc}
    if ece is not None:
        run["calibration"] = {"head_ece": ece}
    return run


def write_exp(d, key, runs, official=True, flat=False):
    meta = {"split": "test", "strict_split": True, "official_run": official, "n_runs": len(runs)}
    if flat:                                              # E2 layout: one run at top level
        body = {**runs[0], "meta": meta}
    else:
        agg = aggregate_runs([{k: v for k, v in r.items() if k != "per_tile"} for r in runs])
        body = {"meta": meta, "aggregate_mean_std": agg, "runs": runs}
    (d / FILES[key]).write_text(json.dumps(body), encoding="utf-8")


def write_routes(d, base_p, ours_p, official=True):
    rng = np.random.default_rng(5)
    systems = {}

    def add(name, counts):
        systems[name] = {"metrics": route_metrics(counts.sum(0)), "unanswerable": 0, "per_tile_counts": counts.tolist()}
    add("oracle/graph", synthetic_tiles(rng, N, 0.0))
    add("vlm/direct", synthetic_tiles(rng, N, base_p))
    for s in (42, 123, 456):
        add(f"e3_s{s}/graph", synthetic_tiles(rng, N, ours_p))
    (d / "e4_gap1_routes.json").write_text(json.dumps({
        "meta": {"official_run": official}, "n_queries": N * 12, "tile_ids": IDS, "queries_by_type": {"passable": 1},
        "systems": systems}), encoding="utf-8")


@pytest.fixture
def results(tmp_path):
    rng = np.random.default_rng(0)
    write_exp(tmp_path, "E1", [make_run(rng, 0.5, 1.9) for _ in range(3)])
    e2 = make_run(rng, 0.7, 30.0)
    e2.pop("metrics")                                      # a VLM has no pixel metrics
    write_exp(tmp_path, "E2", [e2], flat=True)
    write_exp(tmp_path, "E3", [make_run(rng, 0.9, 1.4, ece=0.03, esc=0.2) for _ in range(3)])
    write_exp(tmp_path, "E5/no_gap1", [make_run(rng, 0.9, 1.3, ece=0.03, esc=0.2)])
    write_exp(tmp_path, "E5/no_gap2", [make_run(rng, 0.85, 1.4, esc=0.3)])
    write_exp(tmp_path, "E5/no_gap3", [make_run(rng, 0.9, 6.0, ece=0.03, esc=1.0)])
    write_routes(tmp_path, base_p=0.4, ours_p=0.03)
    (tmp_path / "e4_gap2_calibration.json").write_text(json.dumps({"systems": {
        "yolo_instance_conf": {"ece": 0.2}, "softmax_ts": {"ece": {"mean": 0.06, "std": 0.01}}, "evidential": {"ece": {"mean": 0.03, "std": 0.005}}}}), encoding="utf-8")
    return tmp_path


def table(report, key):
    return next(t for t in report["tables"] if t.key == key)


def test_status_rule():
    assert status([True, True], True) == "Fully" and status([True, False], True) == "Partially"
    assert status([False], True) == "Partially" and status([True], False) == "Partially"
    assert status([False, False], False) == "Not"


def test_full_report_fills_every_table_from_the_files(results):
    r = build_report(results, {"gap3": {"building_macro_f1_min": 0.5}}, n_boot=200)
    assert r["missing"] == [] and r["warnings"] == []
    gaps = {g["gap"]: g for g in r["gaps"]}
    assert gaps["Gap 1"]["addressed"] == "Fully" and "significant" in gaps["Gap 1"]["evidence"]
    assert gaps["Gap 2"]["addressed"] == "Fully" and "0.0600" in gaps["Gap 2"]["evidence"]       # strong baseline shown
    assert gaps["Gap 3"]["addressed"] == "Fully" and gaps["Gap 3"]["base"].startswith("latency 30.0")
    assert "vs E3 w/o Gap 3 (router)" in gaps["Gap 3"]["evidence"]
    main = table(r, "main")
    assert [row[0] for row in main.rows] == ["E1 YOLOv8x-Seg", "E2 VLM zero-shot", "E3 proposed"]
    assert main.rows[2][1] == "3" and "±" in main.rows[2][2] and main.rows[1][8] == "—"          # E2: no pixel F1
    sig = table(r, "significance")
    e1_e3 = next(row for row in sig.rows if row[0] == "E1 → E3" and row[1] == "building macro-F1")
    assert float(e1_e3[2]) > 0.2 and float(e1_e3[4]) > 0.95 and e1_e3[5] == str(N)
    lat = next(row for row in sig.rows if row[0] == "E2 → E3" and row[1] == "latency (s)")
    assert float(lat[2]) < -20
    abl = table(r, "ablation")
    assert [row[0] for row in abl.rows][0] == "E3 proposed" and abl.rows[3][4].startswith("6.0")
    assert "e3" in [row[0].split("/")[0] for row in table(r, "gap1_detail").rows]


def test_a_weak_result_is_not_called_fully_addressed(tmp_path):
    rng = np.random.default_rng(1)
    write_exp(tmp_path, "E3", [make_run(rng, 0.6, 2.5, ece=0.2)])
    e2 = make_run(rng, 0.6, 2.5)
    e2.pop("metrics")
    write_exp(tmp_path, "E2", [e2], flat=True)
    write_routes(tmp_path, base_p=0.2, ours_p=0.2)                          # no better than the VLM
    (tmp_path / "e4_gap2_calibration.json").write_text(json.dumps({"systems": {
        "yolo_instance_conf": {"ece": 0.18}, "evidential": {"ece": 0.2}}}), encoding="utf-8")
    gaps = {g["gap"]: g for g in build_report(tmp_path, n_boot=200)["gaps"]}
    assert gaps["Gap 1"]["addressed"] == "Not" and "not significant" in gaps["Gap 1"]["evidence"]
    assert gaps["Gap 2"]["addressed"] == "Not"
    assert gaps["Gap 3"]["addressed"] == "Not"                              # same latency, over target, F1 below target


def test_missing_inputs_are_listed_and_never_invented(tmp_path):
    r = build_report(tmp_path, n_boot=50)
    assert len(r["missing"]) == len(FILES) + 2
    assert all(g["addressed"] == "Not evaluable" for g in r["gaps"])
    assert "NOT RUN" in table(r, "main").rows[0][1]
    assert table(r, "gap1_detail").rows[0][0] == "NOT RUN"
    assert report_has_no_numbers_for_missing(r)


def report_has_no_numbers_for_missing(r):
    return not re.search(r"\d\.\d", " ".join(" ".join(t.rows[0]) for t in r["tables"] if t.key in ("main", "gap1_detail")))


def test_partial_inputs_give_partial_rows(results):
    (results / FILES["E2"]).unlink()
    (results / "e4_gap2_calibration.json").unlink()
    g = {x["gap"]: x for x in build_report(results, n_boot=100)["gaps"]}
    assert g["Gap 3"]["addressed"] == "Not evaluable" and FILES["E2"] in g["Gap 3"]["missing"]
    assert g["Gap 2"]["addressed"] == "Not evaluable" and g["Gap 2"]["ours"].startswith("ECE 0.03")
    assert g["Gap 1"]["addressed"] != "Not evaluable"


def test_smoke_files_raise_a_warning(results):
    write_routes(results, 0.4, 0.03, official=False)
    rng = np.random.default_rng(2)
    write_exp(results, "E3", [make_run(rng, 0.9, 1.4)], official=False)
    w = build_report(results, n_boot=50)["warnings"]
    assert any("e3_proposed" in x for x in w) and any("routes" in x for x in w)


def test_runs_over_different_tiles_are_refused(tmp_path):
    rng = np.random.default_rng(3)
    a, b = make_run(rng, 0.9, 1.0), make_run(rng, 0.9, 1.0)
    b["per_tile"]["ids"] = list(reversed(IDS))
    write_exp(tmp_path, "E3", [a, b])
    with pytest.raises(ValueError, match="different tiles"):
        build_report(tmp_path, n_boot=50)


def test_systems_on_partly_different_tiles_are_compared_on_the_overlap(tmp_path):
    rng = np.random.default_rng(4)
    a, b = make_run(rng, 0.5, 2.0), make_run(rng, 0.9, 1.0)
    keep = slice(0, 100)
    b["per_tile"] = {"ids": IDS[keep], "cell_cm": b["per_tile"]["cell_cm"][keep], "latency_s": b["per_tile"]["latency_s"][keep]}
    write_exp(tmp_path, "E1", [a])
    write_exp(tmp_path, "E3", [b])
    sig = table(build_report(tmp_path, n_boot=100), "significance")
    assert sig.rows[0][5] == "100"


def test_route_family_pools_seeds_and_reports_spread(results):
    routes = json.loads((results / "e4_gap1_routes.json").read_text(encoding="utf-8"))
    fam = route_family(routes, "e3/graph")
    assert fam["n_seeds"] == 3 and fam["counts"].shape == (N, 3, 2)
    assert route_family(routes, "vlm/direct")["n_seeds"] == 1 and route_family(routes, "nope/graph") is None
    assert route_family(None, "e3/graph") is None


def test_paired_bootstrap_scalar():
    rng = np.random.default_rng(0)
    a = rng.normal(2.0, 0.2, 200)
    d = paired_bootstrap_scalar(a, a - 1.0)
    assert d["diff_point"] == pytest.approx(-1.0) and d["ci95"][1] < 0 and d["p_b_lower"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap_scalar(a, a[:10])


def test_outputs_are_written_and_latex_is_escaped(results, tmp_path):
    r = build_report(results, n_boot=100)
    out = tmp_path / "cmp"
    md = write_report(r, out)
    text = md.read_text(encoding="utf-8")
    assert "Gap-resolution table" in text and "| Gap 1 |" in text
    data = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert {t["key"] for t in data["tables"]} == {"gap_resolution", "main", "significance", "ablation", "gap1_detail", "gap2_detail"}
    tex = (out / "tables" / "main.tex").read_text(encoding="utf-8")
    assert "\\begin{table*}" in tex and "$\\pm$" in tex and "±" not in tex
    gap_tex = (out / "tables" / "gap_resolution.tex").read_text(encoding="utf-8")
    assert "Evidence" not in gap_tex and "\\begin{table}" in gap_tex
    assert tex_escape("a_b 5% & #1 ≤ 2 —") == r"a\_b 5\% \& \#1 $\leq$ 2 --"


def test_criteria_merge_keeps_unspecified_defaults():
    c = merge_criteria({"gap2": {"ece_max": 0.1}})
    assert c["gap2"]["ece_max"] == 0.1 and c["gap2"]["base_system"] == "yolo_instance_conf" and "gap1" in c


def test_undefined_route_metrics_do_not_crash_the_report(tmp_path):
    only_passable = np.zeros((N, 3, 2), int)
    only_passable[:, 0, 1] = 3
    systems = {k: {"metrics": route_metrics(only_passable.sum(0)), "unanswerable": 0,
                   "per_tile_counts": only_passable.tolist()} for k in ("vlm/direct", "e3_s1/graph")}
    (tmp_path / "e4_gap1_routes.json").write_text(json.dumps({"meta": {"official_run": True}, "n_queries": N * 3,
                                                              "tile_ids": IDS, "systems": systems}), encoding="utf-8")
    g = build_report(tmp_path, n_boot=50)["gaps"][0]
    assert g["addressed"] == "Not evaluable" and "undefined" in g["evidence"]


def gap2_file(ece_by_system, n_tiles=N, seeds=3, seed_rng=0):
    """A Gap 2 result file with per-tile bins whose calibration error matches ece_by_system (as a confidence gap)."""
    from tests.test_gap2_scorer import bins_for
    rng = np.random.default_rng(seed_rng)
    names = ("yolo_instance_conf", "softmax_raw", "softmax_ts", "evidential", "evidential_ts")
    runs = [{"seed": s, "tile_ids": IDS, "temperatures": {"softmax": 1.8, "evidential": 0.7},
             "systems": {n: {"ece": 0.0, "per_tile_bins": bins_for(rng, n_tiles, ece_by_system[n]).tolist()} for n in names}}
            for s in range(seeds)]
    systems = {n: {"ece": {"mean": ece_by_system[n], "std": 0.01}, "ece_all": {"mean": 0.1, "std": None},
                   "accuracy": {"mean": 0.7, "std": 0.01}, "nll": None, "brier": None, "aurc_entropy": {"mean": 0.1, "std": 0.0},
                   "aurc_vacuity": None} for n in names}
    return {"meta": {"official_run": True}, "systems": systems, "runs": runs}


def test_gap2_uses_a_paired_bootstrap_over_tiles_when_bins_are_available(results):
    gap_gaps = {"yolo_instance_conf": 0.25, "softmax_raw": 0.2, "softmax_ts": 0.1, "evidential": 0.0, "evidential_ts": 0.0}
    f = gap2_file(gap_gaps)
    f["systems"]["evidential"]["ece"]["mean"] = 0.03
    (results / "e4_gap2_calibration.json").write_text(json.dumps(f), encoding="utf-8")
    g = {x["gap"]: x for x in build_report(results, n_boot=200)["gaps"]}["Gap 2"]
    assert g["addressed"] == "Fully" and "significant" in g["evidence"] and "vs softmax_ts" in g["evidence"]
    assert "Δ ECE vs base" in g["evidence"]


def test_gap2_not_significant_when_calibration_is_the_same(results):
    same = {n: 0.1 for n in ("yolo_instance_conf", "softmax_raw", "softmax_ts", "evidential", "evidential_ts")}
    f = gap2_file(same)
    f["systems"]["evidential"]["ece"]["mean"] = 0.03
    (results / "e4_gap2_calibration.json").write_text(json.dumps(f), encoding="utf-8")
    g = {x["gap"]: x for x in build_report(results, n_boot=200)["gaps"]}["Gap 2"]
    assert "not significant" in g["evidence"] and g["addressed"] == "Partially"       # target met, no real improvement


def test_gap2_detail_table_lists_all_systems_with_temperatures(results):
    f = gap2_file({"yolo_instance_conf": 0.2, "softmax_raw": 0.2, "softmax_ts": 0.1, "evidential": 0.05, "evidential_ts": 0.04})
    (results / "e4_gap2_calibration.json").write_text(json.dumps(f), encoding="utf-8")
    tbl = table(build_report(results, n_boot=50), "gap2_detail")
    assert [r[0] for r in tbl.rows] == ["yolo_instance_conf", "softmax_raw", "softmax_ts", "evidential", "evidential_ts"]
    assert tbl.rows[2][-1].startswith("1.800") and tbl.rows[0][-1] == "—"
    assert table(build_report(results.parent / "empty_none", n_boot=5) if False else build_report(results, n_boot=5), "gap2_detail").rows[0][1] != "NOT RUN"


def test_gap2_runs_over_different_tiles_are_refused(results):
    f = gap2_file({n: 0.1 for n in ("yolo_instance_conf", "softmax_raw", "softmax_ts", "evidential", "evidential_ts")})
    f["runs"][1]["tile_ids"] = list(reversed(IDS))
    (results / "e4_gap2_calibration.json").write_text(json.dumps(f), encoding="utf-8")
    with pytest.raises(ValueError, match="different tiles"):
        build_report(results, n_boot=20)
