import json

import pytest

from experiments.summarize_sweep import summarize


def write(tmp, run, err, fp, fb, fingerprint="abc", system="e1_s42/graph"):
    d = {"queries_fingerprint": fingerprint, "meta": {"query_config": {"geo": {"min_blocked_run_px": run}}},
         "systems": {system: {"metrics": {"n_queries": 100, "connectivity_error_rate": err, "false_passable_rate": fp,
                                          "false_blocked_rate": fb}, "unanswerable": 2}}}
    p = tmp / f"m{run}.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_sweep_picks_the_lowest_error_and_sorts_by_setting(tmp_path):
    paths = [write(tmp_path, 40, 0.20, 0.30, 0.05), write(tmp_path, 10, 0.25, 0.10, 0.30), write(tmp_path, 20, 0.12, 0.15, 0.10)]
    rows, best = summarize(paths, "e1_s42/graph")
    assert [r["min_blocked_run_px"] for r in rows] == [10, 20, 40] and best["min_blocked_run_px"] == 20


def test_ties_go_to_the_smaller_setting_and_mixed_query_sets_are_refused(tmp_path):
    rows, best = summarize([write(tmp_path, 30, 0.1, 0, 0), write(tmp_path, 15, 0.1, 0, 0)], "e1_s42/graph")
    assert best["min_blocked_run_px"] == 15
    with pytest.raises(ValueError, match="different query sets"):
        summarize([write(tmp_path, 10, 0.1, 0, 0, "a"), write(tmp_path, 20, 0.1, 0, 0, "b")], "e1_s42/graph")
    with pytest.raises(KeyError):
        summarize([write(tmp_path, 10, 0.1, 0, 0)], "nope/graph")
