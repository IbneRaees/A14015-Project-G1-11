import json
import sys

import numpy as np
import pytest

from data.floodnet import FloodNetDataset
from experiments import run_vlm_zeroshot as e2
from src.utils.cells import CellConfig, cell_states
from src.vlm.backends import MockVLM
from tests.conftest import make_floodnet_tree

COUNTS = {"train": 2, "val": 2, "test": 4}


def oracle_vlm(ds, cell_cfg):
    """A VLM that answers perfectly: looks up the ground-truth state of the cell it is shown."""
    state = {"i": -1, "k": -1}
    truth = {}

    def responder(img, prompt):
        state["k"] += 1
        tile, k = divmod(state["k"], cell_cfg.rows * cell_cfg.cols)
        if tile not in truth:
            truth[tile] = cell_states(ds[tile]["label"], cell_cfg)
        r, c = divmod(k, cell_cfg.cols)
        word = {0: "none", 1: "not_flooded", 2: "flooded"}
        return json.dumps({"buildings": word[int(truth[tile]["building"][r, c])],
                           "roads": word[int(truth[tile]["road"][r, c])]})
    return MockVLM(responder)


def run_main(monkeypatch, tmp_path, backend, extra=()):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["e2", "--data-root", str(tmp_path / "d"), "--smoke-counts", "2,2,4",
                                       "--log-dir", str(tmp_path / "log"), *extra])
    e2.main(backend)
    return json.loads((tmp_path / "results" / "e2_smoke_metrics.json").read_text(encoding="utf-8"))


def test_perfect_vlm_scores_one_and_output_is_marked_unofficial(monkeypatch, tmp_path):
    make_floodnet_tree(tmp_path / "d", COUNTS)
    ds = FloodNetDataset(tmp_path / "d", "test", expected=COUNTS)
    res = run_main(monkeypatch, tmp_path, oracle_vlm(ds, CellConfig()))
    assert res["meta"]["official_run"] is False and res["meta"]["strict_split"] is False
    assert res["n_tiles"] == 4 and res["parse_failure_rate"] == 0.0 and res["mean_tile_error"] == 0.0
    b = res["cell_metrics"]["building"]
    assert b["macro_f1"] == 1.0 and b["accuracy"] == 1.0
    assert res["cell_metrics"]["road"]["f1_flooded"] == 1.0
    assert res["meta"]["decoding"] == "greedy" and res["meta"]["cells"]["rows"] == 4
    assert len(res["per_tile"]["ids"]) == 4 and np.array(res["per_tile"]["cell_cm"]).shape == (4, 2, 3, 3)
    assert res["bootstrap_95ci"]["building"]["macro_f1"]["point"] == 1.0
    assert not (tmp_path / "results" / "e2_vlm_zeroshot_metrics.json").exists()


def test_garbage_vlm_counts_parse_failures_and_scores_poorly(monkeypatch, tmp_path):
    make_floodnet_tree(tmp_path / "d", COUNTS)
    res = run_main(monkeypatch, tmp_path, MockVLM(lambda i, p: "no idea"))
    assert res["parse_failure_rate"] == 1.0                     # every answer unreadable
    assert res["cell_metrics"]["building"]["presence_f1"] == 0.0


def test_resume_skips_finished_tiles(monkeypatch, tmp_path):
    make_floodnet_tree(tmp_path / "d", COUNTS)
    vlm1 = MockVLM(lambda i, p: '{"buildings":"none","roads":"none"}')
    run_main(monkeypatch, tmp_path, vlm1, extra=["--limit", "2"])
    assert len(vlm1.calls) == 2 * 24
    vlm2 = MockVLM(lambda i, p: '{"buildings":"none","roads":"none"}')
    res = run_main(monkeypatch, tmp_path, vlm2, extra=["--limit", "3"])
    assert len(vlm2.calls) == 24 and res["n_tiles"] == 3        # only the third tile was queried
    lines = (tmp_path / "log" / "partial" / "tiles.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
