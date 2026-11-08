import json
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tqdm")

from data.floodnet import FloodNetDataset                                      # noqa: E402
from experiments import run_gap1_routes as g1                                    # noqa: E402
from src.vlm.backends import MockVLM                                             # noqa: E402
from tests.conftest import make_floodnet_tree                                    # noqa: E402
from tests.test_route_benchmark import scene                                     # noqa: E402

COUNTS = {"train": 2, "val": 2, "test": 3}


@pytest.fixture
def world(tmp_path):
    ids = make_floodnet_tree(tmp_path / "d", COUNTS, size=(256, 256))
    for split, id_list in ids.items():                       # replace the toy labels with the road scene
        for tid in id_list:
            Image.fromarray(scene()).save(tmp_path / "d" / split / f"{split}-label-img" / f"{tid}_lab.png")
    perfect, blind = tmp_path / "perfect", tmp_path / "blind"
    perfect.mkdir(), blind.mkdir()
    for tid in ids["test"]:
        cv2.imwrite(str(perfect / f"{tid}.png"), scene())
        miss = scene()
        miss[miss == 3] = 4                                   # flooding not detected
        cv2.imwrite(str(blind / f"{tid}.png"), miss)
    return tmp_path, ids, perfect, blind


def run_main(tmp_path, args, backend=None):
    sys.argv = ["g1", "--data-root", str(tmp_path / "d"), "--smoke-counts", "2,2,3", "--pairs-per-type", "3",
                "--output", str(tmp_path / "out.json"), "--queries-out", str(tmp_path / "q.json"),
                "--log-dir", str(tmp_path / "log"), *args]
    g1.main(backend)
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))


def test_oracle_and_predictions_are_scored_on_the_same_fixed_queries(world):
    tmp, ids, perfect, blind = world
    res = run_main(tmp, ["--preds", f"good_s1={perfect}", "--preds", f"blind_s1={blind}"])
    assert res["meta"]["official_run"] is False and res["n_queries"] == 27 and len(res["tile_ids"]) == 3
    assert res["queries_by_type"] == {"passable": 9, "blocked": 9, "disconnected": 9}
    s = res["systems"]
    assert set(s) == {f"{n}/{k}" for n in ("oracle", "good_s1", "blind_s1") for k in ("graph", "pixel_cc")}
    for k in ("oracle/graph", "oracle/pixel_cc", "good_s1/graph", "good_s1/pixel_cc"):
        assert s[k]["metrics"]["accuracy"] == 1.0 and s[k]["unanswerable"] == 0
    for k in ("blind_s1/graph", "blind_s1/pixel_cc"):
        assert s[k]["metrics"]["false_passable_rate_blocked"] == 1.0 and s[k]["metrics"]["accuracy_passable"] == 1.0
    assert np.array(s["oracle/graph"]["per_tile_counts"]).shape == (3, 3, 2)
    q = json.loads((tmp / "q.json").read_text(encoding="utf-8"))
    assert q["fingerprint"] == res["queries_fingerprint"] and len(q["queries"]) == 27


def test_queries_do_not_depend_on_which_systems_are_scored(world):
    tmp, ids, perfect, blind = world
    a = run_main(tmp, [])
    b = run_main(tmp, ["--preds", f"blind_s1={blind}"])
    assert a["queries_fingerprint"] == b["queries_fingerprint"]


def test_missing_prediction_file_is_an_error_not_a_silent_skip(world):
    tmp, ids, perfect, blind = world
    (perfect / f"{ids['test'][0]}.png").unlink()
    with pytest.raises(FileNotFoundError):
        run_main(tmp, ["--preds", f"good_s1={perfect}"])


def test_prediction_with_out_of_range_values_is_refused(world):
    tmp, ids, perfect, blind = world
    cv2.imwrite(str(perfect / f"{ids['test'][0]}.png"), np.full((256, 256), 200, np.uint8))
    with pytest.raises(ValueError, match="class ids"):
        run_main(tmp, ["--preds", f"good_s1={perfect}"])


def test_direct_vlm_system_is_scored_and_resumes(world):
    tmp, ids, perfect, blind = world
    vlm = MockVLM(lambda img, p: "Yes")
    res = run_main(tmp, ["--vlm"], backend=vlm)
    m = res["systems"]["vlm/direct"]["metrics"]
    assert len(vlm.calls) == 27 and m["false_passable_rate"] == 1.0 and m["accuracy_passable"] == 1.0
    assert res["meta"]["vlm"] == "mock-vlm" and res["meta"]["route_prompt_sha"]
    vlm2 = MockVLM(lambda img, p: "No")
    res2 = run_main(tmp, ["--vlm"], backend=vlm2)
    assert vlm2.calls == []                                   # every query was already answered
    assert res2["systems"]["vlm/direct"]["metrics"] == m


def test_min_blocked_run_changes_graph_answers_but_not_the_queries(world):
    tmp, ids, perfect, blind = world
    strict = run_main(tmp, ["--preds", f"good_s1={perfect}", "--min-blocked-run-px", "20"])
    lax = run_main(tmp, ["--preds", f"good_s1={perfect}", "--min-blocked-run-px", "400"])
    assert strict["queries_fingerprint"] == lax["queries_fingerprint"]                     # same queries for every setting
    assert strict["systems"]["good_s1/graph"]["metrics"]["accuracy"] == 1.0
    assert lax["systems"]["good_s1/graph"]["metrics"]["false_passable_rate_blocked"] == 1.0   # a 30 px flood no longer blocks
    assert lax["systems"]["good_s1/pixel_cc"]["metrics"]["accuracy"] == 1.0                 # pixel_cc has no such knob
    assert lax["meta"]["query_config"]["geo"]["min_blocked_run_px"] == 400
