import json
import cv2
import numpy as np
import pytest

from src.eval.route_benchmark import (GraphAnswerer, PixelCCAnswerer, QueryConfig, ROUTE_PROMPT, TYPES,
                                      VLMAnswerer, answer_tile, bootstrap_route_ci, draw_markers,
                                      generate_queries, paired_bootstrap_route, parse_yes_no,
                                      queries_fingerprint, route_metrics)
from src.vlm.backends import MockVLM

CFG = QueryConfig(pairs_per_type=3, seed=0)


def scene(size=256):
    """Top road: dry | flooded gap | dry. Bottom road: separate, all dry. Geometry scales with `size`."""
    f = size / 256
    lab = np.zeros((size, size), np.uint8)
    r = lambda v: int(round(v * f))
    lab[r(56):r(65), r(10):r(246)] = 4
    lab[r(56):r(65), r(110):r(140)] = 3
    lab[r(196):r(205), r(10):r(246)] = 4
    return lab


def counts_for(label_pred, queries, kind, cfg=CFG, shape=(256, 256)):
    ans = PixelCCAnswerer(label_pred, shape, cfg) if kind == "pixel_cc" else GraphAnswerer(label_pred, shape, "t", cfg)
    return answer_tile(queries, ans)


def test_queries_cover_all_three_types_and_are_consistent_with_the_mask():
    lab = scene()
    qs = generate_queries(lab, "1001", CFG)
    assert {q["type"] for q in qs} == set(TYPES) and len(qs) == 9
    dmin = CFG.min_dist_frac * 256
    for q in qs:
        (ax, ay), (bx, by) = q["a"], q["b"]
        assert lab[int(round(ay)), int(round(ax))] == 4 and lab[int(round(by)), int(round(bx))] == 4
        assert np.hypot(ax - bx, ay - by) >= dmin - 2
        assert q["gt"] == (q["type"] == "passable")
        top_a, top_b = ay < 128, by < 128
        if q["type"] == "disconnected":
            assert top_a != top_b                       # one point on each road
        elif q["type"] == "blocked":
            assert top_a and top_b and (ax < 125) != (bx < 125)     # opposite sides of the flooded gap
        else:
            assert top_a == top_b                       # same road ...
            if top_a:
                assert (ax < 125) == (bx < 125)         # ... and, on the top road, the same side of the gap


def test_queries_are_deterministic_and_depend_on_tile_and_seed():
    lab = scene()
    a = generate_queries(lab, "1001", CFG)
    assert a == generate_queries(lab, "1001", CFG) and queries_fingerprint(a) == queries_fingerprint(list(a))
    assert a != generate_queries(lab, "1002", CFG)
    assert a != generate_queries(lab, "1001", QueryConfig(pairs_per_type=3, seed=1))


def test_tiles_without_two_dry_points_have_no_queries():
    assert generate_queries(np.zeros((64, 64), np.uint8), "t", CFG) == []
    lab = np.zeros((64, 64), np.uint8)
    lab[30:33, 20:40] = 3                                 # flooded road only: no dry centreline to sample
    assert generate_queries(lab, "t", CFG) == []


@pytest.mark.parametrize("kind", ["pixel_cc", "graph"])
def test_oracle_label_map_answers_every_query_correctly(kind):
    lab = scene()
    qs = generate_queries(lab, "1001", CFG)
    counts, unanswerable = counts_for(lab, qs, kind)
    m = route_metrics(counts)
    assert unanswerable == 0 and m["accuracy"] == 1.0 and m["false_passable_rate"] == 0.0


@pytest.mark.parametrize("kind", ["pixel_cc", "graph"])
def test_a_prediction_that_misses_the_flooding_gives_false_passable_answers(kind):
    lab = scene()
    qs = generate_queries(lab, "1001", CFG)
    pred = lab.copy()
    pred[pred == 3] = 4                                   # flood not detected
    m = route_metrics(counts_for(pred, qs, kind)[0])
    assert m["false_passable_rate_blocked"] == 1.0 and m["false_passable_rate_disconnected"] == 0.0
    assert m["accuracy_passable"] == 1.0


@pytest.mark.parametrize("kind", ["pixel_cc", "graph"])
def test_small_offsets_are_tolerated_and_large_ones_are_unanswerable(kind):
    lab = scene()
    qs = generate_queries(lab, "1001", CFG)
    near = np.roll(lab, 3, axis=0)
    counts, un = counts_for(near, qs, kind)
    assert un == 0 and route_metrics(counts)["accuracy"] == 1.0
    far = np.roll(lab, 40, axis=0)                        # roads now >> snap tolerance from the query points
    counts, un = counts_for(far, qs, kind)
    assert un > 0 and counts[0, 1] == 0                   # unanswerable pairs are scored as "no route"


@pytest.mark.parametrize("kind", ["pixel_cc", "graph"])
def test_prediction_at_a_different_resolution_is_scaled_correctly(kind):
    lab = scene(512)
    qs = generate_queries(lab, "1001", CFG)
    small = cv2.resize(lab, (256, 256), interpolation=cv2.INTER_NEAREST)
    counts, un = counts_for(small, qs, kind, shape=(512, 512))
    assert route_metrics(counts)["accuracy"] == 1.0 and un == 0


def test_route_metrics_by_hand():
    m = route_metrics(np.array([[1, 3], [2, 2], [1, 3]]))
    assert m["n_queries"] == 12 and m["accuracy"] == pytest.approx(0.5)
    assert m["connectivity_error_rate"] == pytest.approx(0.5)
    assert m["false_passable_rate"] == pytest.approx(5 / 8) and m["false_blocked_rate"] == pytest.approx(1 / 4)
    assert m["precision"] == pytest.approx(3 / 8) and m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.5) and m["false_passable_rate_blocked"] == pytest.approx(0.5)
    assert route_metrics(np.zeros((3, 2)))["accuracy"] is None


def synthetic_tiles(rng, n, wrong_p):
    out = []
    for _ in range(n):
        c = np.zeros((3, 2), int)
        for t, truth_yes in enumerate((True, False, False)):
            for _ in range(4):
                wrong = rng.random() < wrong_p
                c[t, int(truth_yes != wrong)] += 1
        out.append(c)
    return np.stack(out)


def test_bootstrap_and_paired_comparison():
    rng = np.random.default_rng(0)
    base, better = synthetic_tiles(rng, 150, 0.4), synthetic_tiles(rng, 150, 0.1)
    ci = bootstrap_route_ci(better, "connectivity_error_rate", n_boot=300)
    assert ci["ci95"][0] <= ci["point"] <= ci["ci95"][1] and ci["point"] < 0.2
    d = paired_bootstrap_route(base, better, "false_passable_rate", n_boot=300)
    assert d["diff_point"] < 0 and d["ci95"][1] < 0 and d["p_b_lower_error"] > 0.95
    assert d["relative_reduction_point"] > 0.5 and d["relative_reduction_ci95"][0] > 0
    with pytest.raises(ValueError):
        paired_bootstrap_route(base, better[:10])


@pytest.mark.parametrize("text,expected", [("Yes", True), ("yes.", True), ("No", False), ("no, it is flooded", False),
                                          ("Yes, there is a route", True), ("The answer is no", False),
                                          ("maybe", None), ("both yes and no", None), ("yes and no", True), ("", None), (None, None)])
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) is expected


def test_draw_markers_marks_both_points_and_scales_to_max_side():
    img = np.full((400, 800, 3), 90, np.uint8)
    out = draw_markers(img, [100, 100], [700, 300], max_side=400)
    assert max(out.shape[:2]) == 400
    assert (out[50, 50] == (255, 0, 0)).all() and (out[150, 350] == (0, 0, 255)).all()   # 100,100 -> 50,50
    assert (img == 90).all()                                                                # input untouched


def test_vlm_answerer_counts_unparsable_replies_as_unanswerable():
    lab = scene()
    qs = generate_queries(lab, "1001", CFG)
    replies = iter(["Yes"] * 3 + ["No"] * 3 + ["garbled"] * 3)
    vlm = MockVLM(lambda img, p: next(replies))
    img = np.full((256, 256, 3), 100, np.uint8)
    counts, un = answer_tile(qs, VLMAnswerer(vlm, img))
    assert len(vlm.calls) == 9 and vlm.calls[0][1] == ROUTE_PROMPT and un == 3
    assert counts[0].tolist() == [0, 3] and counts[1].tolist() == [3, 0] and counts[2].tolist() == [3, 0]


def test_graph_agrees_with_pixel_connectivity_on_clean_masks_across_many_tiles():
    """The benchmark's floor: on ground-truth masks the graph must reproduce the pixel-CC truth."""
    lab = scene()
    wrong = total = 0
    for tid in range(1000, 1040):
        qs = generate_queries(lab, str(tid), CFG)
        counts, un = counts_for(lab, qs, "graph")
        m = route_metrics(counts)
        wrong += round(m["connectivity_error_rate"] * m["n_queries"])
        total += m["n_queries"]
        assert un == 0
    assert total > 300 and wrong == 0


def test_undefined_metrics_give_nulls_not_nan():
    only_passable = np.zeros((5, 3, 2), int)
    only_passable[:, 0, 1] = 3                                  # no blocked/disconnected queries at all
    ci = bootstrap_route_ci(only_passable, "false_passable_rate", n_boot=50)
    assert ci == {"point": None, "ci95": None}
    d = paired_bootstrap_route(only_passable, only_passable, "false_passable_rate", n_boot=50)
    assert d["diff_point"] is None and d["ci95"] is None
    json.dumps([ci, d], allow_nan=False)
    assert bootstrap_route_ci(np.zeros((3, 3, 2), int), "accuracy", n_boot=20) == {"point": None, "ci95": None}
