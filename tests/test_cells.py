import numpy as np
import pytest

from src.utils.cells import (ABSENT, DRY, FLOODED, CellAccumulator, CellConfig, bootstrap_ci,
                             cell_boxes_for, cell_states, paired_bootstrap_diff, summarize_cells,
                             task_scores, tile_confusions, tile_error_rate)

CFG = CellConfig(rows=2, cols=2, building_min_frac=0.1, road_min_frac=0.1)


def label(h=40, w=40):
    return np.zeros((h, w), np.uint8)


def test_states_from_majority_and_presence():
    m = label()
    m[:20, :20] = 1                      # top-left: flooded buildings
    m[:20, 20:] = 2                      # top-right: dry buildings
    m[20:, :20] = 3                      # bottom-left: flooded road
    m[20:, 20:30] = 4                    # bottom-right: dry road (half the cell)
    m[0:2, 0:3] = 2                      # minority dry pixels do not flip the top-left cell
    s = cell_states(m, CFG)
    assert s["building"].tolist() == [[FLOODED, DRY], [ABSENT, ABSENT]]
    assert s["road"].tolist() == [[ABSENT, ABSENT], [FLOODED, DRY]]


def test_presence_threshold_and_tie_goes_to_dry():
    m = label()
    m[0:2, 0:20] = 1                     # 40 px of a 400 px cell = 10%  -> present (>= 0.1)
    m[20:22, 20:40] = 1                  # 40 px of 400 = 10% flooded...
    m[22:24, 20:40] = 2                  # ...plus 40 px dry: tie -> DRY
    s = cell_states(m, CFG)
    assert s["building"][0, 0] == FLOODED and s["building"][1, 1] == DRY
    m2 = label()
    m2[0, 0:19] = 1                      # 19 px: below threshold
    assert cell_states(m2, CFG)["building"][0, 0] == ABSENT


def test_states_do_not_depend_on_resolution():
    m = label(40, 40)
    m[:20, :20] = 1
    big = np.kron(m, np.ones((5, 5), np.uint8))
    assert (cell_states(m, CFG)["building"] == cell_states(big, CFG)["building"]).all()


def test_confusions_and_scores_by_hand():
    gt = {"building": np.array([[2, 1], [1, 0]]), "road": np.zeros((2, 2), int)}
    pr = {"building": np.array([[2, 1], [2, 0]]), "road": np.zeros((2, 2), int)}
    cm = tile_confusions(gt, pr)[0]
    assert cm[2, 2] == 1 and cm[1, 1] == 1 and cm[1, 2] == 1 and cm[0, 0] == 1
    s = task_scores(cm)
    assert s["f1_flooded"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 0)) and s["f1_dry"] == pytest.approx(2 / 3)
    assert s["presence_f1"] == pytest.approx(1.0) and s["accuracy"] == pytest.approx(0.75)
    assert tile_error_rate(gt, pr) == pytest.approx(1 / 3)


def test_tile_error_rate_ignores_empty_tiles():
    z = {"building": np.zeros((2, 2), int), "road": np.zeros((2, 2), int)}
    assert tile_error_rate(z, z) == 0.0


def test_accumulator_and_summary_handle_missing_classes():
    acc = CellAccumulator()
    g = {"building": np.array([[2, 2]]), "road": np.zeros((1, 2), int)}
    acc.update(g, g)
    s = acc.summary()
    assert s["building"]["f1_dry"] is None and s["building"]["f1_flooded"] == 1.0   # state 2 = FLOODED
    assert s["building"]["macro_f1"] == 1.0 and s["road"]["macro_f1"] is None
    assert s["building"]["support"] == {"absent": 0, "dry": 0, "flooded": 2}


def test_boxes_cover_the_image_row_major():
    b = cell_boxes_for((30, 40), CellConfig(rows=3, cols=2))
    assert len(b) == 6 and b[0] == [0, 0, 20, 10] and b[-1] == [20, 20, 40, 30]


def per_tile_cms(rng, n, acc):
    out = []
    for _ in range(n):
        gt = {t: rng.integers(0, 3, (4, 6)) for t in ("building", "road")}
        pr = {t: np.where(rng.random((4, 6)) < acc, gt[t], rng.integers(0, 3, (4, 6))) for t in gt}
        out.append(tile_confusions(gt, pr))
    return np.stack(out)


def test_bootstrap_ci_contains_point_and_paired_diff_detects_a_better_system():
    rng = np.random.default_rng(0)
    weak, strong = per_tile_cms(rng, 200, 0.5), per_tile_cms(np.random.default_rng(1), 200, 0.9)
    ci = bootstrap_ci(strong, n_boot=300)
    assert ci["ci95"][0] <= ci["point"] <= ci["ci95"][1]
    d = paired_bootstrap_diff(weak, strong, n_boot=300)     # note: different gt draws, but shapes match
    assert d["diff_point"] > 0 and d["p_b_better"] > 0.95 and d["ci95"][0] > 0
    with pytest.raises(ValueError):
        paired_bootstrap_diff(weak, strong[:10])
