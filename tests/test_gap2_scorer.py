import json
import math

import numpy as np
import pytest

from src.components.calibration import (CalibrationAccumulator, bootstrap_ece_ci, ece_from_bins, fit_temperature,
                                        paired_bootstrap_ece)
from src.eval.gap2 import Gap2Scorer, N_BINS, SYSTEMS, head_probabilities
from src.utils.segmentation import instances_to_semantic_with_conf

K, H, W = 10, 30, 40


def sq(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def test_instances_to_semantic_with_conf_paints_best_instance_and_marks_coverage():
    polys = [sq(0, 0, 20, 20), sq(10, 10, 30, 30)]
    lab, conf, cov = instances_to_semantic_with_conf(polys, [3, 4], [0.9, 0.4], (40, 40))
    assert lab[5, 5] == 3 and conf[5, 5] == pytest.approx(0.9) and cov[5, 5]
    assert lab[15, 15] == 3 and conf[15, 15] == pytest.approx(0.9)          # overlap: higher confidence wins
    assert lab[25, 25] == 4 and conf[25, 25] == pytest.approx(0.4)
    assert lab[35, 35] == 0 and conf[35, 35] == 0 and not cov[35, 35]
    assert instances_to_semantic_with_conf([], [], [], (4, 4))[2].sum() == 0


def make_tile(rng, acc=0.7, logit_scale=8.0, alpha_peak=50.0):
    gt = rng.integers(0, K, (H, W))
    pred = np.where(rng.random((H, W)) < acc, gt, rng.integers(0, K, (H, W)))
    onehot = np.eye(K)[pred].transpose(2, 0, 1)
    logits = (onehot * logit_scale).astype(np.float32)
    alpha = (1 + onehot * (alpha_peak - 1)).astype(np.float32)
    covered = rng.random((H, W)) < 0.5
    ycls = np.where(rng.random((H, W)) < 0.7, gt, rng.integers(0, K, (H, W)))
    yconf = np.where(covered, rng.uniform(0.3, 0.95, (H, W)), 0).astype(np.float32)
    return gt, alpha, logits, ycls, yconf, covered


def run_scorer(n_tiles=12, t_soft=1.0, t_ev=1.0, **kw):
    rng = np.random.default_rng(0)
    sc, bins = Gap2Scorer(), []
    for _ in range(n_tiles):
        gt, a, z, yl, yc, cov = make_tile(rng, **kw)
        bins.append(sc.update(gt, a, z, yl, yc, cov, t_soft, t_ev))
    return sc, bins


def test_all_systems_reported_with_expected_pixel_sets():
    sc, _ = run_scorer()
    s = sc.summarize()
    assert set(s) == set(SYSTEMS)
    y, e = s["yolo_instance_conf"], s["evidential"]
    assert y["ece_all"] is None and y["nll"] is None and y["brier"] is None            # YOLO: covered pixels, no distribution
    assert e["ece_all"] is not None and e["nll"] is not None and e["aurc_vacuity"] is not None
    assert s["softmax_raw"]["aurc_vacuity"] is None
    assert y["n_pixels"] == e["n_pixels"]                                              # same pixel set for the primary comparison
    assert 0.4 < e["coverage_fraction"] < 0.6
    json.dumps(s, allow_nan=False)


def test_overconfident_head_is_fixed_by_temperature_scaling():
    sc, _ = run_scorer(acc=0.7, logit_scale=8.0)
    raw = sc.summarize()["softmax_raw"]["ece"]
    assert raw > 0.2                                                                    # confidence ~1, accuracy ~0.7
    rng = np.random.default_rng(1)
    z, y = [], []
    for _ in range(6):
        gt, a, logits, *_ = make_tile(rng, acc=0.7, logit_scale=8.0)
        z.append(logits.reshape(K, -1).T[::7]); y.append(gt.reshape(-1)[::7])
    t = fit_temperature(np.concatenate(z), np.concatenate(y))
    assert t > 1.5
    sc2, _ = run_scorer(acc=0.7, logit_scale=8.0, t_soft=t)
    assert sc2.summarize()["softmax_ts"]["ece"] < 0.5 * raw


def test_underconfident_evidential_head_is_helped_by_its_temperature():
    sc, _ = run_scorer(acc=0.9, alpha_peak=3.0)                                         # p = 3/12 = 0.25 at the peak
    ev = sc.summarize()["evidential"]
    assert ev["mean_confidence"] < 0.3 and ev["ece"] > 0.4
    sc2, _ = run_scorer(acc=0.9, alpha_peak=3.0, t_ev=0.15)                             # T < 1 sharpens
    assert sc2.summarize()["evidential_ts"]["ece"] < 0.5 * ev["ece"]


def test_per_tile_bins_reproduce_the_aggregate_ece():
    sc, bins = run_scorer()
    s = sc.summarize()
    for name in SYSTEMS:
        total = np.sum([b[name] for b in bins], axis=0)
        assert total.shape == (N_BINS, 3)
        assert ece_from_bins(total) == pytest.approx(s[name]["ece"], abs=1e-9)


def test_proper_scores_at_the_extremes():
    gt = np.zeros((H, W), int)
    alpha_good = np.ones((K, H, W), np.float32)
    alpha_good[0] = 1e6                                                                  # p(true) ~ 1
    logits_uniform = np.zeros((K, H, W), np.float32)
    sc = Gap2Scorer()
    cov = np.ones((H, W), bool)
    sc.update(gt, alpha_good, logits_uniform, gt, np.full((H, W), 0.9, np.float32), cov, 1.0, 1.0)
    s = sc.summarize()
    assert s["evidential"]["nll"] < 1e-3 and s["evidential"]["brier"] < 1e-3
    assert s["softmax_raw"]["nll"] == pytest.approx(math.log(K), rel=1e-4)             # uniform: ln K
    assert s["softmax_raw"]["brier"] == pytest.approx(1 - 1 / K, rel=1e-4)
    assert s["yolo_instance_conf"]["accuracy"] == 1.0 and s["yolo_instance_conf"]["mean_confidence"] == pytest.approx(0.9)


def test_tile_with_no_covered_pixels_is_handled():
    rng = np.random.default_rng(3)
    gt, a, z, yl, yc, cov = make_tile(rng)
    sc = Gap2Scorer()
    bins = sc.update(gt, a, z, yl, yc, np.zeros_like(cov), 1.0, 1.0)
    assert bins["evidential"][:, 0].sum() == 0
    s = sc.summarize()
    assert s["evidential"] == {"ece": None}


def test_head_probabilities_sum_to_one():
    rng = np.random.default_rng(4)
    gt, a, z, *_ = make_tile(rng)
    for name, p in head_probabilities(a, z, 2.0, 0.5).items():
        assert p.shape == (K, H, W) and np.allclose(p.sum(0), 1, atol=1e-4)


def bins_for(rng, n, conf_gap):
    out = []
    for _ in range(n):
        acc = CalibrationAccumulator(N_BINS)
        c = rng.uniform(0.3, 1.0, 400)
        acc.update(c, rng.random(400) < np.clip(c - conf_gap, 0, 1))
        out.append(np.stack([acc.count, acc.conf_sum, acc.correct_sum], axis=1))
    return np.stack(out)


def test_bin_level_ece_matches_accumulator_and_bootstraps():
    rng = np.random.default_rng(5)
    c, ok = rng.uniform(0.2, 1, 5000), rng.random(5000) < 0.6
    acc = CalibrationAccumulator(N_BINS)
    acc.update(c, ok)
    assert ece_from_bins(np.stack([acc.count, acc.conf_sum, acc.correct_sum], 1)) == pytest.approx(acc.ece())
    assert math.isnan(ece_from_bins(np.zeros((N_BINS, 3))))
    good, bad = bins_for(rng, 100, 0.0), bins_for(rng, 100, 0.25)
    ci = bootstrap_ece_ci(bad, n_boot=200)
    assert ci["ci95"][0] <= ci["point"] <= ci["ci95"][1] and ci["point"] > 0.15
    d = paired_bootstrap_ece(bad, good, n_boot=200)
    assert d["diff_point"] < -0.1 and d["ci95"][1] < 0 and d["p_b_lower"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap_ece(bad, good[:10])
    empty = paired_bootstrap_ece(np.zeros((5, N_BINS, 3)), np.zeros((5, N_BINS, 3)), n_boot=20)
    assert empty == {"diff_point": None, "ci95": None, "p_b_lower": None}
