import numpy as np
import pytest

from src.components.router import (DynamicRouter, RouterConfig, achieved_rate, expected_latency,
                                   fit_threshold, max_escalation_rate, normalized_entropy)
from src.state import Route


def test_normalized_entropy_bounds():
    k = 10
    uniform = np.full((k, 3, 3), 1 / k)
    onehot = np.zeros((k, 3, 3))
    onehot[2] = 1
    assert normalized_entropy(uniform) == pytest.approx(np.ones((3, 3)), abs=1e-5)
    assert normalized_entropy(onehot).max() < 1e-4


def test_fit_threshold_respects_budget_including_ties():
    rng = np.random.default_rng(0)
    scores = rng.random(450)
    for rate in (0.0, 0.05, 0.2, 0.5, 1.0):
        t = fit_threshold(scores, rate)
        assert achieved_rate(scores, t) <= rate + 1e-12
    tied = [0.3] * 10 + [0.9] * 10
    assert achieved_rate(tied, fit_threshold(tied, 0.5)) <= 0.5
    assert achieved_rate(tied, fit_threshold(tied, 0.3)) <= 0.3


def test_latency_budget_maths():
    assert expected_latency(0.3, 3.2, 0.25) == pytest.approx(1.1)
    assert max_escalation_rate(0.3, 3.2, 2.0) == pytest.approx(0.53125)
    assert max_escalation_rate(2.5, 3.2, 2.0) == 0.0
    assert max_escalation_rate(0.1, 1.0, 5.0) == 1.0
    assert max_escalation_rate(0.3, 0.0, 2.0) == 1.0 and max_escalation_rate(0.3, -0.01, 2.0) == 1.0
    p = max_escalation_rate(0.4, 3.0, 2.0)
    assert expected_latency(0.4, 3.0, p) == pytest.approx(2.0)
    assert max_escalation_rate(0.3, 3.2, 2.0, region_cost_frac=0.25) > 0.53125


def hot_spot_map(h=240, w=320, hot=0.9):
    m = np.full((h, w), 0.05, np.float32)
    m[0:60, 240:320] = hot                      # one hot block, top-right
    return m


def test_threshold_must_be_fitted():
    with pytest.raises(RuntimeError, match="fit"):
        DynamicRouter().route(hot_spot_map())


def test_escalates_hot_tile_with_regions_in_original_coordinates():
    r = DynamicRouter(RouterConfig(grid=8, top_k=1, threshold=0.5))
    d = r.route(hot_spot_map(), orig_shape=(2400, 3200))
    assert d.route == Route.VLM and d.score == pytest.approx(0.9, abs=1e-3)
    assert d.regions and all(x0 >= 2400 and y1 <= 600 for x0, y0, x1, y1 in d.regions)


def test_flat_tile_stays_on_fast_path():
    d = DynamicRouter(RouterConfig(threshold=0.5)).route(np.full((100, 100), 0.1, np.float32))
    assert d.route == Route.FAST and d.regions == []


def test_top_k_dilutes_a_single_noisy_cell():
    m = hot_spot_map()
    single = DynamicRouter(RouterConfig(top_k=1, threshold=0.5)).score(m)[0]
    pooled = DynamicRouter(RouterConfig(top_k=8, threshold=0.5)).score(m)[0]
    assert pooled < single


def test_fit_then_route_hits_target_rate():
    rng = np.random.default_rng(1)
    r = DynamicRouter(RouterConfig(grid=4, top_k=2))
    maps = [rng.random((64, 64)).astype(np.float32) * rng.random() for _ in range(200)]
    scores = [r.score(m)[0] for m in maps]
    r.fit(scores, 0.10)
    rate = np.mean([r.route(m).route == Route.VLM for m in maps])
    assert rate <= 0.10


def test_forced_modes_for_ablation_need_no_map_or_threshold():
    assert DynamicRouter(RouterConfig(mode="always_vlm")).route().route == Route.VLM
    assert DynamicRouter(RouterConfig(mode="always_fast")).route().route == Route.FAST
    with pytest.raises(ValueError):
        DynamicRouter(RouterConfig(mode="nope"))
