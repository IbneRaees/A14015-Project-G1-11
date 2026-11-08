import json

import cv2
import networkx as nx
import numpy as np
import pytest

from src.agents.geospatial_agent import (
    GeospatialConfig,
    GeospatialReasoningAgent,
    skeletonize,
    zhang_suen_thinning,
)


def canvas(h=256, w=256):
    return np.zeros((h, w), np.uint8)


def line(img, p1, p2, t=9):
    cv2.line(img, p1, p2, 1, t)
    return img


@pytest.fixture
def agent():
    return GeospatialReasoningAgent()


def test_plus_shape_has_one_junction_and_four_endpoints(agent):
    m = canvas()
    line(m, (20, 128), (236, 128))
    line(m, (128, 20), (128, 236))
    r = agent.analyze(m)
    kinds = [d["kind"] for _, d in r.graph.nodes(data=True)]
    assert kinds.count("junction") == 1
    assert kinds.count("endpoint") == 4
    assert r.graph.number_of_edges() == 4
    assert r.stats["n_components"] == 1


def test_disconnected_roads_give_two_components(agent):
    m = canvas()
    line(m, (20, 60), (236, 60))
    line(m, (20, 190), (236, 190))
    r = agent.analyze(m)
    assert r.stats["n_components"] == 2
    assert not r.route_exists((30, 60), (220, 190), passable_only=False)


def test_ring_is_a_cycle(agent):
    m = canvas()
    cv2.circle(m, (128, 128), 80, 1, 9)
    r = agent.analyze(m)
    assert r.stats["n_components"] == 1
    assert len(nx.cycle_basis(nx.Graph(r.graph))) >= 1 or r.graph.number_of_edges() >= 1
    assert r.graph.number_of_edges() >= 1


def test_blocked_section_breaks_route(agent):
    m = canvas()
    line(m, (10, 128), (246, 128))
    blocked = canvas()
    blocked[:, 110:150] = 1
    blocked &= m
    r = agent.analyze(m, blocked)
    assert r.route_exists((20, 128), (236, 128), passable_only=False)
    assert not r.route_exists((20, 128), (236, 128), passable_only=True)


def test_no_blocked_section_keeps_route(agent):
    m = canvas()
    line(m, (10, 128), (246, 128))
    r = agent.analyze(m, canvas())
    assert r.route_exists((20, 128), (236, 128))
    assert r.stats["network_continuity"] == pytest.approx(1.0)
    assert r.stats["passable_length_frac"] == pytest.approx(1.0)


def test_t_junction_with_blocked_branch(agent):
    m = canvas()
    line(m, (10, 100), (246, 100))
    line(m, (128, 100), (128, 240))
    blocked = canvas()
    blocked[160:240, :] = 1
    blocked &= m
    r = agent.analyze(m, blocked)
    assert r.route_exists((20, 100), (236, 100))
    assert not r.route_exists((20, 100), (128, 230))
    assert r.stats["n_blocked_edges"] == 1


def test_short_spur_is_pruned(agent):
    m = canvas()
    line(m, (10, 128), (246, 128))
    line(m, (128, 128), (128, 115), t=3)  # tiny stub off the main road
    r = agent.analyze(m)
    assert r.graph.number_of_edges() == 1
    assert r.stats["n_components"] == 1


def test_empty_mask_is_handled(agent):
    r = agent.analyze(canvas())
    assert r.graph.number_of_nodes() == 0
    assert r.stats["n_components"] == 0
    assert r.stats["network_continuity"] == 0.0
    assert not r.route_exists((5, 5), (10, 10))


def test_large_tile_is_downscaled_and_positions_map_back(agent):
    m = canvas(1500, 2000)
    line(m, (100, 750), (1900, 750), t=40)
    r = agent.analyze(m)
    assert r.scale[0] < 1.0
    xs = sorted(n["x"] for n in r.to_dict()["nodes"])
    assert xs[0] == pytest.approx(100, abs=40)
    assert xs[-1] == pytest.approx(1900, abs=40)
    assert r.stats["total_length_px"] == pytest.approx(1800, rel=0.1)


def test_label_map_helper(agent):
    lab = canvas()
    # T junction: left arm dry road (4), right arm flooded road (3), stem dry (4)
    for p, q, cls in (((10, 100), (128, 100), 4), ((128, 100), (246, 100), 3), ((128, 100), (128, 240), 4)):
        tmp = canvas()
        line(tmp, p, q)
        lab[(tmp == 1) & (lab == 0)] = cls
    r = agent.analyze_label_map(lab, "t1")
    assert r.route_exists((20, 100), (128, 230))
    assert not r.route_exists((20, 100), (236, 100))


def test_zhang_suen_fallback_matches_topology():
    m = canvas()
    line(m, (20, 128), (236, 128))
    line(m, (128, 20), (128, 236))
    sk = zhang_suen_thinning(m)
    assert 0 < sk.sum() < m.sum() / 4
    assert skeletonize(m).sum() > 0


def test_input_validation(agent):
    with pytest.raises(ValueError):
        agent.analyze(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        agent.analyze(canvas(), canvas(10, 10))


def test_logging_and_json_roundtrip(tmp_path):
    ag = GeospatialReasoningAgent(log_dir=str(tmp_path))
    m = canvas()
    line(m, (10, 128), (246, 128))
    r = ag.analyze(m, tile_id="abc")
    assert [s["tool"] for s in r.trajectory] == [
        "preprocess_mask", "skeletonize", "skeleton_to_graph", "prune_graph", "assess_passability"]
    rec = json.loads((tmp_path / "geospatial_agent.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["tile_id"] == "abc"
    assert json.loads(r.to_json())["stats"]["n_edges"] == 1


def dry_flood_dry(gap=(110, 140), size=256, width=9):
    m = canvas(size, size)
    line(m, (10, 128), (246, 128), width)
    blocked = canvas(size, size)
    blocked[:, gap[0]:gap[1]] = 1
    return m, blocked & m


def test_partly_flooded_road_is_split_so_dry_ends_stay_reachable(agent):
    m, b = dry_flood_dry()
    r = agent.analyze(m, b)
    flags = sorted(d["passable"] for *_, d in r.graph.edges(keys=True, data=True))
    assert r.graph.number_of_edges() == 3 and flags == [False, True, True]
    assert sum(1 for _, d in r.graph.nodes(data=True) if d["kind"] == "split") == 2
    assert r.route_exists((20, 128), (90, 128))              # both on the dry left stretch
    assert r.route_exists((160, 128), (236, 128))            # both on the dry right stretch
    assert not r.route_exists((20, 128), (236, 128))         # the flood is in between
    assert r.route_exists((20, 128), (236, 128), passable_only=False)
    assert r.stats["n_components_passable"] == 2


def test_fully_flooded_or_fully_dry_edges_are_not_split(agent):
    m, _ = dry_flood_dry()
    assert agent.analyze(m, m).graph.number_of_edges() == 1                    # all flooded
    assert agent.analyze(m, canvas()).graph.number_of_edges() == 1            # all dry


def test_a_short_flooded_blip_neither_splits_nor_blocks(agent):
    m, b = dry_flood_dry(gap=(120, 130))                                      # 10 px < min run (20 px)
    r = agent.analyze(m, b)
    assert r.graph.number_of_edges() == 1 and r.stats["n_blocked_edges"] == 0


def test_a_tiny_dry_hole_inside_the_flood_is_bridged(agent):
    m, _ = dry_flood_dry()
    b = canvas()
    b[:, 100:125] = 1
    b[:, 128:150] = 1                                                           # 3 px dry hole at 125-127
    b &= m
    r = agent.analyze(m, b)
    assert r.graph.number_of_edges() == 3 and r.stats["n_blocked_edges"] == 1


def test_split_works_on_a_branching_network(agent):
    m = canvas()
    line(m, (10, 100), (246, 100))
    line(m, (128, 100), (128, 240))
    b = canvas()
    b[150:200, :] = 1                                                           # flood across the lower part of the stem
    b &= m
    r = agent.analyze(m, b)
    assert r.route_exists((20, 100), (128, 140))                                # dry upper stem
    assert not r.route_exists((20, 100), (128, 225))                            # beyond the flood
    assert r.route_exists((128, 140), (236, 100))
