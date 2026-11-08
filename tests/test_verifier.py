import cv2
import numpy as np
import pytest

from src.agents import GeospatialReasoningAgent
from src.agents.verifier_agent import VerifierAgent, VerifierConfig
from src.state import Disposition, TileInput, TileState

K = 10


def alpha_field(h=256, w=256, vacuous_from_x=None, road_class=4):
    """Confident road everywhere, except x >= vacuous_from_x where the model has no evidence."""
    a = np.ones((K, h, w), np.float32)
    a[road_class] = 50
    if vacuous_from_x is not None:
        a[:, :, vacuous_from_x:] = 1.0
    return a


def verifier(thr=0.3, **kw):
    return VerifierAgent(VerifierConfig(threshold=thr, **kw))


def test_threshold_must_be_fitted():
    with pytest.raises(RuntimeError, match="fit"):
        VerifierAgent().assess(alpha_field())


def test_confident_tile_is_auto_reported():
    rep, ver = verifier().assess(alpha_field())
    assert ver.disposition == Disposition.AUTO_REPORT and ver.reasons == []
    assert rep.tile_uncertainty == pytest.approx(10 / 59, abs=1e-3) and rep.n_cells_flagged == 0
    assert rep.mean_confidence > 0.8


def test_half_vacuous_tile_goes_to_human_and_regions_cover_the_vacuous_half():
    rep, ver = verifier().assess(alpha_field(vacuous_from_x=128))
    assert ver.disposition == Disposition.HUMAN_REVIEW and "tile uncertainty" in ver.reasons[0]
    assert rep.n_cells_flagged == 32
    assert all(x0 >= 128 for x0, _, _, _ in rep.flagged_regions) and len(rep.flagged_regions) == 32


def test_confident_grass_does_not_trigger_review_but_vague_roads_do():
    grass = alpha_field(road_class=9)
    rep, ver = verifier().assess(grass)
    assert rep.relevant_mass < 0.1 and ver.disposition == Disposition.AUTO_REPORT
    vague = np.ones((K, 64, 64), np.float32)
    assert verifier().assess(vague)[1].disposition == Disposition.HUMAN_REVIEW


def plus_road():
    m = np.zeros((256, 256), np.uint8)
    cv2.line(m, (20, 128), (236, 128), 1, 9)
    cv2.line(m, (128, 20), (128, 236), 1, 9)
    return GeospatialReasoningAgent().analyze(m)


def test_uncertain_road_edge_is_flagged_even_when_tile_score_is_low():
    road = plus_road()
    a = alpha_field(vacuous_from_x=150)          # only the right arm sits on vacuous pixels
    rep, ver = verifier(thr=0.6).assess(a, road_result=road)
    assert rep.tile_uncertainty < 0.6            # tile as a whole looks fine
    assert len(rep.uncertain_edges) == 1 and rep.uncertain_edges[0]["passable"] is True
    assert ver.disposition == Disposition.HUMAN_REVIEW and "road edge" in ver.reasons[0]
    clean, v2 = verifier(thr=0.6).assess(alpha_field(), road_result=road)
    assert clean.uncertain_edges == [] and v2.disposition == Disposition.AUTO_REPORT


def test_alpha_at_lower_resolution_than_the_tile():
    road = plus_road()
    small = np.ones((K, 128, 128), np.float32)
    small[4] = 50
    small[:, :, 75:] = 1.0                        # same scene at half resolution
    rep, _ = verifier(thr=0.6).assess(small, road_result=road, orig_shape=(256, 256))
    assert len(rep.uncertain_edges) == 1


def test_fit_sets_threshold_for_coverage():
    v = VerifierAgent()
    tiles = [alpha_field(), alpha_field(vacuous_from_x=200), alpha_field(vacuous_from_x=128),
             alpha_field(vacuous_from_x=64)]
    scores = [v.tile_uncertainty(t) for t in tiles]
    thr = v.fit(scores, 0.5)
    assert sum(s <= thr for s in scores) >= 2
    assert v.assess(tiles[0])[1].disposition == Disposition.AUTO_REPORT
    assert v.assess(tiles[3])[1].disposition == Disposition.HUMAN_REVIEW


def test_measures_are_selectable():
    a = alpha_field(vacuous_from_x=128)
    for m in ("vacuity", "entropy", "epistemic"):
        rep, _ = VerifierAgent(VerifierConfig(measure=m, threshold=0.2)).assess(a)
        assert rep.measure == m and 0.0 <= rep.tile_uncertainty <= 1.0
    with pytest.raises(ValueError):
        VerifierAgent(VerifierConfig(measure="nope"))


def test_run_writes_state_and_trajectory():
    st = TileState(TileInput("t9", 256, 256))
    st.arrays["alpha"] = alpha_field(vacuous_from_x=128)
    verifier().run(st, road_result=plus_road())
    assert st.verdict.disposition == Disposition.HUMAN_REVIEW and st.uncertainty.n_cells_flagged > 0
    assert st.trajectory[-1]["tool"] == "verifier.assess" and "human_review" in st.trajectory[-1]["summary"]
    assert '"human_review"' in st.to_json()
