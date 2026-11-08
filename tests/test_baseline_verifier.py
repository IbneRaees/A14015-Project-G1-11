import pytest

from src.agents.baseline_verifier import InstanceConfidenceVerifier
from src.state import Disposition


def test_area_weighted_uncertainty_ignores_irrelevant_classes():
    v = InstanceConfidenceVerifier()
    # building 0.9 conf over 3x the area of a road at 0.5 -> weighted conf 0.8; the tree is ignored
    u = v.tile_uncertainty([1, 4, 6], [0.9, 0.5, 0.1], [300, 100, 5000])
    assert u == pytest.approx(0.2)
    assert v.tile_uncertainty([6, 7], [0.9, 0.1], [10, 10]) == 0.0
    assert v.tile_uncertainty([], [], []) == 0.0


def test_threshold_required_then_decides():
    v = InstanceConfidenceVerifier()
    with pytest.raises(RuntimeError):
        v.assess([1], [0.9], [10], 100)
    v.fit([0.1, 0.2, 0.3, 0.4], 0.75)
    assert v.threshold == pytest.approx(0.3)
    rep, ver = v.assess([1], [0.5], [10], 100)
    assert ver.disposition == Disposition.HUMAN_REVIEW and rep.measure == "instance_conf"
    assert rep.relevant_mass == pytest.approx(0.1)
    assert v.assess([1], [0.95], [10], 100)[1].disposition == Disposition.AUTO_REPORT
