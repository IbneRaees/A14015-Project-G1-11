import numpy as np
import pytest

from src.components.calibration import CalibrationAccumulator, fit_temperature, softmax_np
from src.components.fallback_uncertainty import instance_uncertainty_map


def sq(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def test_instance_uncertainty_uses_best_covering_confidence():
    polys = [sq(0, 0, 20, 20), sq(10, 10, 30, 30)]
    m = instance_uncertainty_map(polys, [0.9, 0.4], (40, 40), uncovered_value=0.5)
    assert m.dtype == np.float32 and m.min() >= 0 and m.max() <= 1
    assert m[5, 5] == pytest.approx(0.1)               # only the 0.9 instance
    assert m[25, 25] == pytest.approx(0.6)             # only the 0.4 instance
    assert m[15, 15] == pytest.approx(0.1)             # overlap: more confident one wins
    assert m[35, 35] == pytest.approx(0.5)             # uncovered


def test_no_instances_gives_constant_prior_and_degenerate_polygons_are_ignored():
    assert np.all(instance_uncertainty_map([], [], (8, 8), 0.3) == np.float32(0.3))
    m = instance_uncertainty_map([np.array([[1, 1], [2, 2]])], [0.9], (8, 8), 0.3)
    assert np.all(m == np.float32(0.3))


def draw(rng, z, t_true):
    p = softmax_np(z, t_true)
    return np.array([rng.choice(len(r), p=r) for r in p])


def test_temperature_recovers_known_miscalibration():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, (30_000, 5))
    y = draw(rng, z, t_true=2.0)                       # labels come from softmax(z / 2); model says softmax(z)
    assert fit_temperature(z, y) == pytest.approx(2.0, rel=0.08)


def test_temperature_is_about_one_when_already_calibrated():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, (30_000, 5))
    y = draw(rng, z, 1.0)
    assert fit_temperature(z, y) == pytest.approx(1.0, rel=0.08)


def test_temperature_scaling_lowers_ece_of_an_overconfident_model():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 3, (40_000, 5))
    y = draw(rng, z, 2.5)

    def ece(t):
        p = softmax_np(z, t)
        acc = CalibrationAccumulator()
        acc.update(p.max(1), p.argmax(1) == y)
        return acc.ece()

    assert ece(fit_temperature(z, y)) < 0.5 * ece(1.0)


def test_fit_temperature_validates_shapes():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((4, 3)), np.zeros(5, int))
