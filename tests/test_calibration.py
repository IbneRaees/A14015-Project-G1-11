import numpy as np
import pytest

from src.components.calibration import (CalibrationAccumulator, RiskCoverageAccumulator, aurc,
                                        fit_threshold_for_coverage, risk_at_coverage,
                                        risk_coverage_curve)


def test_calibrated_vs_overconfident():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.1, 1.0, 300_000)
    good = CalibrationAccumulator()
    good.update(conf, rng.random(conf.size) < conf)
    assert good.ece() < 0.01
    bad = CalibrationAccumulator()
    bad.update(np.full(100_000, 0.99), rng.random(100_000) < 0.6)
    assert bad.ece() == pytest.approx(0.39, abs=0.01)


def test_streaming_equals_one_shot():
    rng = np.random.default_rng(1)
    c = rng.random(10_000)
    ok = rng.random(10_000) < 0.5
    one, two = CalibrationAccumulator(10), CalibrationAccumulator(10)
    one.update(c, ok)
    for i in range(0, 10_000, 2_500):
        two.update(c[i:i + 2_500], ok[i:i + 2_500])
    assert one.ece() == pytest.approx(two.ece())
    assert sum(two.reliability()["bin_count"]) == 10_000


def test_confidence_one_falls_in_last_bin_and_empty_raises():
    a = CalibrationAccumulator(5)
    a.update(np.array([1.0, 1.0]), np.array([1, 0]))
    assert a.count[-1] == 2 and a.ece() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        CalibrationAccumulator().ece()


def test_risk_coverage_informative_vs_random_uncertainty():
    rng = np.random.default_rng(2)
    err = (rng.random(2000) < 0.3).astype(float)
    informative = err + rng.normal(0, 0.3, 2000)          # high where the model is wrong
    random_u = rng.random(2000)
    cov, risk_i = risk_coverage_curve(informative, err)
    _, risk_r = risk_coverage_curve(random_u, err)
    assert risk_i[-1] == pytest.approx(err.mean()) == pytest.approx(risk_r[-1])   # full coverage = overall error
    assert aurc(cov, risk_i) < aurc(cov, risk_r)
    assert risk_at_coverage(cov, risk_i, 0.5) < risk_at_coverage(cov, risk_r, 0.5)


def test_accumulator_curve_matches_exact_curve():
    rng = np.random.default_rng(3)
    u = rng.random(20_000)
    err = rng.random(20_000) < u * 0.6
    acc = RiskCoverageAccumulator(n_bins=2000)
    acc.update(u[:10_000], err[:10_000])
    acc.update(u[10_000:], err[10_000:])
    cov_a, risk_a = acc.curve()
    cov_e, risk_e = risk_coverage_curve(u, err)
    assert aurc(cov_a, risk_a) == pytest.approx(aurc(cov_e, risk_e), abs=2e-3)
    assert risk_a[-1] == pytest.approx(err.mean())


def test_fit_threshold_for_coverage():
    u = np.arange(1, 101) / 100
    t = fit_threshold_for_coverage(u, 0.8)
    assert t == pytest.approx(0.8) and np.mean(u <= t) >= 0.8
    assert np.mean(u <= fit_threshold_for_coverage(u, 1.0)) == 1.0
    with pytest.raises(ValueError):
        fit_threshold_for_coverage(u, 0.0)
