import math

import numpy as np
import pytest
from scipy.special import digamma

from src.components import evidential as ev


def test_uniform_alpha_is_maximally_uncertain():
    s = ev.dirichlet_stats(np.ones((10, 3, 3), np.float32))
    assert s["vacuity"] == pytest.approx(np.ones((3, 3)))
    assert s["entropy"] == pytest.approx(np.ones((3, 3)), abs=1e-5)
    assert s["confidence"] == pytest.approx(np.full((3, 3), 0.1))


def test_strong_evidence_is_confident():
    a = np.ones((10, 2, 2), np.float32)
    a[4] = 500
    s = ev.dirichlet_stats(a)
    assert (s["pred"] == 4).all() and s["confidence"].min() > 0.98
    assert s["vacuity"].max() < 0.03 and s["epistemic"].max() < 0.01


def test_epistemic_matches_monte_carlo():
    alpha = np.array([2.0, 1.0, 3.0])
    rng = np.random.default_rng(0)
    p = rng.dirichlet(alpha, 400_000)
    mc_expected_entropy = -(p * np.log(p)).sum(1).mean()
    formula = digamma(alpha.sum() + 1) - ((alpha / alpha.sum()) * digamma(alpha + 1)).sum()
    assert formula == pytest.approx(mc_expected_entropy, abs=5e-3)
    s = ev.dirichlet_stats(alpha.reshape(3, 1, 1))
    pbar = alpha / alpha.sum()
    mi = (-(pbar * np.log(pbar)).sum() - formula) / math.log(3)
    assert s["epistemic"][0, 0] == pytest.approx(mi, abs=1e-5)


def test_invalid_alpha_and_measure():
    with pytest.raises(ValueError):
        ev.dirichlet_stats(np.zeros((3, 2, 2)))
    with pytest.raises(ValueError):
        ev.uncertainty_map({}, "bogus")


def test_kl_anneal():
    assert ev.kl_anneal(0, 10) == 0.0 and ev.kl_anneal(5, 10) == 0.5 and ev.kl_anneal(50, 10) == 1.0


def one_pixel(alpha, cls):
    return np.array(alpha, np.float64).reshape(-1, 1, 1), np.array([[cls]])


def test_edl_reference_hand_values():
    a, t = one_pixel([1, 1], 0)
    assert ev.edl_loss_reference(a, t) == pytest.approx(1.0)              # psi(2) - psi(1)
    a, t = one_pixel([5, 1], 0)
    assert ev.edl_loss_reference(a, t) == pytest.approx(0.2)              # psi(6) - psi(5) = 1/5, KL = 0


def test_edl_reference_penalises_misleading_evidence():
    a, t = one_pixel([1, 5], 0)                                            # evidence for the wrong class
    no_kl = ev.edl_loss_reference(a, t, anneal=0.0)
    full = ev.edl_loss_reference(a, t, anneal=1.0)
    assert no_kl == pytest.approx(sum(1 / k for k in range(1, 6)))         # H_5
    assert full > no_kl


# ---- torch mirror (runs on your machine; skipped here if torch is missing) ----
def test_torch_loss_matches_numpy_reference():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    alpha = rng.random((10, 6, 7)).astype(np.float32) * 5 + 1
    target = rng.integers(0, 10, (6, 7))
    ref = ev.edl_loss_reference(alpha, target, anneal=0.4)
    out = ev.edl_loss(torch.tensor(alpha)[None], torch.tensor(target)[None], anneal=0.4)
    assert out.item() == pytest.approx(ref, rel=1e-4)


def test_torch_ignore_index_and_head_shapes():
    torch = pytest.importorskip("torch")
    head = ev.EvidentialSegHead(in_channels=16, num_classes=10)
    alpha = head(torch.randn(2, 16, 8, 8), out_size=(32, 32))
    assert alpha.shape == (2, 10, 32, 32) and float(alpha.detach().min()) >= 1.0
    tgt = torch.randint(0, 10, (2, 32, 32))
    tgt[:, :16] = -1
    loss = ev.edl_loss(alpha, tgt)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in head.parameters())
