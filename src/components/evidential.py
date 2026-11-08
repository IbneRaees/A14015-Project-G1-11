"""Evidential (Dirichlet) segmentation head and uncertainty maths (Gap 2).

Per pixel the head outputs non-negative evidence e_k, giving a Dirichlet with
alpha_k = e_k + 1 (Sensoy et al., 2018). From alpha:

    S          = sum_k alpha_k
    p_k        = alpha_k / S                       expected class probability
    vacuity    = K / S                             lack of evidence, in (0, 1]
    entropy    = H[p] / ln K                       total uncertainty, [0, 1]
    epistemic  = (H[p] - E[H(Dir(alpha))]) / ln K  mutual information, [0, 1]

Arrays use channels-first layout: alpha has shape (K, H, W).

The numpy functions are the tested reference. The torch head and loss at the
bottom mirror them and are only defined when torch is installed.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.special import digamma, gammaln

MEASURES = ("vacuity", "entropy", "epistemic")


def dirichlet_stats(alpha: np.ndarray, axis: int = 0) -> Dict[str, np.ndarray]:
    """Per-pixel statistics of a Dirichlet field. All outputs have the class axis removed."""
    a = np.asarray(alpha, np.float32)
    if np.any(a <= 0):
        raise ValueError("alpha must be positive")
    k = a.shape[axis]
    s = a.sum(axis=axis, keepdims=True)
    p = a / s
    total_entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=axis)
    expected_entropy = (digamma(s + 1) - (p * digamma(a + 1)).sum(axis=axis, keepdims=True)).squeeze(axis)
    ln_k = math.log(k)
    return {
        "pred": p.argmax(axis=axis),
        "confidence": p.max(axis=axis),
        "vacuity": (k / s).squeeze(axis),
        "entropy": np.clip(total_entropy / ln_k, 0, 1),
        "epistemic": np.clip((total_entropy - expected_entropy) / ln_k, 0, 1),
        "relevant_p": p,           # caller sums the classes it cares about
    }


def uncertainty_map(stats: Dict[str, np.ndarray], measure: str) -> np.ndarray:
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    return stats[measure]


def kl_anneal(epoch: int, anneal_epochs: int) -> float:
    """Weight of the KL regulariser: ramps from 0 to 1 so early training is not dominated by it."""
    return float(min(1.0, epoch / max(anneal_epochs, 1)))


def edl_loss_reference(alpha: np.ndarray, target: np.ndarray, anneal: float = 1.0) -> float:
    """Numpy reference of the EDL loss, mean over pixels.

    alpha: (K, H, W); target: (H, W) integer class ids.
    loss = sum_k y_k (psi(S) - psi(alpha_k))  +  anneal * KL(Dir(alpha~) || Dir(1)),
    where alpha~ = y + (1 - y) * alpha keeps only the evidence for wrong classes.
    """
    k = alpha.shape[0]
    y = np.moveaxis(np.eye(k, dtype=np.float64)[target], -1, 0)
    a = alpha.astype(np.float64)
    s = a.sum(0)
    fit = (y * (digamma(s)[None] - digamma(a))).sum(0)
    at = y + (1 - y) * a
    st = at.sum(0)
    kl = (gammaln(st) - gammaln(k) - gammaln(at).sum(0)
          + ((at - 1) * (digamma(at) - digamma(st)[None])).sum(0))
    return float((fit + anneal * kl).mean())


# --------------------------------------------------------------------------
# torch parts (defined only when torch is available)
# --------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:  # the numpy functions above still work
    HAS_TORCH = False

if HAS_TORCH:

    class EvidentialSegHead(nn.Module):
        """Maps a backbone feature map to Dirichlet parameters alpha = softplus(logits) + 1."""

        def __init__(self, in_channels: int, num_classes: int = 10, hidden: int = 64):
            super().__init__()
            self.num_classes = num_classes
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, num_classes, 1),
            )

        def forward(self, feats: "torch.Tensor", out_size=None) -> "torch.Tensor":
            logits = self.net(feats)
            if out_size is not None:
                logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
            return F.softplus(logits) + 1.0      # alpha, shape (B, K, H, W)

    def edl_loss(alpha: "torch.Tensor", target: "torch.Tensor", anneal: float = 1.0,
                 ignore_index: int = -1) -> "torch.Tensor":
        """EDL loss for alpha (B, K, H, W) and integer target (B, H, W); same maths as the reference."""
        k = alpha.shape[1]
        valid = target != ignore_index
        t = torch.where(valid, target, torch.zeros_like(target))
        y = F.one_hot(t, k).permute(0, 3, 1, 2).to(alpha.dtype)
        s = alpha.sum(1, keepdim=True)
        fit = (y * (torch.digamma(s) - torch.digamma(alpha))).sum(1)
        at = y + (1 - y) * alpha
        st = at.sum(1, keepdim=True)
        kl = (torch.lgamma(st.squeeze(1)) - math.lgamma(k) - torch.lgamma(at).sum(1)
              + ((at - 1) * (torch.digamma(at) - torch.digamma(st))).sum(1))
        return (fit + anneal * kl)[valid].mean()
