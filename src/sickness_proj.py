"""Sickness-subspace projection: one linear removal, then the official RF.

Train-only: fit SSQ ≈ w·feat. Remove that 1-D subspace at train and test.
No SSQ needed at inference. Not a new network.
"""

from __future__ import annotations

import numpy as np


def fit_sickness_direction(feat: np.ndarray, ssq: np.ndarray) -> np.ndarray:
    x = feat.astype(np.float64)
    y = ssq.astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean()
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    n = np.linalg.norm(w)
    if n < 1e-8:
        return np.zeros_like(w, dtype=np.float32)
    return (w / n).astype(np.float32)


def project_out(feat: np.ndarray, w: np.ndarray) -> np.ndarray:
    if float(np.dot(w, w)) < 1e-12:
        return feat
    coeff = feat @ w
    return (feat - np.outer(coeff, w)).astype(np.float32)
