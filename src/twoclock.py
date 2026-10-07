"""TwoClock: one residual. Slow clock = cybersickness; fast clock = affect.

VR physiology is the sum of a slow integrator (sickness / presence drift)
and a fast residual (affective transients). The method is a causal EMA
subtraction on the shared physiological channels. Head/eye remain as
behavior. No extra backbone is stacked.

Time constant τ ≈ 15 s follows the scale on which SSQ symptoms accumulate,
not a learned module.
"""

from __future__ import annotations

import numpy as np

PHYSIO = (0, 1, 2, 3)  # EDA, BVP, HR, SKT


def ema_alpha(tau_sec: float, fs: float) -> float:
    return float(np.exp(-1.0 / max(tau_sec * fs, 1e-6)))


def two_clock(x: np.ndarray, tau_sec: float = 15.0, fs: float = 50.0, physio_idx=PHYSIO) -> tuple[np.ndarray, np.ndarray]:
    """x: (T, C) one video. Returns (fast, slow) with same shape."""
    x = np.asarray(x, dtype=np.float32)
    alpha = ema_alpha(tau_sec, fs)
    slow = np.empty_like(x)
    state = x[0].copy()
    a = np.float32(alpha)
    b = np.float32(1.0 - alpha)
    for t in range(x.shape[0]):
        state = a * state + b * x[t]
        slow[t] = state
    fast = x.copy()
    idx = list(physio_idx)
    fast[:, idx] = x[:, idx] - slow[:, idx]
    return fast, slow


def window_features(win: np.ndarray) -> np.ndarray:
    """CEAP-style mean/std/median on signal, 1st and 2nd differences."""
    d1 = np.diff(win, axis=0, prepend=win[:1])
    d2 = np.diff(d1, axis=0, prepend=d1[:1])
    parts = []
    for arr in (win, d1, d2):
        parts.append(arr.mean(axis=0))
        parts.append(arr.std(axis=0))
        parts.append(np.median(arr, axis=0))
    speed = np.sqrt(np.diff(win[:, 4], prepend=win[0, 4]) ** 2 + np.diff(win[:, 5], prepend=win[0, 5]) ** 2)
    parts.append(np.array([speed.mean(), speed.std()], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)
