"""OrthoVR: one residual. Affect is physiology orthogonal to kinematic load.

In a headset, EDA/BVP/HR/SKT mix affect with visual-vestibular load.
Rotational speed and acceleration of head and eyes are the observed
cause of that load. Affective looking is angular position and fixation,
which we keep. The method is ridge residualization of physiology on
motion energy, fit on training subjects only, then the official RF.

No extra backbone. No SSQ at inference. Identifiability is the
exclusion restriction: concurrent spin explains the nuisance, not the
valence-arousal residual.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import Ridge

PHYSIO = (0, 1, 2, 3)  # EDA, BVP, HR, SKT
HM_P, HM_Y, EM_P, EM_Y = 4, 5, 6, 7
FS = 50.0


def angle_distance(lon1, lat1, lon2, lat2):
    """Great-circle degrees; same formula as Xue et al. CEAP baseline."""
    rl1, rt1 = math.radians(lon1), math.radians(lat1)
    rl2, rt2 = math.radians(lon2), math.radians(lat2)
    dlon, dlat = rl2 - rl1, rt2 - rt1
    a = math.sin(dlat / 2) ** 2 + math.cos(rt1) * math.cos(rt2) * math.sin(dlon / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(max(0.0, a)))))


def angle_speed_np(pitch: np.ndarray, yaw: np.ndarray, fs: float = FS) -> np.ndarray:
    """pitch, yaw: (N, T) degrees -> speed (N, T) deg/s, copy first sample."""
    rlat1 = np.radians(pitch[:, :-1])
    rlat2 = np.radians(pitch[:, 1:])
    dlat = rlat2 - rlat1
    dlon = np.radians(yaw[:, 1:] - yaw[:, :-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    deg = np.degrees(2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0)))
    spd = np.empty(pitch.shape, dtype=np.float32)
    spd[:, 1:] = (deg * fs).astype(np.float32)
    spd[:, 0] = spd[:, 1]
    return spd


def motion_load(X: np.ndarray) -> np.ndarray:
    """(N, T, 4): head speed, eye speed, |head accel|, |eye accel|."""
    hs = angle_speed_np(X[:, :, HM_P], X[:, :, HM_Y])
    es = angle_speed_np(X[:, :, EM_P], X[:, :, EM_Y])
    ha = np.empty_like(hs)
    ea = np.empty_like(es)
    ha[:, 1:] = np.abs(np.diff(hs, axis=1)) * FS
    ea[:, 1:] = np.abs(np.diff(es, axis=1)) * FS
    ha[:, 0] = ha[:, 1]
    ea[:, 0] = ea[:, 1]
    return np.stack([hs, es, ha, ea], axis=-1)


def window_motion_summary(K: np.ndarray) -> np.ndarray:
    """(N, 8) mean/std of 4 motion channels — the window-level confound."""
    return np.concatenate([K.mean(1), K.std(1)], axis=1).astype(np.float32)


def _fixation_saccade(pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """13 official CEAP eye-movement stats on one window. pitch/yaw length T."""
    n = len(pitch)
    vel_pre = 0.0
    start = end = 0
    fix, sacc_dur, sacc_amp, sacc_vel = [], [], [], []
    sacc_start = 0
    for sample in range(1, n):
        ang = angle_distance(yaw[sample - 1], pitch[sample - 1], yaw[sample], pitch[sample])
        vel = ang * FS
        sacc_vel.append(vel)
        acc = (vel - vel_pre) * FS
        vel_pre = vel
        if abs(vel) > 75 and abs(acc) > 200:
            dur = end - start
            if dur / FS >= 0.15:
                fix.append(dur)
                if len(fix) == 1:
                    if start != 0:
                        sacc_dur.append(start)
                        sacc_amp.append(max(sacc_vel[0:start]) if start else 0.0)
                else:
                    sacc_dur.append(start - sacc_start)
                    sacc_amp.append(max(sacc_vel[sacc_start:start]) if start > sacc_start else 0.0)
                sacc_start = end
            start = sample + 1
            end = sample + 1
            continue
        end += 1
        if sample == n - 1:
            dur = end - start
            if dur / FS >= 0.15:
                fix.append(dur)
            else:
                sacc_dur.append(end - sacc_start)
                sacc_amp.append(max(sacc_vel[sacc_start:end]) if end > sacc_start else 0.0)

    def stats(xs, k):
        if not xs:
            return [0.0] * k
        a = np.asarray(xs, dtype=np.float64)
        if k == 5:
            return [float(len(a)), float(a.mean()), float(a.std()), float(a.max()), float(a.min())]
        return [float(a.mean()), float(a.std()), float(a.max()), float(a.min())]

    return np.array(stats(fix, 5) + stats(sacc_dur, 4) + stats(sacc_amp, 4), dtype=np.float32)


def eye_stats_batch(X: np.ndarray) -> np.ndarray:
    rows = [_fixation_saccade(X[i, :, EM_P], X[i, :, EM_Y]) for i in range(len(X))]
    return np.stack(rows)


def features_from_X(X: np.ndarray, eye: np.ndarray | None = None) -> np.ndarray:
    """Vectorized CEAP mean/std/median on signal, 1st/2nd diffs, head speed."""
    d1 = np.diff(X, axis=1, prepend=X[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    parts = []
    for arr in (X, d1, d2):
        parts.append(arr.mean(axis=1))
        parts.append(arr.std(axis=1))
        parts.append(np.median(arr, axis=1))
    dp = np.diff(X[:, :, HM_P], axis=1, prepend=X[:, :1, HM_P])
    dy = np.diff(X[:, :, HM_Y], axis=1, prepend=X[:, :1, HM_Y])
    speed = np.sqrt(dp ** 2 + dy ** 2)
    parts.append(np.stack([speed.mean(axis=1), speed.std(axis=1)], axis=1))
    base = np.concatenate(parts, axis=1).astype(np.float32)
    if eye is None:
        return base
    return np.concatenate([base, eye], axis=1)


class OrthoVR:
    """Fit B on train subjects; p = B k + affect. k is motion energy, not gaze pose."""

    def __init__(self, alpha: float = 1.0):
        self.ridge = Ridge(alpha=alpha, fit_intercept=True)
        self.k_mean = None
        self.k_std = None
        self.p_mean = None
        self.fitted = False

    def fit(self, X: np.ndarray, pid: np.ndarray | None = None, vid: np.ndarray | None = None):
        K = motion_load(X).astype(np.float64)
        P = X[:, :, list(PHYSIO)].astype(np.float64)
        if pid is not None and vid is not None:
            P = _center_by_video(P, pid, vid)
            K = _center_by_video(K, pid, vid)
        kf = K.reshape(-1, K.shape[-1])
        pf = P.reshape(-1, P.shape[-1])
        self.k_mean = kf.mean(0)
        self.k_std = kf.std(0) + 1e-8
        self.p_mean = pf.mean(0)
        self.ridge.fit((kf - self.k_mean) / self.k_std, pf - self.p_mean)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray, pid: np.ndarray | None = None, vid: np.ndarray | None = None) -> np.ndarray:
        assert self.fitted
        Y = X.copy()
        K = motion_load(X).astype(np.float64)
        P = X[:, :, list(PHYSIO)].astype(np.float64)
        if pid is not None and vid is not None:
            P = _center_by_video(P, pid, vid)
            K = _center_by_video(K, pid, vid)
        kf = K.reshape(-1, K.shape[-1])
        pred = self.ridge.predict((kf - self.k_mean) / self.k_std) + self.p_mean
        res = P.reshape(-1, P.shape[-1]) - pred
        res = res.reshape(P.shape)
        Y[:, :, list(PHYSIO)] = res.astype(np.float32)
        return Y

    def coef_norm(self) -> float:
        return float(np.linalg.norm(self.ridge.coef_))


def _center_by_video(P: np.ndarray, pid: np.ndarray, vid: np.ndarray) -> np.ndarray:
    """Remove per-video DC so B is not a tonic-EDA map. Offline 60 s clips."""
    out = P.copy()
    keys = pid.astype(np.int64) * 100 + vid.astype(np.int64)
    for k in np.unique(keys):
        m = keys == k
        out[m] = out[m] - out[m].mean(axis=(0, 1), keepdims=True)
    return out


def inject_spin(X: np.ndarray, gamma: float, channel=(0, 2)) -> np.ndarray:
    """Add gamma * z(head-speed(t)) into physio at sample rate so it lies in span(k)."""
    Y = X.copy()
    hs = angle_speed_np(X[:, :, HM_P], X[:, :, HM_Y])
    z = (hs - hs.mean()) / (hs.std() + 1e-8)
    for c in channel:
        sd = float(Y[:, :, c].std() + 1e-8)
        Y[:, :, c] = Y[:, :, c] + np.float32(gamma * sd) * z.astype(np.float32)
    return Y
