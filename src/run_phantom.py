"""Phantom-sensor control: how much accuracy does the protocol produce
from a sensor that measures nothing about the viewer?

We keep the released CEAP-360VR pipeline untouched -- same 49-dimensional
feature vector shape, same depth-four random forest, same
leave-one-subject-out folds, same labels -- and replace the sensor
readings with a synthetic signal that contains no physiology, no gaze,
no head motion, and no individual information at all. The phantom
signal is a fixed random embedding of the (clip, window) cell plus
Gaussian noise:

    x_{s,v,t} = mu_{v,t} + sigma * eps,   eps ~ N(0, I)

sigma is a dial for how strongly the phantom sensor is locked to the
stimulus. At sigma = 0 the phantom identifies the clip perfectly; as
sigma grows it becomes pure noise. Sweeping sigma turns the confound
from a switch into a dose-response curve, and lets us ask the question
that matters: at the sigma where the phantom identifies the clip as well
as real headset kinematics do, how well does it recognise affect?

Outputs results/phantom.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 42
N_WIN = 30
SIGMAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def window_index(pid: np.ndarray, vid: np.ndarray) -> np.ndarray:
    """0..29 within each (subject, clip), preserving file order."""
    win = np.zeros(len(pid), dtype=np.int64)
    for p in np.unique(pid):
        for v in np.unique(vid[pid == p]):
            idx = np.flatnonzero((pid == p) & (vid == v))
            win[idx] = np.arange(len(idx))
    return win


def phantom_features(vid, win, n_feat: int, sigma: float, rng) -> np.ndarray:
    """Random embedding of the (clip, window) cell plus Gaussian noise.

    The embedding is drawn once and shared by every participant, so the
    phantom sensor carries stimulus identity and nothing else.
    """
    cells = sorted({(int(v), int(w)) for v, w in zip(vid, win)})
    table = {c: rng.standard_normal(n_feat) for c in cells}
    base = np.stack([table[(int(v), int(w))] for v, w in zip(vid, win)])
    if sigma == 0.0:
        return base
    return base + sigma * rng.standard_normal(base.shape)


def loso(X, y, pid, depth: int = 4) -> dict:
    """Leave-one-subject-out with the released depth-four forest."""
    accs, f1s = [], []
    for s in np.unique(pid):
        te = pid == s
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=depth, random_state=SEED, n_jobs=-1
        )
        clf.fit(X[~te], y[~te])
        p = clf.predict(X[te])
        accs.append(accuracy_score(y[te], p) * 100)
        f1s.append(f1_score(y[te], p, average="weighted", zero_division=0) * 100)
    return {
        "acc": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "per_fold": [float(a) for a in accs],
    }


def main():
    t0 = time.time()
    off = np.load(ROOT / "data" / "ceap_official.npz")
    feat, pid, vid = off["feat"], off["pid"], off["vid"]
    win = window_index(pid, vid)
    assert int(win.max()) == N_WIN - 1
    tasks = {"valence": off["y_val"], "arousal": off["y_aro"]}
    n_feat = feat.shape[1]
    print(f"feat {feat.shape}  clip-id chance {100/len(np.unique(vid)):.1f}%", flush=True)

    out = {
        "protocol": {
            "note": "released 49-d pipeline, depth-4 RF, LOSO, labels untouched",
            "phantom": "x = mu_(clip,window) + sigma * N(0,I); no viewer signal",
            "sigmas": list(SIGMAS),
            "n_features": int(n_feat),
            "seed": SEED,
        },
        "real": {},
        "phantom": {},
    }

    # Reference: the real released features, same forest, same folds.
    for name, y in tasks.items():
        out["real"][name] = loso(feat, y, pid)
        print(f"REAL {name:8s} acc {out['real'][name]['acc']:.2f}", flush=True)
    out["real"]["clip_id"] = loso(feat, vid, pid)
    print(f"REAL clip-id  acc {out['real']['clip_id']['acc']:.2f}", flush=True)

    # Phantom sweep.
    for sigma in SIGMAS:
        rng = np.random.default_rng(SEED)
        Xp = phantom_features(vid, win, n_feat, sigma, rng)
        bag = {}
        for name, y in tasks.items():
            bag[name] = loso(Xp, y, pid)
        bag["clip_id"] = loso(Xp, vid, pid)
        out["phantom"][f"sigma_{sigma}"] = bag
        print(
            f"PHANTOM sigma={sigma:<5} clip-id {bag['clip_id']['acc']:5.2f}  "
            f"valence {bag['valence']['acc']:5.2f}  arousal {bag['arousal']['acc']:5.2f}",
            flush=True,
        )

    out["elapsed_s"] = time.time() - t0
    OUT.mkdir(exist_ok=True)
    path = OUT / "phantom.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {path} in {out['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
