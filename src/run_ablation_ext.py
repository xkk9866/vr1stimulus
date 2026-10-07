"""Modality ablation under both the released and the paper-described pipelines.

Our ablation on the released 49-d features finds physiology dispensable,
which contradicts the corpus paper. A reviewer is right to ask whether the
conclusion is an artefact of the released extractor omitting the derivative
features the paper text describes. This script settles that by running the
identical protocol on both feature sets and, critically, by re-testing the
sensor-free stimulus prior against the *strongest* model either set yields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
N_WIN = 30

# ---- released 49-d blocks
PHY_49 = np.arange(12, 24)
BEH_49 = np.r_[0:12, 24:49]
ALL_49 = np.arange(49)

# ---- paper-described 121-d blocks
PHY_EXT = np.r_[12:24, 61:73, 97:109]
BEH_EXT = np.r_[0:12, 24:49, 49:61, 73:85, 85:97, 109:121]
ALL_EXT = np.arange(121)

TASKS = ("y_val", "y_aro", "y_val3", "y_aro3", "y5")
NICE = {"y_val": "V-2", "y_aro": "A-2", "y_val3": "V-3", "y_aro3": "A-3", "y5": "5-cls"}


def rank_biserial(delta: np.ndarray) -> float:
    nz = delta[delta != 0]
    if nz.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(nz))
    pos = r[nz > 0].sum()
    neg = r[nz < 0].sum()
    return float((pos - neg) / r.sum())


def paired(a: list[float], b: list[float]) -> dict:
    """Wilcoxon plus an exact-style sign-flip permutation test on fold deltas."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if np.allclose(d, 0):
        return {"delta": 0.0, "p": 1.0, "p_perm": 1.0, "rb": 0.0, "wins": 0, "n": len(d)}
    p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    rng = np.random.default_rng(SEED)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(20000, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return {
        "delta": float(d.mean()),
        "p": p,
        "p_perm": float(((null >= obs).sum() + 1) / (len(null) + 1)),
        "rb": rank_biserial(d),
        "wins": int((d > 0).sum()),
        "n": int(len(d)),
    }


def loso(X: np.ndarray, cols: np.ndarray, y: np.ndarray, pid: np.ndarray) -> tuple[float, float, list[float]]:
    acc, f1, per = [], [], []
    for s in np.unique(pid):
        te = pid == s
        tr = ~te
        clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
        clf.fit(X[tr][:, cols], y[tr])
        pred = clf.predict(X[te][:, cols])
        a = accuracy_score(y[te], pred) * 100
        acc.append(a)
        per.append(float(a))
        f1.append(f1_score(y[te], pred, average="weighted", zero_division=0))
    return float(np.mean(acc)), float(np.mean(f1)), per


def prior_loso(y: np.ndarray, pid: np.ndarray, vid: np.ndarray, win: np.ndarray) -> tuple[float, float, list[float]]:
    """Sensor-free majority vote within each (clip, window) cell."""
    n_cls = int(y.max()) + 1
    acc, f1, per = [], [], []
    for s in np.unique(pid):
        te = pid == s
        tr = ~te
        k_tr = vid[tr] * 100 + win[tr]
        k_te = vid[te] * 100 + win[te]
        gmaj = int(np.bincount(y[tr], minlength=n_cls).argmax())
        pmap = {}
        for k in np.unique(k_tr):
            pmap[int(k)] = int(np.bincount(y[tr][k_tr == k], minlength=n_cls).argmax())
        pred = np.array([pmap.get(int(k), gmaj) for k in k_te])
        a = accuracy_score(y[te], pred) * 100
        acc.append(a)
        per.append(float(a))
        f1.append(f1_score(y[te], pred, average="weighted", zero_division=0))
    return float(np.mean(acc)), float(np.mean(f1)), per


def main() -> None:
    off = np.load(ROOT / "data" / "ceap_official.npz")
    ext = np.load(ROOT / "data" / "ceap_deriv.npz")
    pid, vid = off["pid"], off["vid"]
    win = np.tile(np.arange(N_WIN), len(pid) // N_WIN)
    X49, X121 = off["feat"], ext["feat"]

    variants = {
        "released/all": (X49, ALL_49),
        "released/phys": (X49, PHY_49),
        "released/beh": (X49, BEH_49),
        "paper/all": (X121, ALL_EXT),
        "paper/phys": (X121, PHY_EXT),
        "paper/beh": (X121, BEH_EXT),
    }

    res: dict[str, dict] = {}
    for tk in TASKS:
        y = off[tk]
        res[tk] = {"rows": {}, "tests": {}}
        per_fold: dict[str, list[float]] = {}
        for name, (X, cols) in variants.items():
            a, f, per = loso(X, cols, y, pid)
            res[tk]["rows"][name] = {"acc": a, "wf1": f, "n_feat": int(len(cols))}
            per_fold[name] = per
            print(f"{NICE[tk]:6s} {name:16s} n={len(cols):3d} acc={a:6.2f} wf1={f:.4f}", flush=True)

        pa, pf, pper = prior_loso(y, pid, vid, win)
        res[tk]["rows"]["prior(v,t)"] = {"acc": pa, "wf1": pf, "n_feat": 0}
        per_fold["prior(v,t)"] = pper
        print(f"{NICE[tk]:6s} {'prior(v,t)':16s} n=  0 acc={pa:6.2f} wf1={pf:.4f}", flush=True)

        # does adding derivatives change the modality ordering?
        res[tk]["tests"]["released: phys vs all"] = paired(per_fold["released/all"], per_fold["released/phys"])
        res[tk]["tests"]["released: beh vs all"] = paired(per_fold["released/all"], per_fold["released/beh"])
        res[tk]["tests"]["paper: phys vs all"] = paired(per_fold["paper/all"], per_fold["paper/phys"])
        res[tk]["tests"]["paper: beh vs all"] = paired(per_fold["paper/all"], per_fold["paper/beh"])
        res[tk]["tests"]["paper/all vs released/all"] = paired(per_fold["released/all"], per_fold["paper/all"])
        res[tk]["tests"]["paper/phys vs released/phys"] = paired(per_fold["released/phys"], per_fold["paper/phys"])

        # the headline claim, against the best sensor model from either pipeline
        best = max(
            ("released/all", "paper/all", "released/beh", "paper/beh", "released/phys", "paper/phys"),
            key=lambda k: res[tk]["rows"][k]["acc"],
        )
        res[tk]["best_sensor"] = best
        res[tk]["tests"][f"prior vs BEST({best})"] = paired(per_fold[best], per_fold["prior(v,t)"])
        t = res[tk]["tests"][f"prior vs BEST({best})"]
        print(
            f"{NICE[tk]:6s} prior vs best={best:14s} "
            f"delta={t['delta']:+6.2f} p={t['p']:.2e} p_perm={t['p_perm']:.2e} "
            f"rb={t['rb']:.3f} wins={t['wins']}/{t['n']}",
            flush=True,
        )
        res[tk]["per_fold"] = per_fold

    OUT.mkdir(exist_ok=True)
    (OUT / "ablation_ext.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nsaved results/ablation_ext.json")


if __name__ == "__main__":
    main()
