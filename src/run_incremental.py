"""Do the sensors carry information beyond the stimulus? A proper test.

Our earlier evidence was that a gradient-boosted model given the prior's
hard prediction plus the sensors scored below the prior alone. That is weak:
degradation can come from the fusion, from feeding a discrete label instead
of a distribution, or from a learner that cannot ignore noise features.

This script replaces it with a test of the actual hypothesis,

    Y  independent of  S  given  (V, T),

using three ingredients a reviewer can check:

1. Cross-fitted prior *probabilities*. Within each leave-one-subject-out
   fold we estimate p(y | v, t) from the training subjects with Laplace
   smoothing and pass log p as features, so the learner can reproduce the
   prior exactly by using those columns alone and adding sensors can only
   help if they carry extra signal.
2. Proper scoring rules. We report log-loss and the multi-class Brier score
   alongside accuracy, so "no gain" is not an artefact of thresholding.
3. A conditional permutation null. We shuffle the sensor rows *within* each
   (clip, window) cell across subjects, which destroys any subject-specific
   sensor-label link while preserving the stimulus-conditional sensor
   distribution, and re-run the entire pipeline. The observed
   prior-plus-sensors score is then compared against this null.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
N_WIN = 30
N_PERM = 200
ALPHA = 1.0
N_JOBS = -1

PHY = np.arange(12, 24)
BEH = np.r_[0:12, 24:49]
ALL = np.arange(49)
TASKS = ("y_val", "y_aro", "y_val3", "y_aro3", "y5")
NICE = {"y_val": "V-2", "y_aro": "A-2", "y_val3": "V-3", "y_aro3": "A-3", "y5": "5-cls"}


def prior_prob(y_tr, k_tr, k_te, n_cls):
    """Laplace-smoothed p(y | clip, window) from training subjects only."""
    g = (np.bincount(y_tr, minlength=n_cls) + ALPHA) / (len(y_tr) + ALPHA * n_cls)
    table = {}
    for k in np.unique(k_tr):
        c = np.bincount(y_tr[k_tr == k], minlength=n_cls)
        table[int(k)] = (c + ALPHA) / (c.sum() + ALPHA * n_cls)
    P_tr = np.stack([table[int(k)] for k in k_tr])
    P_te = np.stack([table.get(int(k), g) for k in k_te])
    return P_tr, P_te


def brier(y, P, n_cls):
    oh = np.zeros_like(P)
    oh[np.arange(len(y)), y] = 1.0
    return float(((P - oh) ** 2).sum(axis=1).mean())


def metrics(y, P, n_cls):
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return {
        "acc": float(accuracy_score(y, P.argmax(1)) * 100),
        "logloss": float(log_loss(y, P, labels=list(range(n_cls)))),
        "brier": brier(y, P, n_cls),
    }


def zscore(tr, te):
    mu = tr.mean(0, keepdims=True)
    sd = tr.std(0, keepdims=True) + 1e-8
    return (tr - mu) / sd, (te - mu) / sd


def run_fold(feat, cols, y, pid, vid, win, s, n_cls, learner, use_prior, use_sensor):
    te = pid == s
    tr = ~te
    k_tr = vid[tr] * 100 + win[tr]
    k_te = vid[te] * 100 + win[te]
    P_tr, P_te = prior_prob(y[tr], k_tr, k_te, n_cls)

    blocks_tr, blocks_te = [], []
    if use_prior:
        blocks_tr.append(np.log(np.clip(P_tr, 1e-9, 1)))
        blocks_te.append(np.log(np.clip(P_te, 1e-9, 1)))
    if use_sensor:
        a, b = zscore(feat[tr][:, cols], feat[te][:, cols])
        blocks_tr.append(a)
        blocks_te.append(b)
    if not blocks_tr:  # prior only, no learner
        return metrics(y[te], P_te, n_cls)

    Xtr = np.hstack(blocks_tr)
    Xte = np.hstack(blocks_te)
    if learner == "lr":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    else:
        clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=150)
    clf.fit(Xtr, y[tr])
    P = np.zeros((te.sum(), n_cls))
    P[:, clf.classes_] = clf.predict_proba(Xte)
    return metrics(y[te], P, n_cls)


def loso_metrics(feat, cols, y, pid, vid, win, n_cls, learner, use_prior, use_sensor, par=True):
    args = (feat, cols, y, pid, vid, win)
    tail = (n_cls, learner, use_prior, use_sensor)
    subs = np.unique(pid)
    if par:
        rows = Parallel(n_jobs=N_JOBS, batch_size=4)(
            delayed(run_fold)(*args, s, *tail) for s in subs
        )
    else:
        rows = [run_fold(*args, s, *tail) for s in subs]
    keys = rows[0].keys()
    return {k: [r[k] for r in rows] for k in keys}


def _perm_once(feat, y, pid, vid, win, n_cls, seed):
    """One conditional-permutation draw, evaluated by mean LOSO log-loss."""
    fp = permute_within_cell(feat, vid, win, np.random.default_rng(seed))
    m = loso_metrics(fp, ALL, y, pid, vid, win, n_cls, "lr", True, True, par=False)
    return float(np.mean(m["logloss"]))


def paired(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    if np.allclose(d, 0):
        return {"delta": 0.0, "p": 1.0}
    return {"delta": float(d.mean()), "p": float(stats.wilcoxon(d, zero_method="wilcox").pvalue)}


def permute_within_cell(feat, vid, win, rng):
    """Shuffle sensor rows within each (clip, window) cell across subjects."""
    out = feat.copy()
    key = vid * 100 + win
    for k in np.unique(key):
        idx = np.where(key == k)[0]
        out[idx] = feat[rng.permutation(idx)]
    return out


def main() -> None:
    off = np.load(ROOT / "data" / "ceap_official.npz")
    pid, vid = off["pid"], off["vid"]
    win = np.tile(np.arange(N_WIN), len(pid) // N_WIN)
    feat = off["feat"]

    res = {}
    for tk in TASKS:
        y = off[tk]
        n_cls = int(y.max()) + 1
        res[tk] = {"rows": {}, "tests": {}}

        base = loso_metrics(feat, ALL, y, pid, vid, win, n_cls, "lr", True, False)
        res[tk]["rows"]["prior only"] = {k: float(np.mean(v)) for k, v in base.items()}

        for tag, cols in (("phys", PHY), ("beh", BEH), ("all", ALL)):
            for learner in ("lr", "hgb"):
                m = loso_metrics(feat, cols, y, pid, vid, win, n_cls, learner, True, True)
                name = f"prior+{tag} ({learner})"
                res[tk]["rows"][name] = {k: float(np.mean(v)) for k, v in m.items()}
                for metric in ("logloss", "brier", "acc"):
                    res[tk]["tests"][f"{name} :: {metric}"] = paired(base[metric], m[metric])
            m = loso_metrics(feat, cols, y, pid, vid, win, n_cls, "lr", False, True)
            res[tk]["rows"][f"{tag} only (lr)"] = {k: float(np.mean(v)) for k, v in m.items()}

        # conditional permutation null for prior+all under the linear learner
        obs = float(np.mean(loso_metrics(feat, ALL, y, pid, vid, win, n_cls, "lr", True, True)["logloss"]))
        seeds = SEED * 1000 + np.arange(N_PERM)
        null = np.asarray(
            Parallel(n_jobs=N_JOBS, batch_size=1)(
                delayed(_perm_once)(feat, y, pid, vid, win, n_cls, int(s)) for s in seeds
            )
        )
        res[tk]["perm"] = {
            "observed_logloss": obs,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "null_q05": float(np.quantile(null, 0.05)),
            # one-sided: is the real sensor arrangement better (lower loss) than chance?
            "p_lower": float(((null <= obs).sum() + 1) / (len(null) + 1)),
            "n_perm": int(len(null)),
        }
        p = res[tk]["perm"]
        print(
            f"{NICE[tk]:6s} prior logloss={res[tk]['rows']['prior only']['logloss']:.4f}  "
            f"prior+all(lr)={res[tk]['rows']['prior+all (lr)']['logloss']:.4f}  "
            f"perm null={p['null_mean']:.4f}+-{p['null_sd']:.4f}  p={p['p_lower']:.3f}",
            flush=True,
        )

    OUT.mkdir(exist_ok=True)
    (OUT / "incremental.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved results/incremental.json")


if __name__ == "__main__":
    main()
