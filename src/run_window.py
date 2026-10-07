"""Window-length sensitivity and statistical power.

The protocol fixes a two-second window. Several autonomic responses,
electrodermal activity in particular, evolve more slowly than that, so a
reviewer may reasonably ask whether our null on the deviation task is an
artefact of temporal resolution. We therefore rebuild the analysis at
2, 4, 6, 10 and 30 second windows by concatenating consecutive base
windows within a clip, and re-run three things at each length:

  * the standard affect task (binary valence / arousal),
  * the sensor-free stimulus prior,
  * the deviation task.

We also report the minimum effect this design could have detected, so the
null can be read as a bound rather than as an absence of evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
N_WIN = 30
PHYS_CH = [4, 5, 6, 7]          # EDA, BVP, HR, SKT
BEH_CH = [0, 1, 2, 3, 8, 9, 10, 11]


def block_features(X: np.ndarray) -> np.ndarray:
    """mean / std / median per channel over the whole block."""
    return np.concatenate([X.mean(1), X.std(1), np.median(X, axis=1)], axis=1)


def regroup(X, pid, vid, y_val, y_aro, k: int):
    """Merge k consecutive base windows within each (participant, clip)."""
    n_blocks = N_WIN // k
    keep = n_blocks * k
    Xs, ps, vs, ts, yv, ya = [], [], [], [], [], []
    for s in np.unique(pid):
        for v in np.unique(vid):
            m = (pid == s) & (vid == v)
            xs = X[m][:keep]                      # (keep, 100, 12)
            vv, aa = y_val[m][:keep], y_aro[m][:keep]
            xs = xs.reshape(n_blocks, k * xs.shape[1], xs.shape[2])
            Xs.append(xs)
            ps.append(np.full(n_blocks, s))
            vs.append(np.full(n_blocks, v))
            ts.append(np.arange(n_blocks))
            yv.append(vv.reshape(n_blocks, k).mean(1))
            ya.append(aa.reshape(n_blocks, k).mean(1))
    return (np.concatenate(Xs), np.concatenate(ps), np.concatenate(vs),
            np.concatenate(ts), np.concatenate(yv), np.concatenate(ya))


def majority(y, n_cls=2):
    return int(np.bincount(y, minlength=n_cls).argmax())


def loso(feat, y, pid, cols=None):
    accs = []
    for s in np.unique(pid):
        te, tr = pid == s, pid != s
        F = feat if cols is None else feat[:, cols]
        clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
        clf.fit(F[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(F[te])) * 100)
    return np.array(accs)


def prior_loso(y, pid, cell):
    accs = []
    for s in np.unique(pid):
        te, tr = pid == s, pid != s
        fb = majority(y[tr])
        tab = {int(k): majority(y[tr][cell[tr] == k]) for k in np.unique(cell[tr])}
        pred = np.array([tab.get(int(k), fb) for k in cell[te]])
        accs.append(accuracy_score(y[te], pred) * 100)
    return np.array(accs)


def majority_loso(y, pid):
    accs = []
    for s in np.unique(pid):
        te, tr = pid == s, pid != s
        accs.append(accuracy_score(y[te], np.full(te.sum(), majority(y[tr]))) * 100)
    return np.array(accs)


def main():
    d = np.load(ROOT / "data" / "ceap_official.npz")
    cont = np.load(ROOT / "data" / "ceap_va_cont.npz")
    X, pid, vid = d["X"], d["pid"], d["vid"]
    yv_c = cont["y_val_c"].astype(np.float64)
    ya_c = cont["y_aro_c"].astype(np.float64)

    res = {}
    for k, secs in ((1, 2), (2, 4), (3, 6), (5, 10), (15, 30)):
        Xb, pb, vb, tb, yvb, yab = regroup(X, pid, vid, yv_c, ya_c, k)
        feat = block_features(Xb.astype(np.float64))
        n_ch = Xb.shape[2]
        phys_cols = np.array([o * n_ch + c for o in range(3) for c in PHYS_CH])
        cell = vb * 100 + tb

        row = {"n_windows": int(len(yvb))}
        for name, yc in (("V", yvb), ("A", yab)):
            y_bin = (yc >= 5).astype(np.int64)

            # standard task
            a_all = loso(feat, y_bin, pb)
            a_phy = loso(feat, y_bin, pb, phys_cols)
            a_pri = prior_loso(y_bin, pb, cell)

            # deviation task
            dev = np.zeros(len(yc), dtype=np.int64)
            acc_dev, acc_dev_phy, acc_maj = [], [], []
            for s in np.unique(pb):
                te, tr = pb == s, pb != s
                cmap = {int(c): yc[tr][cell[tr] == c].mean() for c in np.unique(cell[tr])}
                g = yc[tr].mean()
                c_tr = np.array([cmap.get(int(c), g) for c in cell[tr]])
                c_te = np.array([cmap.get(int(c), g) for c in cell[te]])
                d_tr = (yc[tr] - c_tr > 0).astype(np.int64)
                d_te = (yc[te] - c_te > 0).astype(np.int64)
                clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
                clf.fit(feat[tr], d_tr)
                acc_dev.append(accuracy_score(d_te, clf.predict(feat[te])) * 100)
                clf2 = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
                clf2.fit(feat[tr][:, phys_cols], d_tr)
                acc_dev_phy.append(
                    accuracy_score(d_te, clf2.predict(feat[te][:, phys_cols])) * 100)
                acc_maj.append(
                    accuracy_score(d_te, np.full(len(d_te), majority(d_tr))) * 100)
            acc_dev = np.array(acc_dev)
            acc_dev_phy = np.array(acc_dev_phy)
            acc_maj = np.array(acc_maj)

            diff = acc_dev - acc_maj
            p_dev = float(stats.wilcoxon(diff).pvalue) if not np.allclose(diff, 0) else 1.0
            sd = float(diff.std(ddof=1))
            # two-sided paired t-test detectable effect at alpha=.05, power=.80
            mde = 2.8 * sd / np.sqrt(len(diff))

            row[name] = {
                "task_all": float(a_all.mean()),
                "task_phys": float(a_phy.mean()),
                "task_prior": float(a_pri.mean()),
                "dev_all": float(acc_dev.mean()),
                "dev_phys": float(acc_dev_phy.mean()),
                "dev_majority": float(acc_maj.mean()),
                "dev_delta": float(diff.mean()),
                "dev_p": p_dev,
                "mde_points": float(mde),
            }
            r = row[name]
            print(f"{secs:2d}s {name}  task: all={r['task_all']:.2f} "
                  f"phys={r['task_phys']:.2f} PRIOR={r['task_prior']:.2f}   "
                  f"dev: all={r['dev_all']:.2f} phys={r['dev_phys']:.2f} "
                  f"maj={r['dev_majority']:.2f} "
                  f"(d={r['dev_delta']:+.2f}, p={r['dev_p']:.3f}, "
                  f"MDE={r['mde_points']:.2f})", flush=True)
        res[f"{secs}s"] = row

    OUT.mkdir(exist_ok=True)
    (OUT / "window.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved results/window.json")


if __name__ == "__main__":
    main()
