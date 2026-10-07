"""Participant-level uncertainty for the headline gap.

A reviewer correctly noted that leave-one-subject-out folds are not
independent replicates: each fold trains on 31 of the same 32 participants,
so a fold-level Wilcoxon test can understate uncertainty. We therefore report
the same comparison three ways and let the reader use the most conservative.

  cluster bootstrap  Resample participants with replacement and recompute the
                     mean paired difference. Uncertainty is attributed to the
                     participant sample, which is the level at which the data
                     are exchangeable and at which we wish to generalise.
  sign-flip test     Randomly flip the sign of each participant's paired
                     difference. This is the permutation test implied by the
                     null that the two systems are equivalent for every
                     participant, and it needs no distributional assumption.
  Wilcoxon           Reported for continuity with the earlier version.

We also report the smallest gap the design could detect, so that the null
results elsewhere in the paper are bounded rather than merely asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
N_WIN = 30
N_BOOT = 20000
N_FLIP = 20000

PHY = np.arange(12, 24)
BEH = np.r_[0:12, 24:49]
ALL = np.arange(49)
TASKS = ("y_val", "y_aro", "y_val3", "y_aro3", "y5")
NICE = {"y_val": "V-2", "y_aro": "A-2", "y_val3": "V-3", "y_aro3": "A-3", "y5": "5-cls"}


def rf_loso(feat, cols, y, pid) -> list[float]:
    out = []
    for s in np.unique(pid):
        te = pid == s
        clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
        clf.fit(feat[~te][:, cols], y[~te])
        out.append(float((clf.predict(feat[te][:, cols]) == y[te]).mean() * 100))
    return out


def prior_loso(y, pid, vid, win) -> list[float]:
    out = []
    n_cls = int(y.max()) + 1
    key = vid * 100 + win
    for s in np.unique(pid):
        te = pid == s
        tr = ~te
        table = {}
        for k in np.unique(key[tr]):
            table[int(k)] = int(np.bincount(y[tr][key[tr] == k], minlength=n_cls).argmax())
        g = int(np.bincount(y[tr], minlength=n_cls).argmax())
        pred = np.asarray([table.get(int(k), g) for k in key[te]])
        out.append(float((pred == y[te]).mean() * 100))
    return out


def cluster_bootstrap(d: np.ndarray, rng) -> dict:
    """Resample participants with replacement; d is one difference per participant."""
    n = len(d)
    draws = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    return {
        "mean": float(d.mean()),
        "se": float(draws.std(ddof=1)),
        "ci_lo": float(np.quantile(draws, 0.025)),
        "ci_hi": float(np.quantile(draws, 0.975)),
        # two-sided bootstrap p by inverting the interval around zero
        "p": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
    }


def sign_flip(d: np.ndarray, rng) -> dict:
    obs = abs(d.mean())
    signs = rng.choice((-1.0, 1.0), size=(N_FLIP, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return {
        "p": float(((null >= obs).sum() + 1) / (N_FLIP + 1)),
        "null_sd": float(null.std()),
    }


def detectable(d: np.ndarray) -> float:
    """Paired difference detectable at 80% power, two-sided 5%, given this spread."""
    return float(2.8 * d.std(ddof=1) / np.sqrt(len(d)))


def main() -> None:
    off = np.load(ROOT / "data" / "ceap_official.npz")
    pid, vid, feat = off["pid"], off["vid"], off["feat"]
    win = np.tile(np.arange(N_WIN), len(pid) // N_WIN)
    rng = np.random.default_rng(SEED)

    res: dict = {"n_participants": int(len(np.unique(pid))), "n_boot": N_BOOT}
    for tk in TASKS:
        y = off[tk]
        pr = np.asarray(prior_loso(y, pid, vid, win))
        best_name, best_vals = None, None
        rows = {}
        for tag, cols in (("phys", PHY), ("beh", BEH), ("all", ALL)):
            v = np.asarray(rf_loso(feat, cols, y, pid))
            rows[tag] = float(v.mean())
            if best_vals is None or v.mean() > best_vals.mean():
                best_name, best_vals = tag, v

        d = pr - best_vals  # gain of the sensor-free prior, one value per participant
        entry = {
            "prior_acc": float(pr.mean()),
            "sensor_acc": {k: v for k, v in rows.items()},
            "best_sensor": best_name,
            "gap": float(d.mean()),
            "cluster_bootstrap": cluster_bootstrap(d, rng),
            "sign_flip": sign_flip(d, rng),
            "wilcoxon_p": float(stats.wilcoxon(d, zero_method="wilcox").pvalue),
            "min_detectable_gap": detectable(d),
            "participants_favouring_prior": int((d > 0).sum()),
        }
        res[tk] = entry
        cb = entry["cluster_bootstrap"]
        print(
            f"{NICE[tk]:6s} prior={entry['prior_acc']:6.2f} best({best_name})="
            f"{best_vals.mean():6.2f} gap={entry['gap']:+6.2f} "
            f"[{cb['ci_lo']:+.2f},{cb['ci_hi']:+.2f}] "
            f"boot p={cb['p']:.4f} flip p={entry['sign_flip']['p']:.4f} "
            f"wilcoxon p={entry['wilcoxon_p']:.2e} "
            f"{entry['participants_favouring_prior']}/{len(d)}",
            flush=True,
        )

    OUT.mkdir(exist_ok=True)
    (OUT / "robust.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved results/robust.json")


if __name__ == "__main__":
    main()
