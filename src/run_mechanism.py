"""Three mechanism probes that the accuracy tables cannot supply.

A. Does the fused Random Forest behave like the stimulus prior?
   We compare the two predictors to each other rather than to the label.
   If the forest is approximating a stimulus lookup, its predictions
   should coincide with the prior far more often than either coincides
   with the ground truth.

B. Is the deviation really unpredictable, or only unpredictable after
   discretisation? We regress the continuous deviation directly and
   report out-of-fold correlation.

C. Which features does the forest actually use? Grouped Gini importance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
N_WIN = 30

PHY = np.arange(12, 24)
BEH = np.r_[0:12, 24:49]
ALL = np.arange(49)

# 49 official features: 3 stats x 12 channels = 36, then 13 gaze-event stats.
CHANNEL_NAMES = [
    "head PC1", "eye PC1", "pupil mean", "pupil phasic",
    "EDA", "BVP", "HR", "SKT",
    "head pitch", "head yaw", "eye pitch", "eye yaw",
]
GROUPS = {
    "head / eye orientation": list(range(0, 2)) + list(range(8, 12)),
    "pupil": [2, 3],
    "physiology": [4, 5, 6, 7],
}


def feature_groups():
    """Map each of the 49 feature columns to a semantic group."""
    g = np.empty(49, dtype=object)
    for name, chans in GROUPS.items():
        for c in chans:
            for stat in range(3):          # mean, std, median blocks
                g[stat * 12 + c] = name
    for j in range(36, 49):
        g[j] = "fixation / saccade"
    return g


def majority(y, n_cls):
    return int(np.bincount(y, minlength=n_cls).argmax())


def prior_predict(y_tr, key_tr, key_te, n_cls):
    fallback = majority(y_tr, n_cls)
    table = {int(k): majority(y_tr[key_tr == k], n_cls) for k in np.unique(key_tr)}
    return np.array([table.get(int(k), fallback) for k in key_te], dtype=np.int64)


def main():
    d = np.load(ROOT / "data" / "ceap_official.npz")
    cont = np.load(ROOT / "data" / "ceap_va_cont.npz")
    pid, vid, feat = d["pid"], d["vid"], d["feat"]
    win = np.tile(np.arange(N_WIN), len(pid) // N_WIN)
    key = vid * 100 + win
    subs = sorted(set(pid.tolist()))
    res = {}

    # ---------- A. does the forest imitate the prior? ----------
    print("=== A. agreement between the fused RF and the stimulus prior ===")
    agree = {}
    for task, y in (("V", d["y_val"]), ("A", d["y_aro"]),
                    ("V3", d["y_val3"]), ("A3", d["y_aro3"]), ("C5", d["y5"])):
        n_cls = int(y.max()) + 1
        a_rf_prior, a_rf_y, a_prior_y, base = [], [], [], []
        for s in subs:
            te, tr = pid == s, pid != s
            clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
            clf.fit(feat[tr], y[tr])
            p_rf = clf.predict(feat[te])
            p_pr = prior_predict(y[tr], key[tr], key[te], n_cls)
            a_rf_prior.append((p_rf == p_pr).mean() * 100)
            a_rf_y.append((p_rf == y[te]).mean() * 100)
            a_prior_y.append((p_pr == y[te]).mean() * 100)
            # chance agreement of two independent predictors with these
            # marginal distributions
            pr_rf = np.bincount(p_rf, minlength=n_cls) / len(p_rf)
            pr_pr = np.bincount(p_pr, minlength=n_cls) / len(p_pr)
            base.append(float((pr_rf * pr_pr).sum()) * 100)
        agree[task] = {
            "rf_vs_prior": float(np.mean(a_rf_prior)),
            "rf_vs_truth": float(np.mean(a_rf_y)),
            "prior_vs_truth": float(np.mean(a_prior_y)),
            "chance_agreement": float(np.mean(base)),
        }
        a = agree[task]
        print(f"  {task:3s} RF~prior={a['rf_vs_prior']:5.2f}  "
              f"RF~truth={a['rf_vs_truth']:5.2f}  "
              f"prior~truth={a['prior_vs_truth']:5.2f}  "
              f"chance={a['chance_agreement']:5.2f}")
    res["rf_prior_agreement"] = agree

    # ---------- B. continuous deviation regression ----------
    print("\n=== B. regressing the continuous deviation ===")
    reg = {}
    for name, yc in (("valence", cont["y_val_c"]), ("arousal", cont["y_aro_c"])):
        yc = yc.astype(np.float64)
        for mod_name, cols in (("all", ALL), ("phys", PHY), ("beh", BEH)):
            true, pred_rf, pred_ridge = [], [], []
            for s in subs:
                te, tr = pid == s, pid != s
                # consensus from training participants only
                cmap = {int(k): yc[tr][key[tr] == k].mean() for k in np.unique(key[tr])}
                gmean = yc[tr].mean()
                c_tr = np.array([cmap.get(int(k), gmean) for k in key[tr]])
                c_te = np.array([cmap.get(int(k), gmean) for k in key[te]])
                d_tr, d_te = yc[tr] - c_tr, yc[te] - c_te
                r = RandomForestRegressor(max_depth=6, random_state=SEED, n_jobs=-1)
                r.fit(feat[tr][:, cols], d_tr)
                pred_rf.append(r.predict(feat[te][:, cols]))
                ri = Ridge(alpha=10.0)
                mu, sd = feat[tr][:, cols].mean(0), feat[tr][:, cols].std(0) + 1e-8
                ri.fit((feat[tr][:, cols] - mu) / sd, d_tr)
                pred_ridge.append(ri.predict((feat[te][:, cols] - mu) / sd))
                true.append(d_te)
            true = np.concatenate(true)
            for lname, pr in (("RF", np.concatenate(pred_rf)),
                              ("Ridge", np.concatenate(pred_ridge))):
                r_p = float(stats.pearsonr(true, pr)[0])
                r2 = float(1 - ((true - pr) ** 2).sum() / ((true - true.mean()) ** 2).sum())
                reg[f"{name}/{mod_name}/{lname}"] = {"pearson_r": r_p, "r2": r2}
                print(f"  {name:8s} {mod_name:5s} {lname:5s} r={r_p:+.4f}  R2={r2:+.4f}")
    res["deviation_regression"] = reg

    # ---------- C. grouped feature importance ----------
    print("\n=== C. grouped Gini importance of the fused RF ===")
    groups = feature_groups()
    imp = {}
    for task, y in (("V", d["y_val"]), ("A", d["y_aro"])):
        acc = {}
        for s in subs:
            tr = pid != s
            clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
            clf.fit(feat[tr], y[tr])
            for j, w in enumerate(clf.feature_importances_):
                acc[groups[j]] = acc.get(groups[j], 0.0) + float(w)
        tot = sum(acc.values())
        imp[task] = {k: 100 * v / tot for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
        print(f"  {task}: " + "  ".join(f"{k} {v:.1f}%" for k, v in imp[task].items()))
    res["grouped_importance"] = imp

    OUT.mkdir(exist_ok=True)
    (OUT / "mechanism.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nsaved results/mechanism.json")


if __name__ == "__main__":
    main()
