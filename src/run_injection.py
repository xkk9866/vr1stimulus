"""Sample-level spin injection: nuisance lies in span(k)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from orthovr import OrthoVR, features_from_X, inject_spin, motion_load

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "ceap_protocol.npz")
X, pid, vid, y = d["X"], d["pid"], d["vid"], d["y_aro"]
SEED = 42
idx_phy = []
for b in range(3):
    base = b * 24
    idx_phy += list(range(base, base + 4)) + list(range(base + 8, base + 12)) + list(range(base + 16, base + 20))
idx_phy = np.array(idx_phy)
speed = motion_load(X)[:, :, 0].mean(1)


def loso_acc_r(Xf):
    accs, rs = [], []
    for s in sorted(set(pid.tolist())):
        te = pid == s
        clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
        clf.fit(Xf[~te], y[~te])
        accs.append(accuracy_score(y[te], clf.predict(Xf[te])) * 100)
        p = clf.predict_proba(Xf[te])
        if p.shape[1] == 2 and p[:, 1].std() > 1e-8 and speed[te].std() > 1e-8:
            rs.append(float(np.corrcoef(p[:, 1], speed[te])[0, 1]))
    return float(np.mean(accs)), float(np.mean(rs) if rs else 0.0)


print("features...", flush=True)
phy = features_from_X(X)[:, idx_phy]
xinj = inject_spin(X, gamma=2.0)
phy_inj = features_from_X(xinj)[:, idx_phy]
raw = loso_acc_r(phy)
inj = loso_acc_r(phy_inj)
print("physio-only", raw, flush=True)
print("physio+inject", inj, flush=True)

accs, rs, bs, r_mean = [], [], [], []
for s in sorted(set(pid.tolist())):
    te = pid == s
    tr = ~te
    m = OrthoVR(alpha=1.0)
    m.fit(xinj[tr], pid[tr], vid[tr])
    Xte = m.transform(xinj[te], pid[te], vid[te])
    ftr = features_from_X(m.transform(xinj[tr], pid[tr], vid[tr]))[:, idx_phy]
    fte = features_from_X(Xte)[:, idx_phy]
    clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
    clf.fit(ftr, y[tr])
    accs.append(accuracy_score(y[te], clf.predict(fte)) * 100)
    p = clf.predict_proba(fte)
    r = 0.0
    if p.shape[1] == 2 and p[:, 1].std() > 1e-8 and speed[te].std() > 1e-8:
        r = float(np.corrcoef(p[:, 1], speed[te])[0, 1])
    rs.append(r)
    bs.append(m.coef_norm())
    eda_mean = Xte[:, :, 0].mean(1)
    r_mean.append(0.0 if eda_mean.std() < 1e-8 or speed[te].std() < 1e-8 else float(np.corrcoef(eda_mean, speed[te])[0, 1]))
    print(f"  subj {int(s)} acc={accs[-1]:.1f} r_pred={r:.3f} r_eda={r_mean[-1]:.3f} B={bs[-1]:.3f}", flush=True)

inj_eda = xinj[:, :, 0].mean(1)
raw_eda_r = float(np.corrcoef(inj_eda, speed)[0, 1])
out = {
    "gamma": 2.0,
    "physio": {"acc": raw[0], "r_speed": raw[1]},
    "injected_physio": {"acc": inj[0], "r_speed": inj[1], "r_eda_mean_speed": raw_eda_r},
    "orthovr_on_injected": {
        "acc": float(np.mean(accs)),
        "r_speed": float(np.mean(rs)),
        "r_eda_mean_speed": float(np.mean(r_mean)),
        "B_mean": float(np.mean(bs)),
    },
}
(ROOT / "results" / "injection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2), flush=True)
