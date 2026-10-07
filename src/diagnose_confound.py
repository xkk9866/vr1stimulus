"""Window-level confound: kinematics vs SSQ. CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "ceap_protocol.npz")
X, feat, pid = d["X"], d["feat"], d["pid"]
yv, ya, yv3, ya3 = d["y_val"], d["y_aro"], d["y_val3"], d["y_aro3"]
ssq = d["ssq"]
N = X.shape[0]


def ang(p1, y1, p2, y2):
    rlat1 = np.radians(p1)
    rlat2 = np.radians(p2)
    dlat = rlat2 - rlat1
    dlon = np.radians(y2 - y1)
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return np.degrees(2 * np.arcsin(np.clip(np.sqrt(a), 0, 1)))


hp, hy, ep, ey = X[:, :, 4], X[:, :, 5], X[:, :, 6], X[:, :, 7]
wh = ang(hp[:, :-1], hy[:, :-1], hp[:, 1:], hy[:, 1:]) * 50
we = ang(ep[:, :-1], ey[:, :-1], ep[:, 1:], ey[:, 1:]) * 50
spd_h = wh.mean(1)
spd_e = we.mean(1)
acc_h = np.abs(np.diff(wh, axis=1)).mean(1) * 50
phy_mean = X[:, :, :4].mean(1)

out = {"speed_h_mean": float(spd_h.mean()), "speed_h_std": float(spd_h.std())}
print("spd_h", spd_h.mean(), spd_h.std(), "spd_e", spd_e.mean(), spd_e.std())

out["r_physio_speed"] = {}
for i, n in enumerate(["EDA", "BVP", "HR", "SKT"]):
    r = float(np.corrcoef(phy_mean[:, i], spd_h)[0, 1])
    out["r_physio_speed"][n] = r
    print(f"  {n} r(mean, head_speed)={r:.4f}")

out["r_label_speed"] = {}
for n, y in [("val", yv), ("aro", ya), ("ssq", ssq)]:
    r = float(np.corrcoef(y.astype(float), spd_h)[0, 1])
    out["r_label_speed"][n] = r
    print(f"  r({n}, head_speed)={r:.4f}")
print("corr arousal vs ssq", float(np.corrcoef(ya.astype(float), ssq)[0, 1]))
print("corr spd vs ssq", float(np.corrcoef(spd_h, ssq)[0, 1]))

K = np.stack([spd_h, spd_e, acc_h], 1)
Kz = (K - K.mean(0)) / (K.std(0) + 1e-8)
out["R2_physio_motion"] = {}
for i, n in enumerate(["EDA", "BVP", "HR", "SKT"]):
    y = phy_mean[:, i]
    yz = (y - y.mean()) / (y.std() + 1e-8)
    B = np.linalg.lstsq(np.c_[np.ones(N), Kz], yz, rcond=None)[0]
    pred = np.c_[np.ones(N), Kz] @ B
    r2 = float(1 - ((yz - pred) ** 2).sum() / ((yz - yz.mean()) ** 2).sum())
    out["R2_physio_motion"][n] = r2
    print(f"R2 {n} ~ motion {r2:.4f}")

idx_phy, idx_kin = [], []
for b in range(3):
    base = b * 24
    idx_phy += list(range(base, base + 4)) + list(range(base + 8, base + 12)) + list(range(base + 16, base + 20))
    idx_kin += list(range(base + 4, base + 8)) + list(range(base + 12, base + 16)) + list(range(base + 20, base + 24))
idx_kin += [72, 73]
idx_phy = np.array(idx_phy)
idx_kin = np.array(idx_kin)


def loso_acc(Xf, y):
    accs = []
    for s in sorted(set(pid.tolist())):
        te = pid == s
        clf = RandomForestClassifier(max_depth=4, random_state=42, n_jobs=-1)
        clf.fit(Xf[~te], y[~te])
        accs.append(accuracy_score(y[te], clf.predict(Xf[te])) * 100)
    return float(np.mean(accs))


print("\nLOSO RF modality split")
split = {}
for y, yn in [(ya, "A"), (yv, "V"), (ya3, "A3"), (yv3, "V3")]:
    row = {
        "fused": loso_acc(feat, y),
        "physio": loso_acc(feat[:, idx_phy], y),
        "kin": loso_acc(feat[:, idx_kin], y),
        "speed": loso_acc(np.c_[spd_h, spd_e, acc_h], y),
    }
    split[yn] = row
    print(yn, {k: f"{v:.2f}" for k, v in row.items()})

out["loso_split"] = split
(ROOT / "results" / "confound_diag.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("saved results/confound_diag.json")
