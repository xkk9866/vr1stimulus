"""Leave-one-video-out: does CEAP affect RF generalize to unseen 360 clips?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "ceap_protocol.npz")
feat, pid, vid = d["feat"], d["pid"], d["vid"]
ymap = {"V": d["y_val"], "A": d["y_aro"], "V3": d["y_val3"], "A3": d["y_aro3"]}

idx_phy, idx_kin = [], []
for b in range(3):
    base = b * 24
    idx_phy += list(range(base, base + 4)) + list(range(base + 8, base + 12)) + list(range(base + 16, base + 20))
    idx_kin += list(range(base + 4, base + 8)) + list(range(base + 12, base + 16)) + list(range(base + 20, base + 24))
idx_kin += [72, 73]
idx_phy = np.array(idx_phy)
idx_kin = np.array(idx_kin)


def run(Xf, y):
    accs, f1s = [], []
    for v in sorted(set(vid.tolist())):
        te = vid == v
        clf = RandomForestClassifier(max_depth=4, random_state=42, n_jobs=-1)
        clf.fit(Xf[~te], y[~te])
        pred = clf.predict(Xf[te])
        accs.append(accuracy_score(y[te], pred) * 100)
        f1s.append(f1_score(y[te], pred, average="weighted", zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s)), [float(a) for a in accs]


out = {}
print("Leave-one-video-out RF (8 clips)")
for yk, y in ymap.items():
    row = {}
    for name, Xf in (("fused", feat), ("physio", feat[:, idx_phy]), ("kin", feat[:, idx_kin])):
        acc, f1, per = run(Xf, y)
        row[name] = {"acc": acc, "wf1": f1, "per_video": per}
        print(f"  {yk:3s} {name:7s} acc={acc:.2f} wf1={f1:.4f} per={['%.1f'%p for p in per]}", flush=True)
    out[yk] = row

# chance from training prior applied to each held-out video
print("\nMajority-class prior (ignore features)")
prior = {}
for yk, y in ymap.items():
    accs = []
    for v in sorted(set(vid.tolist())):
        te = vid == v
        vals, cnt = np.unique(y[~te], return_counts=True)
        pred = np.full(te.sum(), vals[cnt.argmax()])
        accs.append(accuracy_score(y[te], pred) * 100)
    prior[yk] = float(np.mean(accs))
    print(f"  {yk} prior={prior[yk]:.2f}")
out["prior"] = prior

(ROOT / "results" / "lovo.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("saved results/lovo.json")
