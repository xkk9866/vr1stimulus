"""Official-style eye stats + kinematics-only RF. Can Table III be beaten without physiology?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from orthovr import eye_stats_batch

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "ceap_protocol.npz")
X, feat, pid = d["X"], d["feat"], d["pid"]
ymap = {"V": d["y_val"], "A": d["y_aro"], "V3": d["y_val3"], "A3": d["y_aro3"]}

idx_kin = []
for b in range(3):
    base = b * 24
    idx_kin += list(range(base + 4, base + 8)) + list(range(base + 12, base + 16)) + list(range(base + 20, base + 24))
idx_kin += [72, 73]
idx_kin = np.array(idx_kin)

print("computing official fixation/saccade stats...", flush=True)
eye = eye_stats_batch(X)
kin = feat[:, idx_kin]
kin_eye = np.concatenate([kin, eye], axis=1)
fused_eye = np.concatenate([feat, eye], axis=1)
print("kin", kin.shape, "kin+eye", kin_eye.shape, flush=True)


def loso(Xf, y):
    accs, f1s = [], []
    for s in sorted(set(pid.tolist())):
        te = pid == s
        clf = RandomForestClassifier(max_depth=4, random_state=42, n_jobs=-1)
        clf.fit(Xf[~te], y[~te])
        pred = clf.predict(Xf[te])
        accs.append(accuracy_score(y[te], pred) * 100)
        f1s.append(f1_score(y[te], pred, average="weighted", zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


out = {"published": {"V": 66.80, "A": 64.26, "V3": 49.92, "A3": 52.20}}
for name, Xf in (("kin", kin), ("kin+eye", kin_eye), ("fused+eye", fused_eye)):
    out[name] = {}
    for yk, y in ymap.items():
        acc, f1 = loso(Xf, y)
        out[name][yk] = {"acc": acc, "wf1": f1}
        print(f"{name:10s} {yk:3s} acc={acc:.2f} wf1={f1:.4f}", flush=True)

(ROOT / "results" / "kin_sota.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("saved results/kin_sota.json")
