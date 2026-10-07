"""Is CEAP affect classification secretly 360-clip identification via kinematics?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "ceap_protocol.npz")
feat, pid, vid = d["feat"], d["pid"], d["vid"]
ya, yv = d["y_aro"], d["y_val"]

idx_phy, idx_kin = [], []
for b in range(3):
    base = b * 24
    idx_phy += list(range(base, base + 4)) + list(range(base + 8, base + 12)) + list(range(base + 16, base + 20))
    idx_kin += list(range(base + 4, base + 8)) + list(range(base + 12, base + 16)) + list(range(base + 20, base + 24))
idx_kin += [72, 73]
idx_phy = np.array(idx_phy)
idx_kin = np.array(idx_kin)


def loso(Xf, y):
    acc = []
    for s in sorted(set(pid.tolist())):
        te = pid == s
        clf = RandomForestClassifier(max_depth=4, random_state=42, n_jobs=-1)
        clf.fit(Xf[~te], y[~te])
        acc.append(accuracy_score(y[te], clf.predict(Xf[te])) * 100)
    return float(np.mean(acc))


print("8-way video ID chance=12.5")
print("video ID fused", loso(feat, vid - 1))
print("video ID kin  ", loso(feat[:, idx_kin], vid - 1))
print("video ID phy  ", loso(feat[:, idx_phy], vid - 1))

# within-subject corr of demeaned physio vs speed is in diagnose; here: majority-video baseline for affect
print("\nIf we predict the training-majority label of each video id:")


def video_prior_loso(y):
    acc = []
    for s in sorted(set(pid.tolist())):
        te = pid == s
        pred = np.zeros(te.sum(), dtype=y.dtype)
        vids = vid[te]
        for i, v in enumerate(vids):
            m = (~te) & (vid == v)
            # majority class of this video in training subjects
            vals, cnt = np.unique(y[m], return_counts=True)
            pred[i] = vals[cnt.argmax()]
        acc.append(accuracy_score(y[te], pred) * 100)
    return float(np.mean(acc))


print("video-majority V", video_prior_loso(yv))
print("video-majority A", video_prior_loso(ya))
print("video-majority V3", video_prior_loso(d["y_val3"]))
print("video-majority A3", video_prior_loso(d["y_aro3"]))

out = {
    "video_id": {
        "fused": loso(feat, vid - 1),
        "kin": loso(feat[:, idx_kin], vid - 1),
        "phy": loso(feat[:, idx_phy], vid - 1),
        "chance": 12.5,
    },
    "video_majority_label": {
        "V": video_prior_loso(yv),
        "A": video_prior_loso(ya),
        "V3": video_prior_loso(d["y_val3"]),
        "A3": video_prior_loso(d["y_aro3"]),
    },
}
(ROOT / "results" / "video_leak.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
