"""Clip identity decoding and leave-one-video-out, on the official features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
PHY = np.arange(12, 24)
BEH = np.r_[0:12, 24:49]
ALL = np.arange(49)


def main():
    d = np.load(ROOT / "data" / "ceap_official.npz")
    pid, vid, feat = d["pid"], d["vid"], d["feat"]
    tasks = {"V": d["y_val"], "A": d["y_aro"], "V3": d["y_val3"], "A3": d["y_aro3"], "C5": d["y5"]}
    out = {}

    print("=== 8-way clip identification, LOSO (chance 12.5) ===", flush=True)
    out["clip_id"] = {"chance": 12.5}
    for name, cols in (("All", ALL), ("Behavior", BEH), ("Physio", PHY)):
        accs = []
        for s in sorted(set(pid.tolist())):
            te = pid == s
            clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
            clf.fit(feat[~te][:, cols], vid[~te])
            accs.append(accuracy_score(vid[te], clf.predict(feat[te][:, cols])) * 100)
        out["clip_id"][name] = float(np.mean(accs))
        print(f"  {name:9s} {np.mean(accs):.2f}", flush=True)

    print("\n=== leave-one-video-out (8 folds) ===", flush=True)
    out["lovo"] = {}
    for task, y in tasks.items():
        n_cls = int(y.max()) + 1
        row = {}
        for name, cols in (("All", ALL), ("Behavior", BEH), ("Physio", PHY)):
            accs, f1s = [], []
            for v in sorted(set(vid.tolist())):
                te = vid == v
                clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
                clf.fit(feat[~te][:, cols], y[~te])
                pred = clf.predict(feat[te][:, cols])
                accs.append(accuracy_score(y[te], pred) * 100)
                f1s.append(f1_score(y[te], pred, average="weighted", zero_division=0))
            row[name] = {"acc": float(np.mean(accs)), "wf1": float(np.mean(f1s))}
        accs, f1s = [], []
        for v in sorted(set(vid.tolist())):
            te = vid == v
            maj = int(np.bincount(y[~te], minlength=n_cls).argmax())
            pred = np.full(te.sum(), maj)
            accs.append(accuracy_score(y[te], pred) * 100)
            f1s.append(f1_score(y[te], pred, average="weighted", zero_division=0))
        row["Majority"] = {"acc": float(np.mean(accs)), "wf1": float(np.mean(f1s))}
        out["lovo"][task] = row
        print(f"  {task:3s} " + "  ".join(f"{k}={v['acc']:.2f}" for k, v in row.items()), flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "stimulus_id.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/stimulus_id.json", flush=True)


if __name__ == "__main__":
    main()
