"""Subject-dependent half of Xue et al. Table III, plus the stimulus prior.

Official SD: 10-fold cross validation inside each participant, class
balanced, RF with the default max_depth=2. Averaged over 32 participants.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
N_WIN = 30
PHY = np.arange(12, 24)
BEH = np.r_[0:12, 24:49]
ALL = np.arange(49)

PUB_SD = {
    "V": (68.45, 0.6315),
    "A": (71.33, 0.6487),
    "V3": (60.42, 0.5354),
    "A3": (62.38, 0.5457),
    "C5": (51.89, 0.4340),
}


def main():
    d = np.load(ROOT / "data" / "ceap_official.npz")
    pid, vid, feat = d["pid"], d["vid"], d["feat"]
    win = np.tile(np.arange(N_WIN), len(pid) // N_WIN)
    tasks = {
        "V": d["y_val"],
        "A": d["y_aro"],
        "V3": d["y_val3"],
        "A3": d["y_aro3"],
        "C5": d["y5"],
    }
    subs = sorted(set(pid.tolist()))
    out = {"published_sd": PUB_SD, "rows": {}}

    for task, y in tasks.items():
        bags = {k: {"acc": [], "f1": []} for k in ("RF(all)", "RF(phys)", "RF(beh)", "Prior(v)", "Prior(v,t)")}
        for s in subs:
            m = pid == s
            ys, fs, vs, ws = y[m], feat[m], vid[m], win[m]
            n_cls = int(y.max()) + 1
            if len(np.unique(ys)) < 2:
                continue
            counts = np.bincount(ys, minlength=n_cls)
            n_splits = int(min(10, counts[counts > 0].min()))
            if n_splits < 2:
                continue
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            per = {k: [] for k in bags}
            perf = {k: [] for k in bags}
            for tr, te in skf.split(fs, ys):
                maj = int(np.bincount(ys[tr], minlength=n_cls).argmax())
                for name, cols in (("RF(all)", ALL), ("RF(phys)", PHY), ("RF(beh)", BEH)):
                    clf = RandomForestClassifier(max_depth=2, random_state=SEED, n_jobs=-1)
                    clf.fit(fs[tr][:, cols], ys[tr])
                    pred = clf.predict(fs[te][:, cols])
                    per[name].append(accuracy_score(ys[te], pred) * 100)
                    perf[name].append(f1_score(ys[te], pred, average="weighted", zero_division=0))
                for name, keyf in (("Prior(v)", lambda i: vs[i]), ("Prior(v,t)", lambda i: vs[i] * 100 + ws[i])):
                    ktr, kte = keyf(tr), keyf(te)
                    pmap = {
                        int(k): int(np.bincount(ys[tr][ktr == k], minlength=n_cls).argmax())
                        for k in np.unique(ktr)
                    }
                    pred = np.array([pmap.get(int(k), maj) for k in kte])
                    per[name].append(accuracy_score(ys[te], pred) * 100)
                    perf[name].append(f1_score(ys[te], pred, average="weighted", zero_division=0))
            for k in bags:
                bags[k]["acc"].append(float(np.mean(per[k])))
                bags[k]["f1"].append(float(np.mean(perf[k])))
        out["rows"][task] = {
            k: {"acc": float(np.mean(v["acc"])), "wf1": float(np.mean(v["f1"]))} for k, v in bags.items()
        }
        for k, v in out["rows"][task].items():
            print(f"SD {task:3s} {k:12s} acc={v['acc']:.2f} wf1={v['wf1']:.4f}", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "sd_tables.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/sd_tables.json", flush=True)


if __name__ == "__main__":
    main()
