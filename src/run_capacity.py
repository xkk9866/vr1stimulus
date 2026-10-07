"""Is the null on the deviation task a capacity artifact?

Reviewers will ask whether a depth-4 forest is simply too weak to read
the body. We repeat the deviation task with progressively stronger
learners and with every modality, and we test whether any sensor stream
adds accuracy on top of the sensor-free stimulus consensus.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
N_WIN = 30
PHY = np.arange(12, 24)
BEH = np.r_[0:12, 24:49]
ALL = np.arange(49)


def learner(name):
    if name == "RF-d4":
        return RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
    if name == "RF-d12":
        return RandomForestClassifier(max_depth=12, n_estimators=300, random_state=SEED, n_jobs=-1)
    if name == "RF-full":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    if name == "HGB":
        return HistGradientBoostingClassifier(random_state=SEED)
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    raise ValueError(name)


def wilcoxon(a, b):
    keys = sorted(set(a) & set(b))
    delta = np.array([b[k] - a[k] for k in keys], dtype=np.float64)
    if np.allclose(delta, 0):
        return {"delta": 0.0, "p": 1.0}
    st = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return {"delta": float(delta.mean()), "p": float(st.pvalue)}


def main():
    off = np.load(ROOT / "data" / "ceap_official.npz")
    cont = np.load(ROOT / "data" / "ceap_va_cont.npz")
    pid, vid, feat = off["pid"], off["vid"], off["feat"]
    win = np.tile(np.arange(N_WIN), len(pid) // N_WIN)
    subs = sorted(set(pid.tolist()))
    out = {"deviation_capacity": {}, "adds_on_top": {}}

    print("=== capacity on the deviation task ===", flush=True)
    for tname, ycont in (("D_V", cont["y_val_c"]), ("D_A", cont["y_aro_c"])):
        out["deviation_capacity"][tname] = {}
        for lname in ("RF-d4", "RF-d12", "RF-full", "HGB", "LogReg"):
            for mname, cols in (("all", ALL), ("phys", PHY), ("beh", BEH)):
                accs, f1s = [], []
                for s in subs:
                    te = pid == s
                    tr = ~te
                    ktr, kte = vid[tr] * 100 + win[tr], vid[te] * 100 + win[te]
                    cmap = {int(k): float(ycont[tr][ktr == k].mean()) for k in np.unique(ktr)}
                    g = float(ycont[tr].mean())
                    ytr = (ycont[tr] > np.array([cmap.get(int(k), g) for k in ktr])).astype(np.int64)
                    yte = (ycont[te] > np.array([cmap.get(int(k), g) for k in kte])).astype(np.int64)
                    clf = learner(lname)
                    clf.fit(feat[tr][:, cols], ytr)
                    pred = clf.predict(feat[te][:, cols])
                    accs.append(accuracy_score(yte, pred) * 100)
                    f1s.append(f1_score(yte, pred, average="weighted", zero_division=0))
                key = f"{lname}/{mname}"
                out["deviation_capacity"][tname][key] = {
                    "acc": float(np.mean(accs)),
                    "wf1": float(np.mean(f1s)),
                }
                print(f"{tname} {key:16s} acc={np.mean(accs):.2f} wf1={np.mean(f1s):.4f}", flush=True)

    print("\n=== does any sensor add on top of the stimulus consensus? ===", flush=True)
    tasks = {"V": off["y_val"], "A": off["y_aro"], "V3": off["y_val3"], "A3": off["y_aro3"]}
    for tname, y in tasks.items():
        n_cls = int(y.max()) + 1
        bags = {k: {} for k in ("Prior(v,t)", "+phys", "+beh", "+all")}
        for s in subs:
            te = pid == s
            tr = ~te
            ktr, kte = vid[tr] * 100 + win[tr], vid[te] * 100 + win[te]
            maj = int(np.bincount(y[tr], minlength=n_cls).argmax())
            pmap = {int(k): int(np.bincount(y[tr][ktr == k], minlength=n_cls).argmax()) for k in np.unique(ktr)}
            c_tr = np.array([pmap.get(int(k), maj) for k in ktr])
            c_te = np.array([pmap.get(int(k), maj) for k in kte])
            bags["Prior(v,t)"][int(s)] = accuracy_score(y[te], c_te) * 100
            onehot_tr = np.eye(n_cls, dtype=np.float32)[c_tr]
            onehot_te = np.eye(n_cls, dtype=np.float32)[c_te]
            for label, cols in (("+phys", PHY), ("+beh", BEH), ("+all", ALL)):
                Xtr = np.hstack([onehot_tr, feat[tr][:, cols]])
                Xte = np.hstack([onehot_te, feat[te][:, cols]])
                clf = HistGradientBoostingClassifier(random_state=SEED)
                clf.fit(Xtr, y[tr])
                bags[label][int(s)] = accuracy_score(y[te], clf.predict(Xte)) * 100
        out["adds_on_top"][tname] = {
            k: {"acc": float(np.mean(list(v.values())))} for k, v in bags.items()
        }
        for label in ("+phys", "+beh", "+all"):
            out["adds_on_top"][tname][label]["vs_prior"] = wilcoxon(bags["Prior(v,t)"], bags[label])
        print(tname, json.dumps(out["adds_on_top"][tname]), flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "capacity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/capacity.json", flush=True)


if __name__ == "__main__":
    main()
