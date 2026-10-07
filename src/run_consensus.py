"""Stimulus-consensus decomposition on the official CEAP protocol.

y_{s,v,t} = c_{v,t} + d_{s,v,t}

c is the crowd consensus for clip v at window t, estimable from training
subjects alone with no sensor input. d is the individual deviation, the
only part a wearable can contribute.

Rows:
  Prior(global)  majority class over training subjects
  Prior(v)       majority class per clip
  Prior(v,t)     majority class per clip and window index  <- sensor-free
  RF(all)        official fused Random Forest
  RF + consensus consensus prior with a physiological deviation term

LOSO, official 49-d features, RF max_depth=4 random_state=80.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
N_WIN = 30

PHY = np.arange(12, 24)
BEH = np.r_[0:12, 24:49]
ALL = np.arange(49)


def scores(y, pred):
    return (
        float(accuracy_score(y, pred) * 100),
        float(f1_score(y, pred, average="weighted", zero_division=0)),
    )


def wilcoxon(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    delta = np.array([b[k] - a[k] for k in keys], dtype=np.float64)
    if np.allclose(delta, 0):
        return {"delta": 0.0, "p": 1.0}
    st = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return {"delta": float(delta.mean()), "p": float(st.pvalue)}


def majority(y, n_cls):
    counts = np.bincount(y, minlength=n_cls)
    return int(counts.argmax())


def consensus_table(y, vid, win, n_cls, key):
    """Return dict cell -> majority label over the given training rows."""
    table = {}
    if key == "global":
        return {(): majority(y, n_cls)}
    keys = vid if key == "v" else vid * 100 + win
    for k in np.unique(keys):
        table[int(k)] = majority(y[keys == k], n_cls)
    return table


def consensus_predict(table, vid, win, key, fallback):
    if key == "global":
        return np.full(len(vid), table[()], dtype=np.int64)
    keys = vid if key == "v" else vid * 100 + win
    return np.array([table.get(int(k), fallback) for k in keys], dtype=np.int64)


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
    rows = {}
    subj = {}

    def record(name, task, acc, f1, per_subj):
        rows.setdefault(name, {})[task] = {"acc": acc, "wf1": f1}
        subj.setdefault(name, {})[task] = per_subj

    for task, y in tasks.items():
        n_cls = int(y.max()) + 1
        bags = {
            k: {"acc": [], "f1": [], "subj": {}}
            for k in ("Prior(global)", "Prior(v)", "Prior(v,t)", "RF(all)", "RF(phys)", "Consensus+RF(phys)")
        }
        for s in subs:
            te = pid == s
            tr = ~te
            yte = y[te]
            fallback = majority(y[tr], n_cls)

            for name, key in (("Prior(global)", "global"), ("Prior(v)", "v"), ("Prior(v,t)", "vt")):
                tab = consensus_table(y[tr], vid[tr], win[tr], n_cls, key)
                pred = consensus_predict(tab, vid[te], win[te], key, fallback)
                a, f = scores(yte, pred)
                bags[name]["acc"].append(a)
                bags[name]["f1"].append(f)
                bags[name]["subj"][int(s)] = a

            for name, cols in (("RF(all)", ALL), ("RF(phys)", PHY)):
                clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
                clf.fit(feat[tr][:, cols], y[tr])
                pred = clf.predict(feat[te][:, cols])
                a, f = scores(yte, pred)
                bags[name]["acc"].append(a)
                bags[name]["f1"].append(f)
                bags[name]["subj"][int(s)] = a

            # consensus prior + physiological deviation:
            # learn where the body says this person departs from the crowd.
            tab = consensus_table(y[tr], vid[tr], win[tr], n_cls, "vt")
            c_tr = consensus_predict(tab, vid[tr], win[tr], "vt", fallback)
            c_te = consensus_predict(tab, vid[te], win[te], "vt", fallback)
            dev_tr = (y[tr] != c_tr).astype(np.int64)
            if len(np.unique(dev_tr)) < 2:
                pred = c_te
            else:
                clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
                clf.fit(feat[tr][:, PHY], dev_tr)
                flip = clf.predict(feat[te][:, PHY]).astype(bool)
                pred = c_te.copy()
                if n_cls == 2:
                    pred[flip] = 1 - pred[flip]
                else:
                    alt = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
                    alt.fit(feat[tr][:, PHY], y[tr])
                    pred[flip] = alt.predict(feat[te][:, PHY])[flip]
            a, f = scores(yte, pred)
            bags["Consensus+RF(phys)"]["acc"].append(a)
            bags["Consensus+RF(phys)"]["f1"].append(f)
            bags["Consensus+RF(phys)"]["subj"][int(s)] = a

        for name, b in bags.items():
            record(name, task, float(np.mean(b["acc"])), float(np.mean(b["f1"])), b["subj"])
            print(f"{task:3s} {name:20s} acc={np.mean(b['acc']):.2f} wf1={np.mean(b['f1']):.4f}", flush=True)

    tests = {}
    for task in tasks:
        tests[task] = {
            "priorvt_vs_rfall": wilcoxon(subj["RF(all)"][task], subj["Prior(v,t)"][task]),
            "consensus_vs_priorvt": wilcoxon(subj["Prior(v,t)"][task], subj["Consensus+RF(phys)"][task]),
        }
        print(task, tests[task], flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "consensus.json").write_text(
        json.dumps({"rows": rows, "tests": tests, "subj_acc": subj}, indent=2), encoding="utf-8"
    )
    print("saved results/consensus.json", flush=True)


if __name__ == "__main__":
    main()
