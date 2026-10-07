"""The corrected CEAP protocol: deviation from the stimulus consensus.

Official labels are dominated by the clip. Once the crowd trajectory
c_{v,t} is removed, what remains is whether this viewer feels
differently from everyone else watching the same second of the same
clip. That residual is what a wearable must supply and what a therapy
system actually needs.

Task D  : y_cont > c_{v,t}, consensus estimated on training subjects only
Task R  : y_cont > this subject's own median within the clip

Method (one step, no new backbone): session normalization. Each subject's
features are z-scored with that subject's own unlabeled session
statistics before the official Random Forest. Between-person tonic level
is the dominant nuisance once the stimulus is gone.
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
        return {"delta": 0.0, "p": 1.0, "n": len(keys)}
    st = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return {"delta": float(delta.mean()), "p": float(st.pvalue), "n": len(keys)}


def session_norm(feat, pid):
    """Z-score every feature within each subject's own session. No labels used."""
    out = feat.astype(np.float64).copy()
    for p in np.unique(pid):
        m = pid == p
        mu = out[m].mean(0, keepdims=True)
        sd = out[m].std(0, keepdims=True) + 1e-8
        out[m] = (out[m] - mu) / sd
    return out.astype(np.float32)


def causal_norm(feat, pid, vid):
    """Running mean/std within a session, causal. Deployable in a headset."""
    out = feat.astype(np.float64).copy()
    order = np.lexsort((np.arange(len(pid)), vid, pid))
    for p in np.unique(pid):
        idx = order[pid[order] == p]
        x = out[idx]
        csum = np.cumsum(x, axis=0)
        csq = np.cumsum(x ** 2, axis=0)
        n = np.arange(1, len(idx) + 1)[:, None]
        mu = csum / n
        var = np.maximum(csq / n - mu ** 2, 0.0)
        out[idx] = (x - mu) / (np.sqrt(var) + 1e-6)
    return np.nan_to_num(out).astype(np.float32)


def main():
    off = np.load(ROOT / "data" / "ceap_official.npz")
    cont = np.load(ROOT / "data" / "ceap_va_cont.npz")
    pid, vid, feat = off["pid"], off["vid"], off["feat"]
    win = np.tile(np.arange(N_WIN), len(pid) // N_WIN)
    val_c, aro_c = cont["y_val_c"], cont["y_aro_c"]
    assert len(val_c) == len(pid)

    feat_sn = session_norm(feat, pid)
    feat_cn = causal_norm(feat, pid, vid)
    subs = sorted(set(pid.tolist()))

    variants = {
        "RF(all)": (feat, ALL),
        "RF(phys)": (feat, PHY),
        "RF(beh)": (feat, BEH),
        "SessionNorm-RF(all)": (feat_sn, ALL),
        "SessionNorm-RF(phys)": (feat_sn, PHY),
        "CausalNorm-RF(all)": (feat_cn, ALL),
    }

    results = {}
    subj_acc = {}
    for tname, ycont in (("D_V", val_c), ("D_A", aro_c), ("R_V", val_c), ("R_A", aro_c)):
        bags = {k: {"acc": [], "f1": [], "subj": {}} for k in list(variants) + ["Prior(v,t)", "Chance"]}
        for s in subs:
            te = pid == s
            tr = ~te
            if tname.startswith("D_"):
                keys_tr = vid[tr] * 100 + win[tr]
                keys_te = vid[te] * 100 + win[te]
                cmap = {}
                for k in np.unique(keys_tr):
                    cmap[int(k)] = float(ycont[tr][keys_tr == k].mean())
                gmean = float(ycont[tr].mean())
                c_tr = np.array([cmap.get(int(k), gmean) for k in keys_tr])
                c_te = np.array([cmap.get(int(k), gmean) for k in keys_te])
                y_tr = (ycont[tr] > c_tr).astype(np.int64)
                y_te = (ycont[te] > c_te).astype(np.int64)
            else:
                y_all = np.zeros(len(ycont), dtype=np.int64)
                for p in np.unique(pid):
                    for v in np.unique(vid):
                        m = (pid == p) & (vid == v)
                        y_all[m] = (ycont[m] > np.median(ycont[m])).astype(np.int64)
                y_tr, y_te = y_all[tr], y_all[te]

            maj = int(np.bincount(y_tr, minlength=2).argmax())
            a, f = scores(y_te, np.full(len(y_te), maj))
            bags["Chance"]["acc"].append(a)
            bags["Chance"]["f1"].append(f)
            bags["Chance"]["subj"][int(s)] = a

            keys_tr = vid[tr] * 100 + win[tr]
            keys_te = vid[te] * 100 + win[te]
            pmap = {}
            for k in np.unique(keys_tr):
                sel = y_tr[keys_tr == k]
                pmap[int(k)] = int(np.bincount(sel, minlength=2).argmax())
            pred = np.array([pmap.get(int(k), maj) for k in keys_te])
            a, f = scores(y_te, pred)
            bags["Prior(v,t)"]["acc"].append(a)
            bags["Prior(v,t)"]["f1"].append(f)
            bags["Prior(v,t)"]["subj"][int(s)] = a

            for name, (X, cols) in variants.items():
                clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
                clf.fit(X[tr][:, cols], y_tr)
                a, f = scores(y_te, clf.predict(X[te][:, cols]))
                bags[name]["acc"].append(a)
                bags[name]["f1"].append(f)
                bags[name]["subj"][int(s)] = a

        results[tname] = {}
        subj_acc[tname] = {}
        for name, b in bags.items():
            results[tname][name] = {"acc": float(np.mean(b["acc"])), "wf1": float(np.mean(b["f1"]))}
            subj_acc[tname][name] = b["subj"]
            print(f"{tname:4s} {name:22s} acc={np.mean(b['acc']):.2f} wf1={np.mean(b['f1']):.4f}", flush=True)

    tests = {}
    for tname in results:
        sa = subj_acc[tname]
        tests[tname] = {
            "sessionnorm_vs_rf_all": wilcoxon(sa["RF(all)"], sa["SessionNorm-RF(all)"]),
            "sessionnorm_phys_vs_rf_phys": wilcoxon(sa["RF(phys)"], sa["SessionNorm-RF(phys)"]),
            "sessionnorm_vs_chance": wilcoxon(sa["Chance"], sa["SessionNorm-RF(all)"]),
            "rf_all_vs_chance": wilcoxon(sa["Chance"], sa["RF(all)"]),
            "causal_vs_rf_all": wilcoxon(sa["RF(all)"], sa["CausalNorm-RF(all)"]),
        }
        print(tname, json.dumps(tests[tname]), flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "corrected.json").write_text(
        json.dumps({"rows": results, "tests": tests, "subj_acc": subj_acc}, indent=2), encoding="utf-8"
    )
    print("saved results/corrected.json", flush=True)


if __name__ == "__main__":
    main()
