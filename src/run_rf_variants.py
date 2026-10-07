"""RF-only extra: official RF vs TwoClock vs SSQ-subspace projection. LOSO."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from sickness_proj import fit_sickness_direction, project_out

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "ceap_protocol.npz")
pid, yv, ya, yv3, ya3 = d["pid"], d["y_val"], d["y_aro"], d["y_val3"], d["y_aro3"]
feat, feat_fast, ssq = d["feat"], d["feat_fast"], d["ssq"]
SEED = 42


def eval_pred(y, pred):
    return float(accuracy_score(y, pred) * 100), float(f1_score(y, pred, average="weighted", zero_division=0))


def loso():
    for s in sorted(set(pid.tolist())):
        te = pid == s
        yield np.where(~te)[0], np.where(te)[0]


def run(name, X):
    out = {}
    for key, y in (("V", yv), ("A", ya), ("V3", yv3), ("A3", ya3)):
        acc, f1 = [], []
        rs = []
        for tr, te in loso():
            if name == "Proj-RF":
                w = fit_sickness_direction(feat[tr], ssq[tr])
                Xtr, Xte = project_out(feat[tr], w), project_out(feat[te], w)
            else:
                Xtr, Xte = X[tr], X[te]
            clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            a, f = eval_pred(y[te], pred)
            acc.append(a)
            f1.append(f)
            if key in ("V", "A") and hasattr(clf, "predict_proba"):
                p = clf.predict_proba(Xte)
                if p.shape[1] == 2 and p[:, 1].std() > 1e-8 and ssq[te].std() > 1e-8:
                    rs.append(float(np.corrcoef(p[:, 1], ssq[te])[0, 1]))
        out[key] = {"acc": float(np.mean(acc)), "wf1": float(np.mean(f1)), "r": float(np.mean(rs) if rs else 0)}
        print(f"{name:10s} {key:3s} acc={out[key]['acc']:.2f} wf1={out[key]['wf1']:.4f} r={out[key]['r']:.3f}", flush=True)
    return out


def main():
    results = {
        "RF": run("RF", feat),
        "TwoClock-RF": run("TwoClock-RF", feat_fast),
        "Proj-RF": run("Proj-RF", feat),
    }
    (ROOT / "results" / "rf_variants.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
