"""TonicClock: the slow clock is the person, not 15 s of sickness.

CEAP labels are largely stimulus-locked. Leave-one-subject-out fails when
classifiers use each subject's DC offset as a spurious identity cue.
One subtraction, no extra backbone:

    x̃(s) = x(s) − mean_t x(s)     # unlabeled, per subject

Then the official RF (max_depth=4, random_state=80).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "ceap_protocol.npz")
pid = d["pid"]
feat, feat_fast = d["feat"], d["feat_fast"]
ys = {"V": d["y_val"], "A": d["y_aro"], "V3": d["y_val3"], "A3": d["y_aro3"]}
SEED = 80  # official CEAP script


def person_center(X: np.ndarray, pid: np.ndarray, zscore: bool = False) -> np.ndarray:
    Xc = X.astype(np.float32).copy()
    for s in np.unique(pid):
        m = pid == s
        mu = Xc[m].mean(0, keepdims=True)
        Xc[m] -= mu
        if zscore:
            sd = Xc[m].std(0, keepdims=True) + 1e-6
            Xc[m] /= sd
    return Xc


def loso():
    for s in sorted(set(pid.tolist())):
        te = pid == s
        yield np.where(~te)[0], np.where(te)[0]


def clf(name: str):
    if name == "RF":
        return RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=4)
    if name == "SVM":
        return make_pipeline(StandardScaler(), LinearSVC(max_iter=4000, dual="auto"))
    if name == "NB":
        return GaussianNB()
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    raise ValueError(name)


def eval_xy(name, X, y):
    accs, f1s = [], []
    for tr, te in loso():
        m = clf(name)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        accs.append(accuracy_score(y[te], pred) * 100)
        f1s.append(f1_score(y[te], pred, average="weighted", zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


def main():
    variants = {
        "RF": feat,
        "TwoClock-RF": feat_fast,
        "TonicClock-RF": person_center(feat, pid, False),
        "TonicZ-RF": person_center(feat, pid, True),
        "TwoClock+Tonic-RF": person_center(feat_fast, pid, False),
        "TwoClock+TonicZ-RF": person_center(feat_fast, pid, True),
    }
    # also Tonic + other official ML
    extra_ml = {
        "TonicClock-SVM": (person_center(feat, pid, False), "SVM"),
        "TonicClock-NB": (person_center(feat, pid, False), "NB"),
        "TonicZ-SVM": (person_center(feat, pid, True), "SVM"),
    }
    rows = {}
    for vname, X in variants.items():
        model = "RF"
        rows[vname] = {}
        for task, y in ys.items():
            acc, f1 = eval_xy(model, X, y)
            rows[vname][task] = {"acc": acc, "wf1": f1}
            print(f"{vname:22s} {task:3s} acc={acc:.2f} wf1={f1:.4f}", flush=True)
    for vname, (X, model) in extra_ml.items():
        rows[vname] = {}
        for task, y in ys.items():
            acc, f1 = eval_xy(model, X, y)
            rows[vname][task] = {"acc": acc, "wf1": f1}
            print(f"{vname:22s} {task:3s} acc={acc:.2f} wf1={f1:.4f}", flush=True)

    out = ROOT / "results" / "tonicclock.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pub = {
        "RF": (66.80, 64.26, 49.92, 52.20),
        "1D-CNN": (64.27, 67.64, 45.51, 47.17),
        "LSTM": (65.00, 66.30, 44.62, 43.79),
    }
    lines = [
        "# TonicClock vs official CEAP Table III (SI, 2s, 7680 windows, LOSO)",
        "",
        "| Model | V Acc | A Acc | V3 Acc | A3 Acc |",
        "|---|---:|---:|---:|---:|",
        "| RF (published) | 66.80 | 64.26 | 49.92 | 52.20 |",
        "| 1D-CNN (published) | 64.27 | **67.64** | 45.51 | 47.17 |",
        "| LSTM (published) | 65.00 | 66.30 | 44.62 | 43.79 |",
    ]
    for name, cell in rows.items():
        lines.append(
            f"| {name} | {cell['V']['acc']:.2f} | {cell['A']['acc']:.2f} | {cell['V3']['acc']:.2f} | {cell['A3']['acc']:.2f} |"
        )
    (ROOT / "results" / "tonicclock.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
