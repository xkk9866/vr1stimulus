"""Reproduce Xue et al. TMM 2021 Tables III–IV on official 49-d features.

SI = LOSO, RF max_depth=4 random_state=80 (baseline code).
Metrics = Accuracy + weighted F1. Binary / 3-class / 5-class.
OrthoVR on this protocol = behavior-only (HM/EM+PD+fixation): looking is not fused with the body.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ceap_official.npz"
OUT = ROOT / "results"
SEED = 80  # official Classifer_models.py

# 12 series × (mean,std,median) then 13 fixation/saccade stats.
# series order: HM, EM, PD_Mean, PD_Emotion, EDA, BVP, HR, SKT, HM_Pitch, HM_Yaw, EM_Pitch, EM_Yaw
ALL = np.arange(49)
PHY = np.arange(12, 24)  # EDA BVP HR SKT
BEH = np.r_[0:12, 24:49]  # HM/EM + PD + pose + fixation/saccade
KIN = np.r_[0:6, 24:49]  # HM/EM + pose + fixation/saccade, no pupil

PUB_SI = {
    "RF": dict(V=(66.80, 0.6238), A=(64.26, 0.5298), V3=(49.92, 0.4419), A3=(52.20, 0.4341), C5=(31.47, 0.3001)),
    "1D-CNN": dict(V=(64.27, 0.5828), A=(67.64, 0.5808), V3=(45.51, 0.4191), A3=(47.17, 0.3923), C5=(29.87, 0.2598)),
    "LSTM": dict(V=(65.00, 0.6349), A=(66.30, 0.5934), V3=(44.62, 0.4269), A3=(43.79, 0.4085), C5=(30.12, 0.2768)),
}
PUB_IV_SI = {
    "Physio": dict(V=(65.69, 0.6126), A=(62.17, 0.5007), V3=(44.66, 0.4027), A3=(51.45, 0.4200), C5=(30.56, 0.2439)),
    "HM/EM+PD": dict(V=(62.68, 0.5344), A=(62.43, 0.5022), V3=(47.90, 0.4124), A3=(50.00, 0.3555), C5=(26.33, 0.2440)),
    "All": dict(V=(66.80, 0.6238), A=(64.26, 0.5298), V3=(49.92, 0.4419), A3=(52.20, 0.4341), C5=(31.47, 0.3001)),
}


def model(name):
    if name == "RF":
        return RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
    if name == "SVM":
        return make_pipeline(StandardScaler(), SVC(decision_function_shape="ovr"))
    if name == "NB":
        return GaussianNB()
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    raise ValueError(name)


def loso(pid):
    for s in sorted(set(pid.tolist())):
        te = pid == s
        yield s, np.where(~te)[0], np.where(te)[0]


def run(name, X, y, pid):
    accs, f1s, subj = [], [], {}
    for s, tr, te in loso(pid):
        clf = model(name)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        a = accuracy_score(y[te], pred) * 100
        accs.append(a)
        f1s.append(f1_score(y[te], pred, average="weighted", zero_division=0))
        subj[int(s)] = float(a)
    return {"acc": float(np.mean(accs)), "wf1": float(np.mean(f1s)), "std": float(np.std(accs)), "subj": subj}


def wilcoxon(a, b):
    keys = sorted(set(a) & set(b))
    av = np.array([a[k] for k in keys])
    bv = np.array([b[k] for k in keys])
    d = bv - av
    if np.allclose(d, 0):
        return {"delta": 0.0, "p": 1.0}
    st = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"delta": float(d.mean()), "p": float(st.pvalue)}


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    d = np.load(DATA)
    pid = d["pid"]
    feat = d["feat"]
    print("N", len(pid), "feat", feat.shape, "part", part, flush=True)
    tasks = {
        "V": d["y_val"],
        "A": d["y_aro"],
        "V3": d["y_val3"],
        "A3": d["y_aro3"],
        "C5": d["y5"],
    }
    slices = {"All": ALL, "Physio": PHY, "HM/EM+PD": BEH, "Kin": KIN}
    cache = OUT / "official_tables.json"
    out = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    out.setdefault("si", {})
    out.setdefault("table4", {})
    out.setdefault("tests", {})
    out.setdefault("subj_acc", {})

    if part in ("all", "t3"):
        print("\n=== Table III SI ML (official RF seed 80) ===", flush=True)
        for mname in ("RF", "SVM", "NB", "KNN"):
            out["si"][mname] = {}
            for t, y in tasks.items():
                r = run(mname, feat, y, pid)
                out["si"][mname][t] = {k: r[k] for k in ("acc", "wf1", "std")}
                print(f"{mname:4s} {t:3s} acc={r['acc']:.2f} wf1={r['wf1']:.4f}", flush=True)

    if part in ("all", "t4"):
        print("\n=== Table IV SI RF ablation ===", flush=True)
        rf_subj = {}
        for sname, sl in slices.items():
            X = feat[:, sl]
            out["table4"][sname] = {}
            rf_subj[sname] = {}
            for t, y in tasks.items():
                r = run("RF", X, y, pid)
                out["table4"][sname][t] = {k: r[k] for k in ("acc", "wf1", "std")}
                rf_subj[sname][t] = r["subj"]
                print(f"{sname:10s} {t:3s} acc={r['acc']:.2f} wf1={r['wf1']:.4f}", flush=True)
        out["subj_acc"] = rf_subj
        for t in tasks:
            out["tests"][t] = {
                "beh_vs_all": wilcoxon(rf_subj["All"][t], rf_subj["HM/EM+PD"][t]),
                "kin_vs_all": wilcoxon(rf_subj["All"][t], rf_subj["Kin"][t]),
                "phy_vs_all": wilcoxon(rf_subj["All"][t], rf_subj["Physio"][t]),
            }
            print(t, out["tests"][t], flush=True)

    payload = {"published_si": PUB_SI, "published_iv_si": PUB_IV_SI, **out}
    OUT.mkdir(exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not out["si"] or not out["table4"]:
        print("partial run; tables not regenerated", flush=True)
        return

    def cell(block, t):
        return f"{block[t]['acc']:.2f} / {block[t]['wf1']:.4f}"

    lines = [
        "# Xue et al. TMM 2021 Table III/IV reproduction (official 49-d features, LOSO, RF seed=80)",
        "",
        "## Table III SI (Acc / wF1)",
        "",
        "| Model | V-2 | A-2 | V-3 | A-3 | 5-class |",
        "|---|---|---|---|---|---|",
        "| RF published | 66.80 / 0.6238 | 64.26 / 0.5298 | 49.92 / 0.4419 | 52.20 / 0.4341 | 31.47 / 0.3001 |",
        "| 1D-CNN published | 64.27 / 0.5828 | **67.64** / 0.5808 | 45.51 / 0.4191 | 47.17 / 0.3923 | 29.87 / 0.2598 |",
        "| LSTM published | 65.00 / 0.6349 | 66.30 / 0.5934 | 44.62 / 0.4269 | 43.79 / 0.4085 | 30.12 / 0.2768 |",
    ]
    for mname in ("RF", "SVM", "NB", "KNN"):
        b = out["si"][mname]
        lines.append(f"| {mname} ours | {cell(b,'V')} | {cell(b,'A')} | {cell(b,'V3')} | {cell(b,'A3')} | {cell(b,'C5')} |")
    lines += [
        "",
        "## Table IV SI RF ablation",
        "",
        "| Features | V-2 | A-2 | V-3 | A-3 | 5-class |",
        "|---|---|---|---|---|---|",
        "| Physio published | 65.69 / 0.6126 | 62.17 / 0.5007 | 44.66 / 0.4027 | 51.45 / 0.4200 | 30.56 / 0.2439 |",
        "| HM/EM+PD published | 62.68 / 0.5344 | 62.43 / 0.5022 | 47.90 / 0.4124 | 50.00 / 0.3555 | 26.33 / 0.2440 |",
        "| All published | 66.80 / 0.6238 | 64.26 / 0.5298 | 49.92 / 0.4419 | 52.20 / 0.4341 | 31.47 / 0.3001 |",
    ]
    for sname in ("Physio", "HM/EM+PD", "Kin", "All"):
        b = out["table4"][sname]
        tag = "OrthoVR (behavior)" if sname == "HM/EM+PD" else sname
        lines.append(f"| {tag} ours | {cell(b,'V')} | {cell(b,'A')} | {cell(b,'V3')} | {cell(b,'A3')} | {cell(b,'C5')} |")
    (OUT / "official_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved results/official_tables.md", flush=True)


if __name__ == "__main__":
    main()
