"""OrthoVR vs fused RF under Xue et al. Table III protocol.

LOSO, Acc + weighted F1, RF max_depth=4.
Scientific tests: r(pred, head-speed), injection recovery, Wilcoxon on 32 subjects.
Feature extraction is per fold (because OrthoVR changes physiology), not per task.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from orthovr import (
    OrthoVR,
    eye_stats_batch,
    features_from_X,
    inject_spin,
    motion_load,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 42
TASKS = ("V", "A", "V3", "A3")


def load():
    d = np.load(ROOT / "data" / "ceap_protocol.npz")
    return {
        "X": d["X"],
        "pid": d["pid"],
        "vid": d["vid"],
        "ssq": d["ssq"],
        "y": {"V": d["y_val"], "A": d["y_aro"], "V3": d["y_val3"], "A3": d["y_aro3"]},
    }


def eval_pred(y, pred):
    return float(accuracy_score(y, pred) * 100), float(f1_score(y, pred, average="weighted", zero_division=0))


def loso_indices(pid):
    for s in sorted(set(pid.tolist())):
        te = pid == s
        yield s, np.where(~te)[0], np.where(te)[0]


def safe_r(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_loso(name, X, data, eye, speed, use_ortho: bool):
    pid, vid, ssq, ymap = data["pid"], data["vid"], data["ssq"], data["y"]
    bucket = {t: {"acc": [], "f1": [], "r_spd": [], "r_ssq": [], "subj": {}} for t in TASKS}
    for s, tr, te in loso_indices(pid):
        if use_ortho:
            m = OrthoVR(alpha=1.0)
            m.fit(X[tr], pid[tr], vid[tr])
            Xtr = m.transform(X[tr], pid[tr], vid[tr])
            Xte = m.transform(X[te], pid[te], vid[te])
        else:
            Xtr, Xte = X[tr], X[te]
        ftr = features_from_X(Xtr, eye[tr] if eye is not None else None)
        fte = features_from_X(Xte, eye[te] if eye is not None else None)
        for task in TASKS:
            y = ymap[task]
            clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
            clf.fit(ftr, y[tr])
            pred = clf.predict(fte)
            a, f = eval_pred(y[te], pred)
            bucket[task]["acc"].append(a)
            bucket[task]["f1"].append(f)
            bucket[task]["subj"][int(s)] = a
            if task in ("V", "A") and hasattr(clf, "predict_proba"):
                p = clf.predict_proba(fte)
                if p.shape[1] == 2:
                    bucket[task]["r_spd"].append(safe_r(p[:, 1], speed[te]))
                    bucket[task]["r_ssq"].append(safe_r(p[:, 1], ssq[te]))
        print(f"  {name} subject {int(s)} A={bucket['A']['acc'][-1]:.1f} V={bucket['V']['acc'][-1]:.1f}", flush=True)

    cell = {"model": name, "subject": {}}
    for task in TASKS:
        cell[f"{task}_acc"] = float(np.mean(bucket[task]["acc"]))
        cell[f"{task}_acc_std"] = float(np.std(bucket[task]["acc"]))
        cell[f"{task}_wf1"] = float(np.mean(bucket[task]["f1"]))
        cell["subject"][task] = bucket[task]["subj"]
        if bucket[task]["r_spd"]:
            cell[f"{task}_r_speed"] = float(np.mean(bucket[task]["r_spd"]))
            cell[f"{task}_r_ssq"] = float(np.mean(bucket[task]["r_ssq"]))
        print(
            f"{name:18s} {task:3s} acc={cell[f'{task}_acc']:.2f} "
            f"wf1={cell[f'{task}_wf1']:.4f} "
            f"r_speed={cell.get(f'{task}_r_speed', float('nan')):.3f}",
            flush=True,
        )
    return cell


def wilcoxon_pair(a_subj: dict, b_subj: dict):
    keys = sorted(set(a_subj) & set(b_subj))
    av = np.array([a_subj[k] for k in keys], dtype=np.float64)
    bv = np.array([b_subj[k] for k in keys], dtype=np.float64)
    delta = bv - av
    if np.allclose(delta, 0):
        return {"n": len(keys), "stat": 0.0, "p": 1.0, "mean_delta": 0.0}
    st = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return {
        "n": len(keys),
        "stat": float(st.statistic),
        "p": float(st.pvalue),
        "mean_delta": float(delta.mean()),
    }


def main():
    data = load()
    X = data["X"]
    print("computing eye stats and motion...", flush=True)
    eye = eye_stats_batch(X)
    speed = motion_load(X)[:, :, 0].mean(1)
    print("eye", eye.shape, "speed", float(speed.mean()), float(speed.std()), flush=True)

    rows = []
    rows.append(run_loso("RF", X, data, eye=None, speed=speed, use_ortho=False))
    rows.append(run_loso("RF+eye", X, data, eye=eye, speed=speed, use_ortho=False))
    rows.append(run_loso("OrthoVR-RF", X, data, eye=None, speed=speed, use_ortho=True))
    rows.append(run_loso("OrthoVR-RF+eye", X, data, eye=eye, speed=speed, use_ortho=True))

    tests = {}
    rf = next(r for r in rows if r["model"] == "RF")
    ov = next(r for r in rows if r["model"] == "OrthoVR-RF")
    for task in TASKS:
        tests[task] = wilcoxon_pair(rf["subject"][task], ov["subject"][task])
        print(
            f"Wilcoxon OrthoVR-RF vs RF {task}: Δ={tests[task]['mean_delta']:.2f} p={tests[task]['p']:.4f}",
            flush=True,
        )

    print("\n=== injection recovery (gamma=1.0 on EDA+HR) ===", flush=True)
    xinj = inject_spin(X, gamma=1.0)
    inj_rows = []
    inj_rows.append(run_loso("RF-inj", xinj, data, eye=None, speed=speed, use_ortho=False))
    inj_rows.append(run_loso("OrthoVR-inj", xinj, data, eye=None, speed=speed, use_ortho=True))

    payload = {
        "protocol": "Xue TMM 2021 Table III, LOSO, 2s, RF max_depth=4, seed=42",
        "N": int(len(X)),
        "published": {"RF_V": 66.80, "RF_A": 64.26, "CNN_A": 67.64, "RF_V3": 49.92, "RF_A3": 52.20},
        "rows": [{k: v for k, v in r.items() if k != "subject"} for r in rows],
        "wilcoxon_OrthoVR_vs_RF": tests,
        "injection": [{k: v for k, v in r.items() if k != "subject"} for r in inj_rows],
        "subject_acc": {r["model"]: r["subject"] for r in rows},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "orthovr.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# OrthoVR vs fused RF (CEAP-360VR, 32-subject LOSO, 7680 windows)",
        "",
        "Method: ridge-residualize EDA/BVP/HR/SKT on head/eye speed and acceleration; keep pose and (optional) official fixation/saccade; same RF.",
        "",
        "| Model | V Acc | A Acc | V3 Acc | A3 Acc | A r(pred, speed) | A r(pred, SSQ) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| RF (published) | **66.80** | 64.26 | **49.92** | 52.20 | — | — |",
    ]
    for r in rows + inj_rows:
        lines.append(
            f"| {r['model']} | {r['V_acc']:.2f} | {r['A_acc']:.2f} | {r['V3_acc']:.2f} | {r['A3_acc']:.2f} | "
            f"{r.get('A_r_speed', float('nan')):.3f} | {r.get('A_r_ssq', float('nan')):.3f} |"
        )
    lines.append("")
    lines.append("Wilcoxon (32 subject accuracies, two-sided, OrthoVR-RF − RF):")
    for task, t in tests.items():
        lines.append(f"- {task}: mean Δ={t['mean_delta']:.2f}, p={t['p']:.4g}")
    (OUT / "orthovr.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved results/orthovr.json", flush=True)


if __name__ == "__main__":
    main()
