"""Identified OrthoVR tests: within-clip affect, physio residual, spin injection.

SI modality split and LOVO are already in confound_diag.json / lovo.json.
This script adds the tests that make the kinematic leak a method, not only an audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from orthovr import OrthoVR, features_from_X, inject_spin, motion_load

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "ceap_protocol.npz")
X, feat, pid, vid = d["X"], d["feat"], d["pid"], d["vid"]
SEED = 42

idx_phy, idx_kin = [], []
for b in range(3):
    base = b * 24
    idx_phy += list(range(base, base + 4)) + list(range(base + 8, base + 12)) + list(range(base + 16, base + 20))
    idx_kin += list(range(base + 4, base + 8)) + list(range(base + 12, base + 16)) + list(range(base + 20, base + 24))
idx_kin += [72, 73]
idx_phy = np.array(idx_phy)
idx_kin = np.array(idx_kin)

PHY = feat[:, idx_phy]
KIN = feat[:, idx_kin]


def loso_pack(Xf, y):
    accs, f1s, subj = [], [], {}
    for s in sorted(set(pid.tolist())):
        te = pid == s
        clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
        clf.fit(Xf[~te], y[~te])
        pred = clf.predict(Xf[te])
        a = accuracy_score(y[te], pred) * 100
        accs.append(a)
        f1s.append(f1_score(y[te], pred, average="weighted", zero_division=0))
        subj[int(s)] = float(a)
    return {"acc": float(np.mean(accs)), "wf1": float(np.mean(f1s)), "subj": subj, "accs": accs}


def wilcoxon(a, b):
    av = np.array([a[k] for k in sorted(a)])
    bv = np.array([b[k] for k in sorted(b)])
    delta = bv - av
    if np.allclose(delta, 0):
        return {"mean_delta": 0.0, "p": 1.0}
    st = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return {"mean_delta": float(delta.mean()), "p": float(st.pvalue)}


def ortho_physio_features():
    """Per-fold residual physio features; kinematics discarded (identified body signal)."""
    acc = {t: [] for t in ("V", "A", "V3", "A3", "Vrel", "Arel")}
    f1 = {t: [] for t in acc}
    subj = {t: {} for t in acc}
    ymap = {
        "V": d["y_val"],
        "A": d["y_aro"],
        "V3": d["y_val3"],
        "A3": d["y_aro3"],
    }
    cont = np.load(ROOT / "data" / "ceap_va_cont.npz")
    ymap["Vrel"] = cont["y_val_rel"]
    ymap["Arel"] = cont["y_aro_rel"]
    for s in sorted(set(pid.tolist())):
        te = pid == s
        tr = ~te
        m = OrthoVR(alpha=1.0)
        m.fit(X[tr], pid[tr], vid[tr])
        Xtr = m.transform(X[tr], pid[tr], vid[tr])
        Xte = m.transform(X[te], pid[te], vid[te])
        ftr = features_from_X(Xtr)[:, idx_phy]
        fte = features_from_X(Xte)[:, idx_phy]
        print(f"  OrthoVR body subject {int(s)} coef={m.coef_norm():.3f}", flush=True)
        for t, y in ymap.items():
            clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
            clf.fit(ftr, y[tr])
            pred = clf.predict(fte)
            a = accuracy_score(y[te], pred) * 100
            acc[t].append(a)
            f1[t].append(f1_score(y[te], pred, average="weighted", zero_division=0))
            subj[t][int(s)] = float(a)
    return {
        t: {"acc": float(np.mean(acc[t])), "wf1": float(np.mean(f1[t])), "subj": subj[t]}
        for t in acc
    }


def main():
    if not (ROOT / "data" / "ceap_va_cont.npz").exists():
        raise SystemExit("run src/extract_va_cont.py first")
    cont = np.load(ROOT / "data" / "ceap_va_cont.npz")
    tasks = {
        "V": d["y_val"],
        "A": d["y_aro"],
        "V3": d["y_val3"],
        "A3": d["y_aro3"],
        "Vrel": cont["y_val_rel"],
        "Arel": cont["y_aro_rel"],
    }
    out = {"si": {}, "within_clip": {}, "injection": {}}

    print("=== SI + within-clip modality split ===", flush=True)
    for t, y in tasks.items():
        row = {
            "fused": loso_pack(feat, y),
            "physio": loso_pack(PHY, y),
            "kin": loso_pack(KIN, y),
        }
        store = {
            k: {"acc": v["acc"], "wf1": v["wf1"]}
            for k, v in row.items()
        }
        store["wilcoxon_kin_vs_fused"] = wilcoxon(row["fused"]["subj"], row["kin"]["subj"])
        store["wilcoxon_kin_vs_physio"] = wilcoxon(row["physio"]["subj"], row["kin"]["subj"])
        key = "within_clip" if t.endswith("rel") else "si"
        out[key][t] = store
        print(
            f"{t:5s} fused={row['fused']['acc']:.2f} phy={row['physio']['acc']:.2f} "
            f"kin={row['kin']['acc']:.2f} kin-fused Δ={store['wilcoxon_kin_vs_fused']['mean_delta']:.2f} "
            f"p={store['wilcoxon_kin_vs_fused']['p']:.4g}",
            flush=True,
        )
        out.setdefault("subject", {}).setdefault(t, {})
        for name in row:
            out["subject"][t][name] = row[name]["subj"]

    print("=== OrthoVR body (physio residual, no kinematics) ===", flush=True)
    out["orthovr_body"] = ortho_physio_features()
    for t, r in out["orthovr_body"].items():
        print(f"  {t:5s} acc={r['acc']:.2f} wf1={r['wf1']:.4f}", flush=True)

    print("=== spin injection gamma=1 into EDA+HR; physio-only RF ===", flush=True)
    xinj = inject_spin(X, 1.0)
    finj = features_from_X(xinj)
    phy_inj = finj[:, idx_phy]
    speed = motion_load(X)[:, :, 0].mean(1)
    for t in ("V", "A", "Vrel", "Arel"):
        y = tasks[t]
        raw = loso_pack(PHY, y)
        inj = loso_pack(phy_inj, y)
        # r(pred, speed) on last fold-average via a single global RF is wrong; use LOSO mean r
        rs_raw, rs_inj = [], []
        for s in sorted(set(pid.tolist())):
            te = pid == s
            for Xf, bag in ((PHY, rs_raw), (phy_inj, rs_inj)):
                clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
                clf.fit(Xf[~te], y[~te])
                if len(np.unique(y[~te])) < 2:
                    bag.append(0.0)
                    continue
                p = clf.predict_proba(Xf[te])
                col = 1 if p.shape[1] > 1 else 0
                if p[:, col].std() < 1e-8 or speed[te].std() < 1e-8:
                    bag.append(0.0)
                else:
                    bag.append(float(np.corrcoef(p[:, col], speed[te])[0, 1]))
        out["injection"][t] = {
            "physio_acc": raw["acc"],
            "injected_acc": inj["acc"],
            "r_speed_raw": float(np.mean(rs_raw)),
            "r_speed_inj": float(np.mean(rs_inj)),
        }
        print(out["injection"][t], flush=True)

    # OrthoVR on injected physio: should kill the injected leak
    print("=== OrthoVR on injected signals ===", flush=True)
    acc_v, r_v = [], []
    y = tasks["A"]
    for s in sorted(set(pid.tolist())):
        te = pid == s
        tr = ~te
        m = OrthoVR(alpha=1.0)
        m.fit(xinj[tr], pid[tr], vid[tr])
        ftr = features_from_X(m.transform(xinj[tr], pid[tr], vid[tr]))[:, idx_phy]
        fte = features_from_X(m.transform(xinj[te], pid[te], vid[te]))[:, idx_phy]
        clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
        clf.fit(ftr, y[tr])
        pred = clf.predict(fte)
        acc_v.append(accuracy_score(y[te], pred) * 100)
        p = clf.predict_proba(fte)
        if p.shape[1] == 2 and p[:, 1].std() > 1e-8 and speed[te].std() > 1e-8:
            r_v.append(float(np.corrcoef(p[:, 1], speed[te])[0, 1]))
    out["injection"]["A_orthovr"] = {"acc": float(np.mean(acc_v)), "r_speed": float(np.mean(r_v) if r_v else 0)}
    print(out["injection"]["A_orthovr"], flush=True)

    (ROOT / "results" / "identified.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/identified.json")


if __name__ == "__main__":
    main()
