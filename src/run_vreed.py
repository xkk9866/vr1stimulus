"""Replicate the three core findings of this paper on VREED.

VREED (Tabbaa et al., IMWUT 2021) is the natural second corpus: 34
participants, 12 360-degree clips, eye tracking plus ECG and GSR, and
the same fixed-stimulus evaluation template. The released extracted
features cover the 26 complete subjects 101-117 and 119-127 (312
clip-level windows). Labels used by the original paper are the designed
quadrant of the clip (Quad_Cat), which is a property of the stimulus;
we also evaluate the self-reported post-exposure valence and arousal.

Core tests, all leave-one-subject-out:
  1. Sensor-free stimulus prior vs RF on designed quadrant and on
     binarised self-report.
  2. Modality ablation (eye / ECG / GSR) and 12-way clip identification.
  3. Deviation from the crowd on continuous self-report, plus
     leave-one-video-out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
VREED = ROOT / "data" / "vreed"
OUT = ROOT / "results"
SEED = 80

# 26 complete subjects used in the released feature tables
SUBJECTS = [i for i in range(101, 128) if i != 118]

# pickle trial index -> Num_Code, identified by duration + quadrant
# (JNG and BNS are both 180 s / quadrant 1; they occupy fixed slots 0 and 4)
TRIAL_TO_VID = {
    0: 5,   # JNG
    1: 3,   # BRZ
    2: 2,   # DST
    3: 7,   # EXR
    4: 4,   # BNS
    5: 12,  # PRS
    6: 11,  # BOT
    7: 10,  # RFS
    8: 1,   # RPW
    9: 6,   # RST
    10: 9,  # TNT
    11: 8,  # ZMZ
}


def rank_biserial(delta: np.ndarray) -> float:
    nz = delta[delta != 0]
    if nz.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(nz))
    return float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())


def paired(a, b) -> dict:
    d = np.asarray(b, float) - np.asarray(a, float)
    if np.allclose(d, 0):
        return {"delta": 0.0, "p": 1.0, "p_perm": 1.0, "rb": 0.0, "wins": 0, "n": len(d)}
    p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    rng = np.random.default_rng(SEED)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(20000, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return {
        "delta": float(d.mean()),
        "p": p,
        "p_perm": float(((null >= obs).sum() + 1) / (len(null) + 1)),
        "rb": rank_biserial(d),
        "wins": int((d > 0).sum()),
        "n": int(len(d)),
    }


def load_tables():
    eye = pd.read_csv(
        VREED / "04 Eye Tracking Data" / "02 Eye Tracking Data (Features Extracted)" / "EyeTracking_FeaturesExtracted.csv"
    )
    ecg = pd.read_csv(
        VREED / "05 ECG-GSR Data" / "02 ECG-GSR (Features Extracted)" / "ECG_FeaturesExtracted.csv"
    )
    gsr = pd.read_csv(
        VREED / "05 ECG-GSR Data" / "02 ECG-GSR (Features Extracted)" / "GSR_FeaturesExtracted.csv"
    )
    post = pd.read_excel(
        VREED / "03 Self-Reported Questionnaires" / "02 Post Exposure Ratings.xlsx"
    )
    assert len(eye) == 312 and len(ecg) == 312 and len(gsr) == 312
    n = 312
    pid = np.repeat(SUBJECTS, 12)
    trial = np.tile(np.arange(12), len(SUBJECTS))
    vid = np.array([TRIAL_TO_VID[int(t)] for t in trial], dtype=np.int64)
    quad = eye["Quad_Cat"].to_numpy(dtype=np.int64)
    # sanity: designed quadrant repeats every subject
    assert np.all(quad.reshape(-1, 12) == quad[:12]), "feature rows are not in pickle trial order"

    rating = {}
    for _, r in post.iterrows():
        rating[(int(r["ID"]), int(r["Num_Code"]))] = (
            float(r["POST_Valence"]),
            float(r["POST_Arousal"]),
            int(r["Quad_Cat"]),
        )
    val = np.array([rating[(int(p), int(v))][0] for p, v in zip(pid, vid)])
    aro = np.array([rating[(int(p), int(v))][1] for p, v in zip(pid, vid)])
    quad_from_rating = np.array([rating[(int(p), int(v))][2] for p, v in zip(pid, vid)])
    # designed quadrant in the feature table must match the ratings join
    n_mismatch = int((quad != quad_from_rating).sum())
    n_nan = int((~np.isfinite(val) | ~np.isfinite(aro)).sum())

    eye_x = eye.drop(columns=["Quad_Cat"]).to_numpy(dtype=np.float64)
    ecg_x = ecg.drop(columns=["Quad_Cat"]).to_numpy(dtype=np.float64)
    gsr_x = gsr.drop(columns=["Quad_Cat"]).to_numpy(dtype=np.float64)
    n_feat_nan = {
        "eye": int((~np.isfinite(eye_x)).sum()),
        "ecg": int((~np.isfinite(ecg_x)).sum()),
        "gsr": int((~np.isfinite(gsr_x)).sum()),
    }
    print(
        f"vid-join quadrant mismatches: {n_mismatch}/{n}  "
        f"SAM NaNs: {n_nan}/{n}  "
        f"feat NaNs eye/ecg/gsr="
        f"{n_feat_nan['eye']}/{n_feat_nan['ecg']}/{n_feat_nan['gsr']}"
    )
    for arr in (eye_x, ecg_x, gsr_x):
        arr[~np.isfinite(arr)] = 0.0

    return {
        "pid": pid,
        "vid": vid,
        "quad": quad,
        "val": val,
        "aro": aro,
        "eye": eye_x,
        "ecg": ecg_x,
        "gsr": gsr_x,
        "all": np.hstack([eye_x, ecg_x, gsr_x]),
        "n_mismatch": n_mismatch,
        "n_nan": n_nan,
        "n_feat_nan": n_feat_nan,
    }


def swap_jng_bns(vid: np.ndarray) -> np.ndarray:
    """Swap Num_Code 5 (JNG) and 4 (BNS); both 180 s / quadrant 1."""
    out = np.asarray(vid, dtype=np.int64).copy()
    m5, m4 = out == 5, out == 4
    out[m5] = 4
    out[m4] = 5
    return out


def prior_loso(y, pid, vid):
    n_cls = int(y.max()) + 1
    acc, f1, per = [], [], []
    for s in np.unique(pid):
        te, tr = pid == s, pid != s
        gmaj = int(np.bincount(y[tr], minlength=n_cls).argmax())
        pmap = {}
        for v in np.unique(vid[tr]):
            pmap[int(v)] = int(np.bincount(y[tr][vid[tr] == v], minlength=n_cls).argmax())
        pred = np.array([pmap.get(int(v), gmaj) for v in vid[te]])
        a = accuracy_score(y[te], pred) * 100
        acc.append(a)
        per.append(float(a))
        f1.append(f1_score(y[te], pred, average="weighted", zero_division=0))
    return float(np.mean(acc)), float(np.mean(f1)), per


def rf_loso(X, y, pid):
    acc, f1, per = [], [], []
    for s in np.unique(pid):
        te, tr = pid == s, pid != s
        clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
        clf.fit(np.nan_to_num(X[tr]), y[tr])
        pred = clf.predict(np.nan_to_num(X[te]))
        a = accuracy_score(y[te], pred) * 100
        acc.append(a)
        per.append(float(a))
        f1.append(f1_score(y[te], pred, average="weighted", zero_division=0))
    return float(np.mean(acc)), float(np.mean(f1)), per


def majority_loso(y, pid):
    acc, per = [], []
    for s in np.unique(pid):
        te, tr = pid == s, pid != s
        maj = int(np.bincount(y[tr], minlength=int(y.max()) + 1).argmax())
        a = accuracy_score(y[te], np.full(te.sum(), maj)) * 100
        acc.append(a)
        per.append(float(a))
    return float(np.mean(acc)), per


def lovo(X, y, vid):
    acc = []
    for v in np.unique(vid):
        te, tr = vid == v, vid != v
        maj = int(np.bincount(y[tr], minlength=int(y.max()) + 1).argmax())
        clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
        clf.fit(np.nan_to_num(X[tr]), y[tr])
        pred = clf.predict(np.nan_to_num(X[te]))
        acc.append(
            {
                "vid": int(v),
                "rf": float(accuracy_score(y[te], pred) * 100),
                "maj": float(accuracy_score(y[te], np.full(te.sum(), maj)) * 100),
            }
        )
    return acc


def decompose(val, aro, pid, vid):
    """Subject vs stimulus vs residual on the clip-level self-report."""
    out = {}
    for name, y in (("valence", val), ("arousal", aro)):
        # balanced? 26 x 12
        subs = np.unique(pid)
        vids = np.unique(vid)
        cube = np.zeros((len(subs), len(vids)))
        smap = {int(s): i for i, s in enumerate(subs)}
        vmap = {int(v): i for i, v in enumerate(vids)}
        for i in range(len(y)):
            cube[smap[int(pid[i])], vmap[int(vid[i])]] = y[i]
        mu = cube.mean()
        a = cube.mean(1) - mu
        b = cube.mean(0) - mu
        r = cube - mu - a[:, None] - b[None, :]
        tot = float(((cube - mu) ** 2).sum())
        out[name] = {
            "subject_pct": 100.0 * float((a ** 2).sum() * cube.shape[1]) / tot,
            "stimulus_pct": 100.0 * float((b ** 2).sum() * cube.shape[0]) / tot,
            "residual_pct": 100.0 * float((r ** 2).sum()) / tot,
        }
    return out


def main():
    d = load_tables()
    pid, vid = d["pid"], d["vid"]
    # designed 4-class quadrant (what the VREED paper classifies)
    yq = d["quad"]
    finite = np.isfinite(d["val"]) & np.isfinite(d["aro"])
    if not finite.all():
        print(f"dropping {int((~finite).sum())} rows with missing SAM from binary tasks")
    yv = np.zeros(len(d["val"]), dtype=np.int64)
    ya = np.zeros(len(d["aro"]), dtype=np.int64)
    yv[finite] = (d["val"][finite] >= 5).astype(np.int64)
    ya[finite] = (d["aro"][finite] >= 5).astype(np.int64)

    feats = {
        "all": d["all"],
        "eye": d["eye"],
        "ecg": d["ecg"],
        "gsr": d["gsr"],
        "phys": np.hstack([d["ecg"], d["gsr"]]),
    }

    res = {
        "n_subjects": int(len(np.unique(pid))),
        "n_clips": int(len(np.unique(vid))),
        "n_windows": int(len(pid)),
        "vid_join_mismatches": d["n_mismatch"],
        "n_sam_nan": int(d["n_nan"]),
        "n_feat_nan": d["n_feat_nan"],
        "variance": decompose(d["val"], d["aro"], pid, vid),
        "tasks": {},
    }

    tasks = {
        "quad4": yq,
        "valence_bin": yv,
        "arousal_bin": ya,
    }
    task_mask = {
        "quad4": np.ones(len(pid), dtype=bool),
        "valence_bin": finite,
        "arousal_bin": finite,
    }

    for tname, y in tasks.items():
        mask = task_mask[tname]
        pid_t, vid_t, y_t = pid[mask], vid[mask], y[mask]
        feats_t = {k: arr[mask] for k, arr in feats.items()}
        print(f"\n=== {tname}  n_cls={int(y_t.max())+1}  balance={np.bincount(y_t)} ===")
        bag = {"rows": {}, "tests": {}, "per_fold": {}}
        pa, pf, pper = prior_loso(y_t, pid_t, vid_t)
        bag["rows"]["prior(v)"] = {"acc": pa, "wf1": pf}
        bag["per_fold"]["prior(v)"] = pper
        print(f"  prior(v)          {pa:6.2f}")

        ma, mper = majority_loso(y_t, pid_t)
        bag["rows"]["majority"] = {"acc": ma}
        bag["per_fold"]["majority"] = mper
        print(f"  majority          {ma:6.2f}")

        for fname, X in feats_t.items():
            a, f, per = rf_loso(X, y_t, pid_t)
            bag["rows"][f"RF({fname})"] = {"acc": a, "wf1": f}
            bag["per_fold"][f"RF({fname})"] = per
            print(f"  RF({fname:4s})        {a:6.2f}")

        best = max(
            [k for k in bag["rows"] if k.startswith("RF")],
            key=lambda k: bag["rows"][k]["acc"],
        )
        bag["best_sensor"] = best
        bag["tests"]["prior vs best"] = paired(bag["per_fold"][best], pper)
        bag["tests"]["prior vs majority"] = paired(mper, pper)
        t = bag["tests"]["prior vs best"]
        print(
            f"  prior vs {best}: delta={t['delta']:+.2f} p={t['p']:.2e} "
            f"p_perm={t['p_perm']:.2e} rb={t['rb']:.3f} wins={t['wins']}/{t['n']}"
        )

        # clip identification from each modality (only meaningful once)
        if tname == "quad4":
            yvid = vid_t.copy()
            uniq = {int(v): i for i, v in enumerate(np.unique(yvid))}
            yvid = np.array([uniq[int(v)] for v in yvid])
            bag["clip_id"] = {}
            for fname, X in feats_t.items():
                a, _, _ = rf_loso(X, yvid, pid_t)
                bag["clip_id"][fname] = a
                print(f"  clip-id RF({fname:4s}) {a:6.2f}  (chance {100/12:.2f})")
            bag["lovo"] = {
                fname: lovo(X, y_t, vid_t)
                for fname, X in (("all", feats_t["all"]), ("eye", feats_t["eye"]),
                                 ("phys", feats_t["phys"]))
            }

        res["tasks"][tname] = bag

    # deviation: sign of self-report minus training-crowd mean for that clip
    print("\n=== deviation from crowd (self-report) ===")
    res["deviation"] = {}
    for name, ycont in (("valence", d["val"]), ("arousal", d["aro"])):
        per_rf, per_maj = [], []
        for s in np.unique(pid):
            te, tr = pid == s, pid != s
            cmap = {int(v): float(ycont[tr][vid[tr] == v].mean()) for v in np.unique(vid[tr])}
            g = float(ycont[tr].mean())
            y_tr = (ycont[tr] > np.array([cmap.get(int(v), g) for v in vid[tr]])).astype(np.int64)
            y_te = (ycont[te] > np.array([cmap.get(int(v), g) for v in vid[te]])).astype(np.int64)
            maj = int(np.bincount(y_tr, minlength=2).argmax())
            clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
            clf.fit(np.nan_to_num(d["all"][tr]), y_tr)
            per_rf.append(accuracy_score(y_te, clf.predict(np.nan_to_num(d["all"][te]))) * 100)
            per_maj.append(accuracy_score(y_te, np.full(te.sum(), maj)) * 100)
        res["deviation"][name] = {
            "RF_all": float(np.mean(per_rf)),
            "majority": float(np.mean(per_maj)),
            "test": paired(per_maj, per_rf),
        }
        print(
            f"  {name}: RF {np.mean(per_rf):.2f}  maj {np.mean(per_maj):.2f}  "
            f"delta={res['deviation'][name]['test']['delta']:+.2f} p={res['deviation'][name]['test']['p']:.3f}"
        )

        # continuous residual regression
        pred = np.zeros_like(ycont)
        for s in np.unique(pid):
            te, tr = pid == s, pid != s
            cmap = {int(v): float(ycont[tr][vid[tr] == v].mean()) for v in np.unique(vid[tr])}
            g = float(ycont[tr].mean())
            d_tr = ycont[tr] - np.array([cmap.get(int(v), g) for v in vid[tr]])
            d_te = ycont[te] - np.array([cmap.get(int(v), g) for v in vid[te]])
            reg = RandomForestRegressor(max_depth=4, random_state=SEED, n_jobs=-1)
            reg.fit(np.nan_to_num(d["all"][tr]), d_tr)
            pred[te] = reg.predict(np.nan_to_num(d["all"][te]))
        # rebuild target with LOSO crowd means for correlation
        target = np.zeros_like(ycont)
        for s in np.unique(pid):
            te, tr = pid == s, pid != s
            cmap = {int(v): float(ycont[tr][vid[tr] == v].mean()) for v in np.unique(vid[tr])}
            g = float(ycont[tr].mean())
            target[te] = ycont[te] - np.array([cmap.get(int(v), g) for v in vid[te]])
        r = float(np.corrcoef(pred, target)[0, 1])
        res["deviation"][name]["continuous_r"] = r
        print(f"  {name} continuous deviation r={r:.3f}")

    OUT.mkdir(exist_ok=True)
    (OUT / "vreed.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nsaved results/vreed.json")


if __name__ == "__main__":
    main()
