"""Build 2-second CASE windows and run the three core tests.

CASE (Sharma et al., Scientific Data 2019) is a non-immersive fixed-stimulus
corpus: 30 participants, 8 film clips, continuous joystick valence/arousal,
and eight peripheral physiological channels. There is no head or eye
tracking, so a positive replication here cannot be attributed to VR
kinematics.

Raw DAQ files have no video-ID column. We reconstruct the timeline from
the published viewing order and clip durations, keep only the eight
elicitation clips (dropping start, end, and blue inter-clip screens),
downsample physiology from 1000 Hz to 20 Hz, and cut non-overlapping
2-second windows. Features are mean/std/median of each channel and of
its first difference.
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
CASE = ROOT / "data" / "case_git"
OUT_NPZ = ROOT / "data" / "case_windows.npz"
OUT = ROOT / "results"
SEED = 80
WIN = 2.0
FS_ANN = 20.0
FS_PHY = 1000.0
DS = int(FS_PHY / FS_ANN)  # 50

VID_MAP = {
    "amusing-1": 1,
    "amusing-2": 2,
    "boring-1": 3,
    "boring-2": 4,
    "relaxed-1": 5,
    "relaxed-2": 6,
    "scary-1": 7,
    "scary-2": 8,
}
PHY_COLS = ["ecg", "bvp", "gsr", "rsp", "skt", "emg_zygo", "emg_coru", "emg_trap"]


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


def load_meta():
    dur = {}
    for line in (CASE / "metadata" / "videos_duration.txt").read_text(encoding="utf-8").splitlines()[1:]:
        parts = [p for p in line.replace('"', "").split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        dur[parts[0].strip()] = float(parts[1]) / 1000.0
    seqs = pd.read_csv(CASE / "metadata" / "seqs_order.txt", sep="\t")
    # columns seq_sub1 .. seq_sub30, rows are timeline items
    order = {}
    for s in range(1, 31):
        col = f"seq_sub{s}"
        order[s] = [str(x).strip().strip('"') for x in seqs[col].tolist()]
    return dur, order


def timeline(items, dur):
    """Return list of (vid_id, t0, t1) for elicitation clips only."""
    t = 0.0
    out = []
    for name in items:
        d = dur[name]
        if name in VID_MAP:
            out.append((VID_MAP[name], t, t + d))
        t += d
    return out, t


def stats3(x: np.ndarray) -> list[float]:
    if x.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(np.mean(x)), float(np.std(x)), float(np.median(x))]


def build():
    dur, order = load_meta()
    feats, yv, ya, yvc, yac, pid, vid, win = [], [], [], [], [], [], [], []
    n_ok = 0
    for s in range(1, 31):
        segs, total = timeline(order[s], dur)
        ann_p = CASE / "data" / "raw" / "annotations" / f"sub{s}_joystick.txt"
        phy_p = CASE / "data" / "raw" / "physiological" / f"sub{s}_DAQ.txt"
        if not ann_p.exists() or not phy_p.exists():
            print(f"skip sub{s}: missing files", flush=True)
            continue
        ann = pd.read_csv(ann_p, sep="\t", header=None, names=["t", "val", "aro"])
        # downsample physiology: every 50th row (~20 Hz)
        phy = pd.read_csv(
            phy_p,
            sep="\t",
            header=None,
            names=["t"] + PHY_COLS,
            usecols=range(9),
        )
        phy = phy.iloc[::DS].reset_index(drop=True)
        t_ann = ann["t"].to_numpy(dtype=np.float64)
        val = ann["val"].to_numpy(dtype=np.float64) / 26225.0  # [-1, 1]
        aro = ann["aro"].to_numpy(dtype=np.float64) / 26225.0
        t_phy = phy["t"].to_numpy(dtype=np.float64)
        P = phy[PHY_COLS].to_numpy(dtype=np.float64)
        n_win_s = 0
        for v, t0, t1 in segs:
            # 2 s windows inside the clip. T is the sequential index among
            # kept windows, so a quality-filter skip cannot leave a gap in T.
            k_clip = 0
            n_w = int((t1 - t0) // WIN)
            for w in range(n_w):
                a, b = t0 + w * WIN, t0 + (w + 1) * WIN
                ma = (t_ann >= a) & (t_ann < b)
                mp = (t_phy >= a) & (t_phy < b)
                if ma.sum() < 5 or mp.sum() < 5:
                    continue
                vc = float(val[ma].mean())
                ac = float(aro[ma].mean())
                block = P[mp]
                f = []
                for c in range(block.shape[1]):
                    f.extend(stats3(block[:, c]))
                    f.extend(stats3(np.diff(block[:, c])))
                feats.append(np.asarray(f, dtype=np.float32))
                yvc.append(vc)
                yac.append(ac)
                yv.append(1 if vc >= 0 else 0)
                ya.append(1 if ac >= 0 else 0)
                pid.append(s)
                vid.append(v)
                win.append(k_clip)
                k_clip += 1
                n_win_s += 1
        n_ok += 1
        print(f"sub{s:02d} windows={n_win_s} clips={len(segs)} phy_rows={len(phy)}", flush=True)

    feat = np.stack(feats)
    np.savez_compressed(
        OUT_NPZ,
        feat=feat,
        y_val=np.array(yv, np.int64),
        y_aro=np.array(ya, np.int64),
        y_val_c=np.array(yvc, np.float32),
        y_aro_c=np.array(yac, np.float32),
        pid=np.array(pid, np.int64),
        vid=np.array(vid, np.int64),
        win=np.array(win, np.int64),
    )
    print(f"saved {OUT_NPZ} subjects={n_ok} N={len(pid)} dim={feat.shape}", flush=True)
    return OUT_NPZ


def prior_loso(y, pid, vid, win=None):
    n_cls = int(y.max()) + 1
    acc, f1, per = [], [], []
    for s in np.unique(pid):
        te, tr = pid == s, pid != s
        gmaj = int(np.bincount(y[tr], minlength=n_cls).argmax())
        if win is None:
            key_tr, key_te = vid[tr], vid[te]
        else:
            key_tr = vid[tr] * 10000 + win[tr]
            key_te = vid[te] * 10000 + win[te]
        pmap = {}
        for k in np.unique(key_tr):
            pmap[int(k)] = int(np.bincount(y[tr][key_tr == k], minlength=n_cls).argmax())
        pred = np.array([pmap.get(int(k), gmaj) for k in key_te])
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


def decompose(y, pid, vid):
    subs, vids = np.unique(pid), np.unique(vid)
    # mean per (subject, clip)
    cube = np.full((len(subs), len(vids)), np.nan)
    smap = {int(s): i for i, s in enumerate(subs)}
    vmap = {int(v): i for i, v in enumerate(vids)}
    for i in range(len(y)):
        si, vi = smap[int(pid[i])], vmap[int(vid[i])]
        if np.isnan(cube[si, vi]):
            cube[si, vi] = 0.0
            n = 0
        else:
            n = 1  # we'll redo properly
    cube = np.zeros((len(subs), len(vids)))
    cnt = np.zeros_like(cube)
    for i in range(len(y)):
        si, vi = smap[int(pid[i])], vmap[int(vid[i])]
        cube[si, vi] += y[i]
        cnt[si, vi] += 1
    cube = cube / np.maximum(cnt, 1)
    mu = cube.mean()
    a = cube.mean(1) - mu
    b = cube.mean(0) - mu
    r = cube - mu - a[:, None] - b[None, :]
    tot = float(((cube - mu) ** 2).sum())
    return {
        "subject_pct": 100.0 * float((a ** 2).sum() * cube.shape[1]) / tot,
        "stimulus_pct": 100.0 * float((b ** 2).sum() * cube.shape[0]) / tot,
        "residual_pct": 100.0 * float((r ** 2).sum()) / tot,
    }


def run_tests(npz_path: Path):
    d = np.load(npz_path)
    feat, pid, vid, win = d["feat"], d["pid"], d["vid"], d["win"]
    res = {
        "n_subjects": int(len(np.unique(pid))),
        "n_clips": int(len(np.unique(vid))),
        "n_windows": int(len(pid)),
        "feat_dim": int(feat.shape[1]),
        "variance": {
            "valence": decompose(d["y_val_c"], pid, vid),
            "arousal": decompose(d["y_aro_c"], pid, vid),
        },
        "tasks": {},
    }
    print("variance", json.dumps(res["variance"]), flush=True)

    for tname, y in (("valence", d["y_val"]), ("arousal", d["y_aro"])):
        print(f"\n=== {tname} balance={np.bincount(y)} ===", flush=True)
        bag = {"rows": {}, "tests": {}, "per_fold": {}}
        pa, pf, pper = prior_loso(y, pid, vid, win=None)
        bag["rows"]["prior(v)"] = {"acc": pa, "wf1": pf}
        bag["per_fold"]["prior(v)"] = pper
        print(f"  prior(v)     {pa:6.2f}", flush=True)
        pa2, pf2, pper2 = prior_loso(y, pid, vid, win=win)
        bag["rows"]["prior(v,t)"] = {"acc": pa2, "wf1": pf2}
        bag["per_fold"]["prior(v,t)"] = pper2
        print(f"  prior(v,t)   {pa2:6.2f}", flush=True)
        ma, mper = majority_loso(y, pid)
        bag["rows"]["majority"] = {"acc": ma}
        bag["per_fold"]["majority"] = mper
        print(f"  majority     {ma:6.2f}", flush=True)
        a, f, per = rf_loso(feat, y, pid)
        bag["rows"]["RF(phys)"] = {"acc": a, "wf1": f}
        bag["per_fold"]["RF(phys)"] = per
        print(f"  RF(phys)     {a:6.2f}", flush=True)
        bag["tests"]["prior(v,t) vs RF"] = paired(per, pper2)
        bag["tests"]["prior(v) vs RF"] = paired(per, pper)
        t = bag["tests"]["prior(v,t) vs RF"]
        print(
            f"  prior(v,t) vs RF: delta={t['delta']:+.2f} p={t['p']:.2e} "
            f"p_perm={t['p_perm']:.2e} rb={t['rb']:.3f} wins={t['wins']}/{t['n']}",
            flush=True,
        )
        # clip identification
        uniq = {int(v): i for i, v in enumerate(np.unique(vid))}
        yvid = np.array([uniq[int(v)] for v in vid])
        ca, _, _ = rf_loso(feat, yvid, pid)
        bag["clip_id"] = ca
        print(f"  clip-id RF   {ca:6.2f}  (chance {100/8:.2f})", flush=True)
        res["tasks"][tname] = bag

    print("\n=== deviation ===", flush=True)
    res["deviation"] = {}
    for name, ycont in (("valence", d["y_val_c"]), ("arousal", d["y_aro_c"])):
        per_rf, per_maj = [], []
        pred = np.zeros_like(ycont)
        target = np.zeros_like(ycont)
        for s in np.unique(pid):
            te, tr = pid == s, pid != s
            key_tr = vid[tr] * 10000 + win[tr]
            key_te = vid[te] * 10000 + win[te]
            cmap, cnt = {}, {}
            for k, yc in zip(key_tr, ycont[tr]):
                cmap[int(k)] = cmap.get(int(k), 0.0) + float(yc)
                cnt[int(k)] = cnt.get(int(k), 0) + 1
            for k in cmap:
                cmap[k] /= cnt[k]
            g = float(ycont[tr].mean())
            c_tr = np.array([cmap.get(int(k), g) for k in key_tr])
            c_te = np.array([cmap.get(int(k), g) for k in key_te])
            y_tr = (ycont[tr] > c_tr).astype(np.int64)
            y_te = (ycont[te] > c_te).astype(np.int64)
            maj = int(np.bincount(y_tr, minlength=2).argmax())
            clf = RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
            clf.fit(np.nan_to_num(feat[tr]), y_tr)
            per_rf.append(accuracy_score(y_te, clf.predict(np.nan_to_num(feat[te]))) * 100)
            per_maj.append(accuracy_score(y_te, np.full(te.sum(), maj)) * 100)
            target[te] = ycont[te] - c_te
            reg = RandomForestRegressor(max_depth=4, random_state=SEED, n_jobs=-1)
            reg.fit(np.nan_to_num(feat[tr]), ycont[tr] - c_tr)
            pred[te] = reg.predict(np.nan_to_num(feat[te]))
        r = float(np.corrcoef(pred, target)[0, 1]) if target.std() > 0 else 0.0
        res["deviation"][name] = {
            "RF": float(np.mean(per_rf)),
            "majority": float(np.mean(per_maj)),
            "test": paired(per_maj, per_rf),
            "continuous_r": r,
        }
        print(
            f"  {name}: RF {np.mean(per_rf):.2f} maj {np.mean(per_maj):.2f} "
            f"delta={res['deviation'][name]['test']['delta']:+.2f} "
            f"p={res['deviation'][name]['test']['p']:.3f} r={r:.3f}",
            flush=True,
        )

    OUT.mkdir(exist_ok=True)
    (OUT / "case.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved results/case.json", flush=True)


def main():
    if not OUT_NPZ.exists():
        build()
    else:
        print(f"using existing {OUT_NPZ}", flush=True)
    run_tests(OUT_NPZ)


if __name__ == "__main__":
    main()
