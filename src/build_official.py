"""Rebuild CEAP official 12-channel tensors and 49 ML features.

Faithful port of Xue et al. baseline:
  1_Classifier_DataExtraction.py
  2_Classifier_FeatureExtraction.py
from github.com/cwi-dis/CEAP-360VR-Dataset

Processed_PD_Data is not in the public repo. PD_Mean = (LPD+RPD)/2;
PD_Emotion = per-video demeaned PD (phasic pupil). Both min-max
normalized per clip, as the baseline does for PD_Mean / PD_Emotion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CEAP = ROOT / "data" / "ceap"
OUT = ROOT / "data" / "ceap_official.npz"
FS = 50
WIN = 2
N_VID = 8
DUR = 60
N_DST = FS * DUR
N_WIN = DUR // WIN
WIN_N = FS * WIN


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _series(items, key):
    return np.array([float(s[key]) for s in items], dtype=np.float64)


def _interp(x, n_out):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(n_out, dtype=np.float64)
    src = np.linspace(0.0, 1.0, num=x.size)
    dst = np.linspace(0.0, 1.0, num=n_out)
    return np.interp(dst, src, x)


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    r = float(np.nanmax(x) - np.nanmin(x))
    if r < 1e-12:
        return np.zeros_like(x)
    return (x - np.nanmin(x)) / r


def _py_to_xyz(pitch_deg, yaw_deg):
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    return np.cos(p) * np.sin(y), np.cos(p) * np.cos(y), np.sin(p)


def angle_distance(lon1, lat1, lon2, lat2):
    rl1, rt1 = math.radians(lon1), math.radians(lat1)
    rl2, rt2 = math.radians(lon2), math.radians(lat2)
    dlon, dlat = rl2 - rl1, rt2 - rt1
    a = math.sin(dlat / 2) ** 2 + math.cos(rt1) * math.cos(rt2) * math.sin(dlon / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(max(0.0, a)))))


def saccade_fix_stats(pitch, yaw):
    n = len(pitch)
    vel_pre = 0.0
    start = end = 0
    fix, sacc_dur, sacc_amp, sacc_vel = [], [], [], []
    sacc_start = 0
    for sample in range(1, n):
        ang = angle_distance(yaw[sample - 1], pitch[sample - 1], yaw[sample], pitch[sample])
        vel = ang * FS
        sacc_vel.append(vel)
        acc = (vel - vel_pre) * FS
        vel_pre = vel
        if abs(vel) > 75 and abs(acc) > 200:
            dur = end - start
            if dur / FS >= 0.15:
                fix.append(dur)
                if len(fix) == 1:
                    if start != 0:
                        sacc_dur.append(start)
                        sacc_amp.append(max(sacc_vel[0:start]) if start else 0.0)
                else:
                    sacc_dur.append(start - sacc_start)
                    sacc_amp.append(max(sacc_vel[sacc_start:start]) if start > sacc_start else 0.0)
                sacc_start = end
            start = sample + 1
            end = sample + 1
            continue
        end += 1
        if sample == n - 1:
            dur = end - start
            if dur / FS >= 0.15:
                fix.append(dur)
            else:
                sacc_dur.append(end - sacc_start)
                sacc_amp.append(max(sacc_vel[sacc_start:end]) if end > sacc_start else 0.0)

    def block(xs, nstat, with_count=False):
        if not xs:
            return ([0.0] + [0.0] * nstat) if with_count else [0.0] * nstat
        a = np.asarray(xs, dtype=np.float64)
        stats = [float(a.mean()), float(a.std()), float(a.max()), float(a.min())]
        if with_count:
            return [float(len(a))] + stats
        return stats

    return np.array(block(fix, 4, True) + block(sacc_dur, 4) + block(sacc_amp, 4), dtype=np.float32)


def va_labels(v, a):
    vb = 0 if v < 5 else 1
    ab = 0 if a < 5 else 1
    v3 = 0 if v < 3 else (2 if v >= 6 else 1)
    a3 = 0 if a < 3 else (2 if a >= 6 else 1)
    if v > 5 and a > 5:
        c5 = 0
    elif v > 5 and a < 5:
        c5 = 1
    elif v < 5 and a < 5:
        c5 = 2
    elif v < 5 and a > 5:
        c5 = 3
    else:
        c5 = 4
    return vb, ab, v3, a3, c5


def pca1(x, y, z):
    m = np.stack([x, y, z], axis=1)
    m = np.nan_to_num(m)
    if np.allclose(m, m[0]):
        return np.zeros(len(m))
    return PCA(1).fit_transform(m).ravel()


def subject(pid: int):
    qpath = CEAP / "2_QuestionnaireData" / f"P{pid}_Questionnaire_Data.json"
    ann = _load(CEAP / "3_AnnotationData" / "Frame" / f"P{pid}_Annotation_FrameData.json")
    beh = _load(CEAP / "4_BehaviorData" / "Frame" / f"P{pid}_Behavior_FrameData.json")
    phy = _load(CEAP / "5_PhysioData" / "Frame" / f"P{pid}_Physio_FrameData.json")
    va_all = ann["ContinuousAnnotation_FrameData"][0]["Video_Annotation_FrameData"]
    beh_all = beh["Behavior_FrameData"][0]["Video_Behavior_FrameData"]
    phy_all = phy["Physio_FrameData"][0]["Video_Physio_FrameData"]
    sam_v = sam_a = None
    if qpath.exists():
        q = _load(qpath)["QuestionnaireData"][0]
        sam = q.get("Video_SAMRating_VideoTime_Data")
        if sam and len(sam) == 8:
            sam_v = [float(s["ValenceValue"]) for s in sam]
            sam_a = [float(s["ArousalValue"]) for s in sam]

    X, feat, yv, ya, yv3, ya3, y5, pid_l, vid_l = [], [], [], [], [], [], [], [], []
    for vid in range(N_VID):
        va = va_all[vid]["TimeStamp_Valence_Arousal"]
        valence = _interp(_series(va, "Valence"), N_DST)
        arousal = _interp(_series(va, "Arousal"), N_DST)
        eda = _interp(_series(phy_all[vid]["EDA_FrameData"], "EDA"), N_DST)
        bvp = _interp(_series(phy_all[vid]["BVP_FrameData"], "BVP"), N_DST)
        hr = _interp(_series(phy_all[vid]["HR_FrameData"], "HR"), N_DST)
        skt = _interp(_series(phy_all[vid]["SKT_FrameData"], "SKT"), N_DST)
        hm_p = _interp(_series(beh_all[vid]["HM"], "Pitch"), N_DST)
        hm_y = _interp(_series(beh_all[vid]["HM"], "Yaw"), N_DST)
        em_p = _interp(_series(beh_all[vid]["EM"], "Pitch"), N_DST)
        em_y = _interp(_series(beh_all[vid]["EM"], "Yaw"), N_DST)
        lpd = _interp(_series(beh_all[vid]["LPD"], "PD"), N_DST)
        rpd = _interp(_series(beh_all[vid]["RPD"], "PD"), N_DST)
        pd = (lpd + rpd) / 2.0
        pd_emo = pd - np.nanmean(pd)
        hx, hy, hz = _py_to_xyz(hm_p, hm_y)
        ex, ey, ez = _py_to_xyz(em_p, em_y)
        hm = _norm(pca1(hx, hy, hz))
        em = _norm(pca1(ex, ey, ez))
        pd_n = _norm(pd)
        pd_e = _norm(pd_emo)
        hr_n = _norm(hr)
        skt_n = _norm(skt)
        # DL 12-ch = sub_split columns 0-11
        ch = np.stack(
            [hm, em, _norm(hm_p), _norm(hm_y), _norm(em_p), _norm(em_y), pd_n, pd_e, eda, bvp, hr_n, skt_n],
            axis=1,
        ).astype(np.float32)
        # ML 12 series in feature-extraction order, then fixation
        ml12 = np.stack([hm, em, pd_n, pd_e, eda, bvp, hr_n, skt_n, hm_p, hm_y, em_p, em_y], axis=1)
        for s in range(N_WIN):
            sl = slice(s * WIN_N, (s + 1) * WIN_N)
            w = np.nan_to_num(ch[sl])
            v = float(np.mean(valence[sl]))
            a = float(np.mean(arousal[sl]))
            vb, ab, v3, a3, c5 = va_labels(v, a)
            arr = np.nan_to_num(ml12[sl])
            parts = []
            for c in range(arr.shape[1]):
                parts.extend([arr[:, c].mean(), arr[:, c].std(), float(np.median(arr[:, c]))])
            eye = saccade_fix_stats(em_p[sl], em_y[sl])
            f = np.concatenate([np.array(parts, dtype=np.float32), eye])
            X.append(w)
            feat.append(f.astype(np.float32))
            yv.append(vb)
            ya.append(ab)
            yv3.append(v3)
            ya3.append(a3)
            y5.append(c5)
            pid_l.append(pid)
            vid_l.append(vid + 1)
    return X, feat, yv, ya, yv3, ya3, y5, pid_l, vid_l


def main():
    bag = {k: [] for k in ("X", "feat", "y_val", "y_aro", "y_val3", "y_aro3", "y5", "pid", "vid")}
    n_ok = 0
    for pid in tqdm(range(1, 33), desc="official CEAP"):
        try:
            X, feat, yv, ya, yv3, ya3, y5, pid_l, vid_l = subject(pid)
        except OSError:
            continue
        n_ok += 1
        bag["X"].extend(X)
        bag["feat"].extend(feat)
        bag["y_val"].extend(yv)
        bag["y_aro"].extend(ya)
        bag["y_val3"].extend(yv3)
        bag["y_aro3"].extend(ya3)
        bag["y5"].extend(y5)
        bag["pid"].extend(pid_l)
        bag["vid"].extend(vid_l)
    feat = np.stack(bag["feat"])
    print("feat dim", feat.shape, "expect (7680, 49)")
    np.savez_compressed(
        OUT,
        X=np.stack(bag["X"]).astype(np.float32),
        feat=feat.astype(np.float32),
        y_val=np.array(bag["y_val"], np.int64),
        y_aro=np.array(bag["y_aro"], np.int64),
        y_val3=np.array(bag["y_val3"], np.int64),
        y_aro3=np.array(bag["y_aro3"], np.int64),
        y5=np.array(bag["y5"], np.int64),
        pid=np.array(bag["pid"], np.int64),
        vid=np.array(bag["vid"], np.int64),
    )
    print(f"saved {OUT} subjects={n_ok} N={len(bag['pid'])}")


if __name__ == "__main__":
    main()
