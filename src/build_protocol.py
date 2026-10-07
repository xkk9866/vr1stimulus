"""Build CEAP windows matching Xue et al. TMM 2021: 2 s, 50 Hz, non-overlap.

Paper sample size: 32 subjects × 8 videos × 30 segments = 7680.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ssq import parse_ipq, parse_item_string, parse_tlx, score_ssq
from twoclock import two_clock, window_features

ROOT = Path(__file__).resolve().parents[1]
CEAP = ROOT / "data" / "ceap"
OUT = ROOT / "data" / "ceap_protocol.npz"
FS = 50
WIN = 2
N_VID = 8
DUR = 60
N_DST = FS * DUR  # 3000
N_WIN = DUR // WIN  # 30
WIN_N = FS * WIN  # 100


def _series(frame_list, key):
    return np.array([float(s[key]) for s in frame_list], dtype=np.float32)


def _resample(x, n_out):
    if x.size == 0:
        return np.zeros(n_out, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=x.size)
    dst = np.linspace(0.0, 1.0, num=n_out)
    return np.interp(dst, src, x).astype(np.float32)


def _group_ssq(q):
    s0 = score_ssq(parse_item_string(q["SSQ 0"]))["total"]
    s1 = score_ssq(parse_item_string(q["SSQ 1"]))["total"]
    s2 = score_ssq(parse_item_string(q["SSQ 2"]))["total"]
    ipq1, ipq2 = parse_ipq(q["IPQ 1"]), parse_ipq(q["IPQ 2"])
    order = [int(x) for x in str(q["VideoOrder"]).split() if x.strip()]
    if len(order) != 8:
        order = list(range(1, 9))
    ssq = [0.0] * 8
    ipq = [0.0] * 8
    for i, vid in enumerate(order):
        idx = vid - 1
        ssq[idx] = (s1 - s0) if i < 4 else (s2 - s1)
        ipq[idx] = ipq1 if i < 4 else ipq2
    return ssq, ipq


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def subject(pid: int) -> dict | None:
    try:
        q = _load(CEAP / "2_QuestionnaireData" / f"P{pid}_Questionnaire_Data.json")["QuestionnaireData"][0]
        ann = _load(CEAP / "3_AnnotationData" / "Frame" / f"P{pid}_Annotation_FrameData.json")
        beh = _load(CEAP / "4_BehaviorData" / "Frame" / f"P{pid}_Behavior_FrameData.json")
        phy = _load(CEAP / "5_PhysioData" / "Frame" / f"P{pid}_Physio_FrameData.json")
    except OSError:
        return None
    va_all = ann["ContinuousAnnotation_FrameData"][0]["Video_Annotation_FrameData"]
    beh_all = beh["Behavior_FrameData"][0]["Video_Behavior_FrameData"]
    phy_all = phy["Physio_FrameData"][0]["Video_Physio_FrameData"]
    ssq, ipq = _group_ssq(q)
    bag = {k: [] for k in ("X", "Xfast", "feat", "feat_fast", "y_val", "y_aro", "y_val3", "y_aro3", "ssq", "ipq", "pid", "vid")}
    for vid in range(N_VID):
        va = va_all[vid]["TimeStamp_Valence_Arousal"]
        valence = _resample(_series(va, "Valence"), N_DST)
        arousal = _resample(_series(va, "Arousal"), N_DST)
        eda = _resample(_series(phy_all[vid]["EDA_FrameData"], "EDA"), N_DST)
        bvp = _resample(_series(phy_all[vid]["BVP_FrameData"], "BVP"), N_DST)
        hr = _resample(_series(phy_all[vid]["HR_FrameData"], "HR"), N_DST)
        skt = _resample(_series(phy_all[vid]["SKT_FrameData"], "SKT"), N_DST)
        hm_p = _resample(_series(beh_all[vid]["HM"], "Pitch"), N_DST)
        hm_y = _resample(_series(beh_all[vid]["HM"], "Yaw"), N_DST)
        em_p = _resample(_series(beh_all[vid]["EM"], "Pitch"), N_DST)
        em_y = _resample(_series(beh_all[vid]["EM"], "Yaw"), N_DST)
        x = np.stack([eda, bvp, hr, skt, hm_p, hm_y, em_p, em_y], axis=1)
        x = np.nan_to_num(x)
        fast, _slow = two_clock(x, tau_sec=15.0, fs=float(FS))
        for s in range(N_WIN):
            sl = slice(s * WIN_N, (s + 1) * WIN_N)
            w, wf = x[sl], fast[sl]
            v, a = float(np.mean(valence[sl])), float(np.mean(arousal[sl]))
            bag["X"].append(w)
            bag["Xfast"].append(wf)
            bag["feat"].append(window_features(w))
            bag["feat_fast"].append(window_features(wf))
            bag["y_val"].append(1 if v >= 5 else 0)
            bag["y_aro"].append(1 if a >= 5 else 0)
            bag["y_val3"].append(0 if v < 3 else (2 if v >= 6 else 1))
            bag["y_aro3"].append(0 if a < 3 else (2 if a >= 6 else 1))
            bag["ssq"].append(ssq[vid])
            bag["ipq"].append(ipq[vid])
            bag["pid"].append(pid)
            bag["vid"].append(vid + 1)
    return bag


def main() -> int:
    bag = None
    n_ok = 0
    for pid in tqdm(range(1, 33), desc="CEAP protocol"):
        part = subject(pid)
        if part is None:
            continue
        n_ok += 1
        if bag is None:
            bag = {k: list(v) for k, v in part.items()}
        else:
            for k, v in part.items():
                bag[k].extend(v)
    if not bag:
        raise SystemExit("no subjects")
    np.savez_compressed(
        OUT,
        X=np.stack(bag["X"]).astype(np.float32),
        Xfast=np.stack(bag["Xfast"]).astype(np.float32),
        feat=np.stack(bag["feat"]).astype(np.float32),
        feat_fast=np.stack(bag["feat_fast"]).astype(np.float32),
        y_val=np.array(bag["y_val"], np.int64),
        y_aro=np.array(bag["y_aro"], np.int64),
        y_val3=np.array(bag["y_val3"], np.int64),
        y_aro3=np.array(bag["y_aro3"], np.int64),
        ssq=np.array(bag["ssq"], np.float32),
        ipq=np.array(bag["ipq"], np.float32),
        pid=np.array(bag["pid"], np.int64),
        vid=np.array(bag["vid"], np.int64),
    )
    print(f"saved {OUT} subjects={n_ok} N={len(bag['pid'])} (paper 32x8x30=7680)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
