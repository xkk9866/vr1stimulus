"""Continuous 2 s VA means, aligned to ceap_protocol.npz windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CEAP = ROOT / "data" / "ceap"
FS, WIN, N_VID, DUR = 50, 2, 8, 60
N_DST = FS * DUR
N_WIN = DUR // WIN
WIN_N = FS * WIN


def resample(x, n_out):
    if x.size == 0:
        return np.zeros(n_out, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=x.size)
    dst = np.linspace(0.0, 1.0, num=n_out)
    return np.interp(dst, src, x).astype(np.float32)


def main():
    proto = np.load(ROOT / "data" / "ceap_protocol.npz")
    pid_all, vid_all = proto["pid"], proto["vid"]
    val = np.zeros(len(pid_all), np.float32)
    aro = np.zeros(len(pid_all), np.float32)
    i = 0
    for pid in tqdm(range(1, 33), desc="continuous VA"):
        path = CEAP / "3_AnnotationData" / "Frame" / f"P{pid}_Annotation_FrameData.json"
        with open(path, encoding="utf-8") as f:
            ann = json.load(f)
        va_all = ann["ContinuousAnnotation_FrameData"][0]["Video_Annotation_FrameData"]
        for vid in range(N_VID):
            va = va_all[vid]["TimeStamp_Valence_Arousal"]
            v = resample(np.array([float(s["Valence"]) for s in va], np.float32), N_DST)
            a = resample(np.array([float(s["Arousal"]) for s in va], np.float32), N_DST)
            for s in range(N_WIN):
                sl = slice(s * WIN_N, (s + 1) * WIN_N)
                val[i] = float(v[sl].mean())
                aro[i] = float(a[sl].mean())
                i += 1
    assert i == len(pid_all)
    rel_v = np.zeros(len(val), np.int64)
    rel_a = np.zeros(len(aro), np.int64)
    for p in np.unique(pid_all):
        for v in np.unique(vid_all):
            m = (pid_all == p) & (vid_all == v)
            rel_v[m] = (val[m] > np.median(val[m])).astype(np.int64)
            rel_a[m] = (aro[m] > np.median(aro[m])).astype(np.int64)
    out = ROOT / "data" / "ceap_va_cont.npz"
    np.savez_compressed(out, y_val_c=val, y_aro_c=aro, y_val_rel=rel_v, y_aro_rel=rel_a)
    print(
        "saved",
        out,
        "val",
        val.mean(),
        "aro",
        aro.mean(),
        "within-clip std val",
        float(np.mean([val[(pid_all == p) & (vid_all == v)].std() for p in np.unique(pid_all) for v in np.unique(vid_all)])),
    )


if __name__ == "__main__":
    main()
