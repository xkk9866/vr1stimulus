"""Build per-trial VREED features with participant and stimulus identity preserved.

VREED (Tabbaa et al., IMWUT 2021) is the second immersive corpus we test.
The released feature tables drop the participant and video identifiers, which
makes subject-independent evaluation impossible, so we extract features
ourselves from the released pre-processed signals.

Structure of the released .dat files, established empirically:
  ECG-GSR : 13 segments per participant. Segment 0 is the rest baseline
            (label 4); segments 1-12 are the twelve videos in a fixed
            canonical order, identical for every participant. Column 0 is
            GSR in microsiemens, column 1 is zero-mean ECG, ~2000 Hz.
  Eye     : 12 segments in a fixed canonical order (a different order from
            ECG-GSR). Channels are [epoch_ms, yaw_L, pitch_L, blink_L,
            yaw_R, pitch_R, blink_R] at ~70 Hz.

Because the ordering is canonical rather than randomised, the segment index
identifies the video. We align the two modalities by ranking segment
duration within each quadrant, then assert that the quadrant labels agree.
"""

from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VR = ROOT / "data" / "vreed"
OUT = ROOT / "data" / "vreed_trials.npz"
ECG_DIR = VR / "05 ECG-GSR Data" / "01 ECG-GSR Data (Pre-Processed)"
ET_DIR = VR / "04 Eye Tracking Data" / "01 Eye Tracking Data (Pre-Processed)"
FS_ECG = 2000.0
FS_ET = 70.0
N_TRIAL = 12


def _stats(x: np.ndarray) -> list[float]:
    if x.size == 0:
        return [0.0] * 5
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.median(x)),
        float(np.min(x)),
        float(np.max(x)),
    ]


def _peaks(x: np.ndarray, refractory: int) -> np.ndarray:
    """Local maxima above one standard deviation, with a refractory gap."""
    if x.size < 3:
        return np.array([], dtype=int)
    thr = x.mean() + x.std()
    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > thr))[0] + 1
    keep: list[int] = []
    for i in cand:
        if not keep or i - keep[-1] >= refractory:
            keep.append(int(i))
    return np.asarray(keep, dtype=int)


def gsr_features(g: np.ndarray) -> list[float]:
    """Tonic level, variability and phasic peak counts."""
    d = np.diff(g) if g.size > 1 else np.zeros(1)
    pk = _peaks(g, int(0.5 * FS_ECG))
    vl = _peaks(-g, int(0.5 * FS_ECG))
    slope = float(np.polyfit(np.arange(g.size), g, 1)[0]) if g.size > 2 else 0.0
    return (
        _stats(g)
        + [float(np.var(g)), float(np.ptp(g)), slope]
        + _stats(d)
        + [float(len(pk)), float(len(vl)), float(len(pk)) / max(len(vl), 1)]
    )


def ecg_features(e: np.ndarray) -> list[float]:
    """Amplitude statistics plus heart-rate variability from R-peak intervals."""
    r = _peaks(e, int(0.3 * FS_ECG))
    if len(r) > 3:
        rr = np.diff(r) / FS_ECG * 1000.0  # ms
        drr = np.diff(rr)
        hrv = [
            float(np.mean(rr)),
            float(np.median(rr)),
            float(np.std(rr)),                                   # SDNN
            float(np.sqrt(np.mean(drr ** 2))) if drr.size else 0.0,  # RMSSD
            float(np.mean(np.abs(drr) > 50)) if drr.size else 0.0,   # pNN50
            float(np.mean(np.abs(drr) > 20)) if drr.size else 0.0,   # pNN20
            60000.0 / float(np.mean(rr)),                        # BPM
            float(np.min(rr)),
            float(np.max(rr)),
        ]
    else:
        hrv = [0.0] * 9
    return _stats(e) + [float(np.var(e)), float(np.ptp(e))] + hrv


def et_features(a: np.ndarray) -> list[float]:
    """Gaze position, gaze speed and blink statistics for both eyes."""
    out: list[float] = []
    for ch in (1, 2, 4, 5):  # yaw/pitch, left and right eye
        s = a[ch]
        out += _stats(s)
        out += _stats(np.abs(np.diff(s)) * FS_ET if s.size > 1 else np.zeros(1))
    for ch in (3, 6):  # blink flags
        b = a[ch]
        out += [float(np.mean(b)), float(np.sum(np.diff(b) > 0))]
    # combined gaze speed magnitude
    yaw = 0.5 * (a[1] + a[4])
    pit = 0.5 * (a[2] + a[5])
    if yaw.size > 1:
        spd = np.hypot(np.diff(yaw), np.diff(pit)) * FS_ET
    else:
        spd = np.zeros(1)
    out += _stats(spd) + [float(np.percentile(spd, 90)) if spd.size else 0.0]
    return out


def _rank_within_quadrant(lengths: list[int], labels: list[int]) -> list[int]:
    """Position of each trial within its quadrant when sorted by duration.

    Duration is proportional to video length in both modalities, so this
    gives a modality-independent handle on which video a segment is.
    """
    rank = [0] * len(labels)
    for q in set(labels):
        idx = [i for i, l in enumerate(labels) if l == q]
        order = sorted(idx, key=lambda i: lengths[i])
        for r, i in enumerate(order):
            rank[i] = r
    return rank


def main() -> None:
    ecg_files = {int(os.path.basename(p)[:3]): p for p in glob.glob(str(ECG_DIR / "*.dat"))}
    et_files = {int(os.path.basename(p)[:3]): p for p in glob.glob(str(ET_DIR / "*.dat"))}

    rows_phy, rows_beh, pid_l, quad_l, vid_l = [], [], [], [], []
    skipped = []
    for pid in sorted(set(ecg_files) & set(et_files)):
        de = pickle.load(open(ecg_files[pid], "rb"), encoding="latin1")
        dt = pickle.load(open(et_files[pid], "rb"), encoding="latin1")
        if len(de["Data"]) != 13 or len(dt["Data"]) != N_TRIAL:
            skipped.append(pid)
            continue
        lab_e = [int(x) for x in np.asarray(de["Labels"]).ravel()][1:13]
        lab_t = [int(x) for x in np.asarray(dt["Labels"]).ravel()]
        len_e = [np.asarray(x).shape[0] for x in de["Data"][1:13]]
        len_t = [np.asarray(x).shape[1] for x in dt["Data"]]
        rk_e = _rank_within_quadrant(len_e, lab_e)
        rk_t = _rank_within_quadrant(len_t, lab_t)

        # (quadrant, duration-rank) is the join key between the two orderings
        key_t = {(lab_t[i], rk_t[i]): i for i in range(N_TRIAL)}
        assert len(key_t) == N_TRIAL, f"ambiguous eye ordering for {pid}"

        for i in range(N_TRIAL):
            j = key_t[(lab_e[i], rk_e[i])]
            seg = np.asarray(de["Data"][i + 1])
            rows_phy.append(gsr_features(seg[:, 0]) + ecg_features(seg[:, 1]))
            rows_beh.append(et_features(np.asarray(dt["Data"][j])))
            pid_l.append(pid)
            quad_l.append(lab_e[i])
            vid_l.append((lab_e[i], rk_e[i]))

    phy = np.nan_to_num(np.asarray(rows_phy, dtype=np.float64))
    beh = np.nan_to_num(np.asarray(rows_beh, dtype=np.float64))
    pid = np.asarray(pid_l, dtype=np.int64)
    quad = np.asarray(quad_l, dtype=np.int64)
    uniq = sorted(set(vid_l))
    vid = np.asarray([uniq.index(v) for v in vid_l], dtype=np.int64)

    print(f"participants kept {len(set(pid_l))}  skipped {skipped}")
    print(f"trials {len(pid)}  phys dim {phy.shape[1]}  beh dim {beh.shape[1]}")
    print(f"distinct videos {len(uniq)}  quadrant counts {np.bincount(quad)}")
    assert len(uniq) == N_TRIAL, "video identity not recovered"
    # each video must belong to exactly one quadrant, i.e. the label is a
    # deterministic function of the stimulus
    for v in range(N_TRIAL):
        assert len(set(quad[vid == v].tolist())) == 1, "quadrant varies within a video"
    print("verified: quadrant label is a deterministic function of the video")

    np.savez_compressed(OUT, phy=phy, beh=beh, pid=pid, quad=quad, vid=vid)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
