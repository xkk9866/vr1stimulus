"""Rebuild CEAP features with the derivative statistics the corpus paper describes.

The released feature extractor computes mean/std/median of the twelve raw
channels only. The corpus paper's text additionally describes the first and
second differentials of the physiological series. Our modality ablation
disagrees with the published one, and a physiological block without
derivative features is the most likely explanation, so we build both
feature sets and report the ablation under each:

  released artefact : 49 features  (exactly what the released code emits)
  paper-described   : 121 features (adds d1/d2 mean/std/median per channel)

Layout of the 121-d vector
  0  -  35   mean/std/median of the 12 raw channels   (as released)
  36 -  48   fixation / saccade statistics            (as released)
  49 -  84   mean/std/median of the first differences
  85 - 120   mean/std/median of the second differences

Channel order (from the released extractor) is
  0 hm  1 em  2 pd_mean  3 pd_phasic  4 eda  5 bvp  6 hr  7 skt
  8 hm_pitch  9 hm_yaw  10 em_pitch  11 em_yaw
so channels 4-7 are physiological and 0-3, 8-11 are behavioural.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from tqdm import tqdm

from build_official import (
    CEAP,
    FS,
    N_DST,
    N_VID,
    N_WIN,
    WIN_N,
    _interp,
    _load,
    _norm,
    _py_to_xyz,
    pca1,
    saccade_fix_stats,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ceap_deriv.npz"

# index blocks into the 121-d vector
PHY_CH = [4, 5, 6, 7]
BEH_CH = [0, 1, 2, 3, 8, 9, 10, 11]


def _stats3(x: np.ndarray) -> list[float]:
    return [float(x.mean()), float(x.std()), float(np.median(x))]


def _chan_block(arr: np.ndarray, order: int) -> list[float]:
    """mean/std/median per channel of the `order`-th discrete difference."""
    out: list[float] = []
    for c in range(arr.shape[1]):
        s = arr[:, c]
        for _ in range(order):
            s = np.diff(s)
        if s.size == 0:
            out.extend([0.0, 0.0, 0.0])
        else:
            out.extend(_stats3(s * (FS ** order)))
    return out


def subject_ext(pid: int):
    beh = _load(CEAP / "4_BehaviorData" / "Frame" / f"P{pid}_Behavior_FrameData.json")
    phy = _load(CEAP / "5_PhysioData" / "Frame" / f"P{pid}_Physio_FrameData.json")
    beh_all = beh["Behavior_FrameData"][0]["Video_Behavior_FrameData"]
    phy_all = phy["Physio_FrameData"][0]["Video_Physio_FrameData"]

    feats = []
    for vid in range(N_VID):
        def ser(block, key):
            return _interp(
                np.array([float(s[key]) for s in phy_all[vid][block]], dtype=np.float64), N_DST
            )

        eda = ser("EDA_FrameData", "EDA")
        bvp = ser("BVP_FrameData", "BVP")
        hr = ser("HR_FrameData", "HR")
        skt = ser("SKT_FrameData", "SKT")

        def bser(block, key):
            return _interp(
                np.array([float(s[key]) for s in beh_all[vid][block]], dtype=np.float64), N_DST
            )

        hm_p = bser("HM", "Pitch")
        hm_y = bser("HM", "Yaw")
        em_p = bser("EM", "Pitch")
        em_y = bser("EM", "Yaw")
        lpd = bser("LPD", "PD")
        rpd = bser("RPD", "PD")
        pd = (lpd + rpd) / 2.0
        pd_emo = pd - np.nanmean(pd)
        hx, hy, hz = _py_to_xyz(hm_p, hm_y)
        ex, ey, ez = _py_to_xyz(em_p, em_y)
        ml12 = np.stack(
            [
                _norm(pca1(hx, hy, hz)),
                _norm(pca1(ex, ey, ez)),
                _norm(pd),
                _norm(pd_emo),
                eda,
                bvp,
                _norm(hr),
                _norm(skt),
                hm_p,
                hm_y,
                em_p,
                em_y,
            ],
            axis=1,
        )
        for s in range(N_WIN):
            sl = slice(s * WIN_N, (s + 1) * WIN_N)
            arr = np.nan_to_num(ml12[sl])
            base: list[float] = []
            for c in range(arr.shape[1]):
                base.extend(_stats3(arr[:, c]))
            eye = saccade_fix_stats(em_p[sl], em_y[sl])
            f = (
                base
                + list(np.asarray(eye, dtype=np.float64))
                + _chan_block(arr, 1)
                + _chan_block(arr, 2)
            )
            feats.append(np.asarray(f, dtype=np.float32))
    return feats


def main() -> None:
    off = np.load(ROOT / "data" / "ceap_official.npz")
    ref = off["feat"]
    bag = []
    for pid in tqdm(range(1, 33), desc="derivative features"):
        try:
            bag.extend(subject_ext(pid))
        except OSError:
            continue
    feat = np.stack(bag)
    print("feat dim", feat.shape, "expect (7680, 121)")

    # the first 49 columns must reproduce the released extractor exactly
    d = np.abs(feat[:, :49] - ref).max()
    print(f"max abs deviation from released 49-d features: {d:.3e}")
    assert d < 1e-3, "base features do not reproduce the released pipeline"

    np.savez_compressed(OUT, feat=feat.astype(np.float32))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
