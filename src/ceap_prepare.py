"""Build 2-second multimodal windows from CEAP-360VR Frame JSON."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ssq import parse_ipq, parse_item_string, parse_tlx, score_ssq

ROOT = Path(__file__).resolve().parents[1]
CEAP = ROOT / "data" / "ceap"
# Optional second location for an unpacked corpus copy, e.g.
#   CEAP_360VR_DIR=/path/to/CEAP-360VR-Dataset/CEAP-360VR
SPARSE = Path(os.environ.get("CEAP_360VR_DIR", str(CEAP)))
OUT = ROOT / "data" / "windows.npz"

HZ = 10
WIN_SEC = 2.0
STRIDE_SEC = 1.0
N_VID = 8
CHANNELS = ("eda", "bvp", "hr", "skt", "hm_pitch", "hm_yaw", "em_pitch", "em_yaw", "pupil")


def _find_ceap() -> Path:
    for cand in (CEAP, SPARSE):
        q = cand / "2_QuestionnaireData" / "P1_Questionnaire_Data.json"
        if q.exists():
            return cand
    raise FileNotFoundError("CEAP-360VR not found. Run src/download_ceap.py first.")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _series(frame_list: list, key: str) -> np.ndarray:
    return np.array([float(s[key]) for s in frame_list], dtype=np.float32)


def _resample(x: np.ndarray, n_out: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros(n_out, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=x.size)
    dst = np.linspace(0.0, 1.0, num=n_out)
    return np.interp(dst, src, x).astype(np.float32)


def _group_ssq(q: dict) -> tuple[list[float], list[float], list[float]]:
    s0 = score_ssq(parse_item_string(q["SSQ 0"]))["total"]
    s1 = score_ssq(parse_item_string(q["SSQ 1"]))["total"]
    s2 = score_ssq(parse_item_string(q["SSQ 2"]))["total"]
    ipq1 = parse_ipq(q["IPQ 1"])
    ipq2 = parse_ipq(q["IPQ 2"])
    tlx1 = parse_tlx(q["NASA-TLX 1"])
    tlx2 = parse_tlx(q["NASA-TLX 2"])
    order = [int(x) for x in str(q["VideoOrder"]).split() if x.strip()]
    if len(order) != 8:
        order = list(range(1, 9))
    ssq = [0.0] * 8
    ipq = [0.0] * 8
    tlx = [0.0] * 8
    for i, vid in enumerate(order):
        idx = vid - 1
        if i < 4:
            ssq[idx] = s1 - s0
            ipq[idx] = ipq1
            tlx[idx] = tlx1
        else:
            ssq[idx] = s2 - s1
            ipq[idx] = ipq2
            tlx[idx] = tlx2
    return ssq, ipq, tlx


def _subject_windows(root: Path, pid: int) -> dict[str, list]:
    q = _load_json(root / "2_QuestionnaireData" / f"P{pid}_Questionnaire_Data.json")["QuestionnaireData"][0]
    ann = _load_json(root / "3_AnnotationData" / "Frame" / f"P{pid}_Annotation_FrameData.json")
    beh = _load_json(root / "4_BehaviorData" / "Frame" / f"P{pid}_Behavior_FrameData.json")
    phy = _load_json(root / "5_PhysioData" / "Frame" / f"P{pid}_Physio_FrameData.json")
    va_all = ann["ContinuousAnnotation_FrameData"][0]["Video_Annotation_FrameData"]
    beh_all = beh["Behavior_FrameData"][0]["Video_Behavior_FrameData"]
    phy_all = phy["Physio_FrameData"][0]["Video_Physio_FrameData"]
    sam = q["Video_SAMRating_VideoTime_Data"]
    ssq, ipq, tlx = _group_ssq(q)

    out = {k: [] for k in ("X", "y_val", "y_aro", "val", "aro", "ssq", "ipq", "tlx", "pid", "vid", "hm_energy")}
    n_win = int(WIN_SEC * HZ)
    stride = int(STRIDE_SEC * HZ)

    for vid in range(N_VID):
        va = va_all[vid]["TimeStamp_Valence_Arousal"]
        n_src = len(va)
        if n_src < 20:
            continue
        duration = 60.0
        n_dst = int(duration * HZ)
        valence = _resample(_series(va, "Valence"), n_dst)
        arousal = _resample(_series(va, "Arousal"), n_dst)
        eda = _resample(_series(phy_all[vid]["EDA_FrameData"], "EDA"), n_dst)
        bvp = _resample(_series(phy_all[vid]["BVP_FrameData"], "BVP"), n_dst)
        hr = _resample(_series(phy_all[vid]["HR_FrameData"], "HR"), n_dst)
        skt = _resample(_series(phy_all[vid]["SKT_FrameData"], "SKT"), n_dst)
        hm_p = _resample(_series(beh_all[vid]["HM"], "Pitch"), n_dst)
        hm_y = _resample(_series(beh_all[vid]["HM"], "Yaw"), n_dst)
        em_p = _resample(_series(beh_all[vid]["EM"], "Pitch"), n_dst)
        em_y = _resample(_series(beh_all[vid]["EM"], "Yaw"), n_dst)
        lpd = beh_all[vid].get("LPD") or []
        rpd = beh_all[vid].get("RPD") or []
        pupil_src = lpd if lpd else rpd
        if pupil_src and isinstance(pupil_src[0], dict) and "PD" in pupil_src[0]:
            left = _resample(_series(lpd, "PD"), n_dst) if lpd else None
            right = _resample(_series(rpd, "PD"), n_dst) if rpd else left
            pupil = 0.5 * (left + right) if left is not None and right is not None else (left if left is not None else right)
        else:
            pupil = np.zeros(n_dst, dtype=np.float32)

        sig = np.stack([eda, bvp, hr, skt, hm_p, hm_y, em_p, em_y, pupil], axis=1)
        dhm = np.sqrt(np.diff(hm_p, prepend=hm_p[0]) ** 2 + np.diff(hm_y, prepend=hm_y[0]) ** 2)
        sam_v = float(sam[vid]["ValenceValue"])
        sam_a = float(sam[vid]["ArousalValue"])

        for start in range(0, n_dst - n_win + 1, stride):
            sl = slice(start, start + n_win)
            x = sig[sl]
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0)
            v = float(np.nanmean(valence[sl]))
            a = float(np.nanmean(arousal[sl]))
            out["X"].append(x)
            out["val"].append(v)
            out["aro"].append(a)
            out["y_val"].append(1 if v >= 5.0 else 0)
            out["y_aro"].append(1 if a >= 5.0 else 0)
            out["ssq"].append(ssq[vid])
            out["ipq"].append(ipq[vid])
            out["tlx"].append(tlx[vid])
            out["pid"].append(pid)
            out["vid"].append(vid + 1)
            out["hm_energy"].append(float(dhm[sl].mean()))
            _ = (sam_v, sam_a)
    return out


def main() -> int:
    root = _find_ceap()
    print(f"Using CEAP root: {root}")
    bag = {k: [] for k in ("X", "y_val", "y_aro", "val", "aro", "ssq", "ipq", "tlx", "pid", "vid", "hm_energy")}
    for pid in tqdm(range(1, 33), desc="subjects"):
        try:
            part = _subject_windows(root, pid)
        except Exception as exc:  # noqa: BLE001
            print(f"skip P{pid}: {exc}")
            continue
        for k, v in part.items():
            bag[k].extend(v)
    if not bag["X"]:
        raise RuntimeError("No windows extracted")
    X = np.stack(bag["X"]).astype(np.float32)
    # per-channel z-score using training-agnostic dataset stats is leakage;
    # we z-score later inside each CV fold. Save raw.
    np.savez_compressed(
        OUT,
        X=X,
        y_val=np.array(bag["y_val"], dtype=np.int64),
        y_aro=np.array(bag["y_aro"], dtype=np.int64),
        val=np.array(bag["val"], dtype=np.float32),
        aro=np.array(bag["aro"], dtype=np.float32),
        ssq=np.array(bag["ssq"], dtype=np.float32),
        ipq=np.array(bag["ipq"], dtype=np.float32),
        tlx=np.array(bag["tlx"], dtype=np.float32),
        pid=np.array(bag["pid"], dtype=np.int64),
        vid=np.array(bag["vid"], dtype=np.int64),
        hm_energy=np.array(bag["hm_energy"], dtype=np.float32),
        channels=np.array(CHANNELS),
    )
    print(f"saved {OUT} X={X.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
