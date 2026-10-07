"""Kennedy (1993) Simulator Sickness Questionnaire scoring."""

from __future__ import annotations

import numpy as np

NAUSEA_ITEMS = (1, 6, 7, 8, 9, 15, 16)
OCULO_ITEMS = (2, 3, 4, 5, 9, 10, 11)
DISORIENT_ITEMS = (5, 8, 10, 11, 12, 13, 14)
W_N, W_O, W_D, W_TS = 9.54, 7.58, 13.92, 3.74


def parse_item_string(raw: str) -> np.ndarray:
    vals = np.array([float(x) for x in str(raw).split() if x.strip()], dtype=np.float64)
    if vals.size != 16:
        raise ValueError(f"Expected 16 SSQ items, got {vals.size}: {raw!r}")
    if vals.min() >= 1.0:
        vals = vals - 1.0
    return np.clip(vals, 0.0, 3.0)


def score_ssq(items: np.ndarray) -> dict[str, float]:
    items = np.asarray(items, dtype=np.float64).reshape(16)
    nausea = float(items[[i - 1 for i in NAUSEA_ITEMS]].sum() * W_N)
    oculo = float(items[[i - 1 for i in OCULO_ITEMS]].sum() * W_O)
    dis = float(items[[i - 1 for i in DISORIENT_ITEMS]].sum() * W_D)
    return {
        "nausea": nausea,
        "oculo": oculo,
        "disorient": dis,
        "total": float((nausea + oculo + dis) * W_TS),
    }


def parse_ipq(raw: str) -> float:
    vals = np.array([float(x) for x in str(raw).split() if x.strip()], dtype=np.float64)
    return float(vals.mean()) if vals.size else 0.0


def parse_tlx(raw: str) -> float:
    vals = np.array([float(x) for x in str(raw).split() if x.strip()], dtype=np.float64)
    return float(vals.mean()) if vals.size else 0.0
