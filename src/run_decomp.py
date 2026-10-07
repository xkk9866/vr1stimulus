"""Exact orthogonal variance decomposition of the continuous ratings.

Supersedes the three-term model in run_variance.py, which lumped the
subject-by-stimulus interaction into the residual and so overstated how
much of the rating is unstructured idiosyncrasy.

The design is fully balanced (32 subjects x 8 clips x 30 windows, one
observation per cell), so the five-term model

    y_svt = mu + a_s + b_v + d_vt + (ab)_sv + r_svt

is exact and its terms are mutually orthogonal, i.e. the sums of squares
add to the total. Here

    a_s     subject main effect        (this person rates high overall)
    b_v     clip main effect           (this clip is pleasant)
    d_vt    within-clip time course    (this moment of the clip is tense)
    (ab)_sv subject x clip interaction (this person reacted atypically to
                                        this clip -- a systematic individual
                                        difference, not noise)
    r_svt   subject x time-within-clip residual

b_v + d_vt reproduces the stimulus term of the three-term model, and
a_s is unchanged, so the earlier headline numbers are recovered exactly;
what changes is that the old residual is now split into a systematic
interaction component and a genuine remainder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
N_SUB, N_VID, N_WIN = 32, 8, 30


def cube(y: np.ndarray, pid: np.ndarray, vid: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Reshape the flat window list into a balanced (subject, clip, window) cube."""
    subs = np.sort(np.unique(pid))
    vids = np.sort(np.unique(vid))
    out = np.full((len(subs), len(vids), N_WIN), np.nan)
    for i, s in enumerate(subs):
        for j, v in enumerate(vids):
            m = (pid == s) & (vid == v)
            out[i, j, win[m]] = y[m]
    assert not np.isnan(out).any(), "design is not balanced"
    return out


def decompose(y: np.ndarray) -> dict:
    """Five-term orthogonal decomposition of a balanced (S, V, T) cube."""
    mu = y.mean()
    a = y.mean(axis=(1, 2)) - mu                       # (S,)
    b = y.mean(axis=(0, 2)) - mu                       # (V,)
    d = y.mean(axis=0) - y.mean(axis=(0, 2))[:, None]  # (V, T)
    sv = y.mean(axis=2) - a[:, None] - b[None, :] - mu  # (S, V)
    r = (
        y
        - mu
        - a[:, None, None]
        - b[None, :, None]
        - d[None, :, :]
        - sv[:, :, None]
    )
    n_s, n_v, n_t = y.shape
    ss = {
        "subject": float((a ** 2).sum() * n_v * n_t),
        "clip": float((b ** 2).sum() * n_s * n_t),
        "within_clip_time": float((d ** 2).sum() * n_s),
        "subject_x_clip": float((sv ** 2).sum() * n_t),
        "residual": float((r ** 2).sum()),
    }
    total = float(((y - mu) ** 2).sum())
    recon = sum(ss.values())
    assert abs(recon - total) / total < 1e-8, f"not orthogonal: {recon} vs {total}"
    pct = {k: 100.0 * v / total for k, v in ss.items()}
    pct["stimulus_total"] = pct["clip"] + pct["within_clip_time"]
    pct["individual_total"] = pct["subject"] + pct["subject_x_clip"]
    return {"ss": ss, "total": total, "pct": pct}


def reliability(y: np.ndarray) -> dict:
    """Inter-participant consistency of the 240-point rating traces.

    This is a reliability estimate, not a performance ceiling: it says how
    much of one participant's trace is shared with the group, which is the
    quantity the stimulus prior exploits.
    """
    traces = y.reshape(y.shape[0], -1)
    n = len(traces)
    pair = []
    for i in range(n):
        for j in range(i + 1, n):
            pair.append(np.corrcoef(traces[i], traces[j])[0, 1])
    loo = []
    for i in range(n):
        other = np.delete(traces, i, axis=0).mean(axis=0)
        loo.append(np.corrcoef(traces[i], other)[0, 1])
    pair = np.array(pair)
    loo = np.array(loo)
    # Spearman-Brown prediction of the reliability of the 31-subject mean
    rbar = float(pair.mean())
    sb = (n - 1) * rbar / (1 + (n - 2) * rbar)
    return {
        "pairwise_mean": rbar,
        "pairwise_sd": float(pair.std()),
        "loo_mean": float(loo.mean()),
        "loo_sd": float(loo.std()),
        "spearman_brown_group_mean": float(sb),
    }


def deviation_structure(y: np.ndarray) -> dict:
    """How much of the individual deviation from the crowd is systematic.

    d_svt = y_svt - mean_s' y_s'vt decomposes into a part that is constant
    within a (subject, clip) pair -- a stable individual difference a model
    could in principle learn -- and a part that varies window to window.
    """
    c = y.mean(axis=0)                     # (V, T) crowd trace
    dev = y - c[None, :, :]                # (S, V, T)
    per_sv = dev.mean(axis=2)              # (S, V) systematic part
    resid = dev - per_sv[:, :, None]
    tot = float((dev ** 2).sum())
    return {
        "systematic_pct": 100.0 * float((per_sv ** 2).sum() * y.shape[2]) / tot,
        "window_varying_pct": 100.0 * float((resid ** 2).sum()) / tot,
        "dev_sd": float(dev.std()),
    }


def main() -> None:
    off = np.load(ROOT / "data" / "ceap_official.npz")
    cont = np.load(ROOT / "data" / "ceap_va_cont.npz")
    pid, vid = off["pid"], off["vid"]
    win = np.tile(np.arange(N_WIN), len(pid) // N_WIN)

    res = {}
    for name, key in (("valence", "y_val_c"), ("arousal", "y_aro_c")):
        y = cube(cont[key], pid, vid, win)
        dec = decompose(y)
        res[name] = {
            "decomposition": dec,
            "reliability": reliability(y),
            "deviation": deviation_structure(y),
        }
        p = dec["pct"]
        print(f"\n=== {name} ===")
        print(f"  clip main            {p['clip']:6.2f}%")
        print(f"  within-clip time     {p['within_clip_time']:6.2f}%")
        print(f"  -> stimulus total    {p['stimulus_total']:6.2f}%")
        print(f"  subject main         {p['subject']:6.2f}%")
        print(f"  subject x clip       {p['subject_x_clip']:6.2f}%")
        print(f"  -> individual total  {p['individual_total']:6.2f}%")
        print(f"  residual (S x T|V)   {p['residual']:6.2f}%")
        rel = res[name]["reliability"]
        print(f"  pairwise r {rel['pairwise_mean']:.3f}  loo r {rel['loo_mean']:.3f}")
        dv = res[name]["deviation"]
        print(f"  deviation: systematic {dv['systematic_pct']:.1f}%  window-varying {dv['window_varying_pct']:.1f}%")

    OUT.mkdir(exist_ok=True)
    (OUT / "decomp.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nsaved results/decomp.json")


if __name__ == "__main__":
    main()
