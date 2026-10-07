"""Variance decomposition, agreement ceiling, and effect sizes.

Answers three questions the accuracy tables cannot:
  1. How much of a momentary rating is determined by the stimulus rather
     than by the individual?
  2. How well do participants agree with each other, i.e. what is the
     ceiling for any model of the shared component?
  3. What are the effect sizes behind the Wilcoxon tests we report?

Balanced design: 32 subjects x 8 clips x 30 windows = 7680 rows, so the
additive decomposition below is exact and its components are orthogonal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
N_WIN = 30


def decompose(y: np.ndarray, sid: np.ndarray, cell: np.ndarray) -> dict:
    """y = mu + a_subject + b_cell + residual on a balanced design."""
    mu = y.mean()
    subs = np.unique(sid)
    cells = np.unique(cell)
    a = np.zeros(len(y))
    for s in subs:
        m = sid == s
        a[m] = y[m].mean() - mu
    b = np.zeros(len(y))
    for c in cells:
        m = cell == c
        b[m] = y[m].mean() - mu
    resid = y - mu - a - b
    tot = y.var()
    return {
        "total_var": float(tot),
        "subject_frac": float(a.var() / tot),
        "stimulus_frac": float(b.var() / tot),
        "residual_frac": float(resid.var() / tot),
    }


def clip_vs_window(y: np.ndarray, vid: np.ndarray, cell: np.ndarray) -> dict:
    """Split the stimulus term into a clip part and a within-clip time part."""
    mu = y.mean()
    c_cell = np.array([y[cell == c].mean() for c in np.unique(cell)])
    cell_of = {c: i for i, c in enumerate(np.unique(cell))}
    idx = np.array([cell_of[c] for c in cell])
    cellmean = c_cell[idx]
    clipmean = np.zeros(len(y))
    for v in np.unique(vid):
        m = vid == v
        clipmean[m] = y[m].mean()
    between_clip = (clipmean - mu).var()
    within_clip = (cellmean - clipmean).var()
    return {
        "between_clip_var": float(between_clip),
        "within_clip_time_var": float(within_clip),
        "between_clip_frac": float(between_clip / y.var()),
        "within_clip_time_frac": float(within_clip / y.var()),
    }


def agreement(y: np.ndarray, sid: np.ndarray, cell: np.ndarray) -> dict:
    """Mean pairwise correlation between participants' 240-point traces, and
    the leave-one-out correlation of each participant against the mean of
    the others (the practical ceiling for a shared-component model)."""
    subs = np.unique(sid)
    cells = np.unique(cell)
    pos = {c: i for i, c in enumerate(cells)}
    M = np.full((len(subs), len(cells)), np.nan)
    for i, s in enumerate(subs):
        m = sid == s
        for val, c in zip(y[m], cell[m]):
            M[i, pos[c]] = val

    pair = []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            pair.append(np.corrcoef(M[i], M[j])[0, 1])
    loo = []
    for i in range(len(subs)):
        others = np.delete(M, i, axis=0).mean(0)
        loo.append(np.corrcoef(M[i], others)[0, 1])
    return {
        "mean_pairwise_r": float(np.mean(pair)),
        "sd_pairwise_r": float(np.std(pair)),
        "mean_loo_r": float(np.mean(loo)),
        "sd_loo_r": float(np.std(loo)),
    }


def label_entropy(y: np.ndarray, vid: np.ndarray, win: np.ndarray) -> dict:
    """How much of the label's uncertainty the stimulus alone removes."""

    def H(v):
        p = np.bincount(v, minlength=int(v.max()) + 1).astype(float)
        p = p[p > 0] / p.sum()
        return float(-(p * np.log2(p)).sum())

    h0 = H(y)
    for key, name in ((vid, "v"), (vid * 100 + win, "vt")):
        pass
    out = {"H_y": h0}
    for keys, name in ((vid, "H_y_given_v"), (vid * 100 + win, "H_y_given_vt")):
        tot = 0.0
        for k in np.unique(keys):
            m = keys == k
            tot += m.mean() * H(y[m])
        out[name] = float(tot)
    out["info_from_v"] = h0 - out["H_y_given_v"]
    out["info_from_vt"] = h0 - out["H_y_given_vt"]
    out["frac_explained_vt"] = out["info_from_vt"] / h0
    return out


def rank_biserial(delta: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the effect size for Wilcoxon."""
    nz = delta[delta != 0]
    if len(nz) == 0:
        return 0.0
    r = stats.rankdata(np.abs(nz))
    pos = r[nz > 0].sum()
    neg = r[nz < 0].sum()
    return float((pos - neg) / r.sum())


def main():
    d = np.load(ROOT / "data" / "ceap_official.npz")
    cont = np.load(ROOT / "data" / "ceap_va_cont.npz")
    pid, vid = d["pid"], d["vid"]
    win = np.tile(np.arange(N_WIN), len(pid) // N_WIN)
    cell = vid * 100 + win

    res = {}
    for name, y in (("valence", cont["y_val_c"]), ("arousal", cont["y_aro_c"])):
        y = y.astype(np.float64)
        res[name] = {
            "decomposition": decompose(y, pid, cell),
            "stimulus_split": clip_vs_window(y, vid, cell),
            "agreement": agreement(y, pid, cell),
        }
        dec = res[name]["decomposition"]
        spl = res[name]["stimulus_split"]
        agr = res[name]["agreement"]
        print(f"--- {name} (continuous rating) ---")
        print(f"  stimulus (clip x window) {dec['stimulus_frac']*100:5.1f}% of variance")
        print(f"  subject main effect      {dec['subject_frac']*100:5.1f}%")
        print(f"  idiosyncratic residual   {dec['residual_frac']*100:5.1f}%")
        print(f"  of which between-clip    {spl['between_clip_frac']*100:5.1f}%"
              f"  within-clip time {spl['within_clip_time_frac']*100:5.1f}%")
        print(f"  mean pairwise r={agr['mean_pairwise_r']:.3f}"
              f"  leave-one-out r={agr['mean_loo_r']:.3f}")

    print("\n--- label entropy (discrete tasks) ---")
    for task, key in (("V", "y_val"), ("A", "y_aro"), ("V3", "y_val3"),
                      ("A3", "y_aro3"), ("C5", "y5")):
        e = label_entropy(d[key], vid, win)
        res.setdefault("entropy", {})[task] = e
        print(f"  {task:3s} H(y)={e['H_y']:.3f} bits  "
              f"H(y|v,t)={e['H_y_given_vt']:.3f}  "
              f"stimulus removes {e['frac_explained_vt']*100:.1f}%")

    # effect sizes for the headline comparisons
    cons = json.loads((OUT / "consensus.json").read_text(encoding="utf-8"))
    eff = {}
    print("\n--- effect sizes, prior(v,t) vs RF(all) ---")
    for task in ("V", "A", "V3", "A3", "C5"):
        a = cons["subj_acc"]["RF(all)"][task]
        b = cons["subj_acc"]["Prior(v,t)"][task]
        keys = sorted(set(a) & set(b))
        delta = np.array([b[k] - a[k] for k in keys])
        st = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        rb = rank_biserial(delta)
        boot = np.array([
            np.mean(np.random.default_rng(i).choice(delta, len(delta), replace=True))
            for i in range(10000)
        ])
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        wins = int((delta > 0).sum())
        eff[task] = {
            "delta": float(delta.mean()), "p": float(st.pvalue),
            "rank_biserial": rb, "ci95": ci, "wins": wins, "n": len(delta),
        }
        print(f"  {task:3s} d={delta.mean():+6.2f} pts  CI95=[{ci[0]:+.2f},{ci[1]:+.2f}]"
              f"  r_rb={rb:+.3f}  p={st.pvalue:.2e}  wins {wins}/{len(delta)}")
    res["effect_sizes"] = eff

    OUT.mkdir(exist_ok=True)
    (OUT / "variance.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nsaved results/variance.json")


if __name__ == "__main__":
    main()
