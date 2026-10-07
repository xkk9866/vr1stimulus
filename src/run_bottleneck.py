"""Stimulus-information bottleneck: an identified decomposition of
clip-mixed leave-one-subject-out accuracy.

The plug-in shortcut lookup m(Chat, T) is not a valid ceiling: when Chat is
near chance it addresses the wrong cell and falls BELOW the majority class,
so LOSO - SIA can be positive while LOSO itself is at or under the floor.
A rational shortcut-exploiting predictor would never do that; it would fall
back on the marginal.

We therefore replace it by the best decoder of everything the sensors know
about the stimulus, and nothing else:

    S --> q = p(V | S)  -->  g(q, T) --> Y

q is cross-fitted (inner folds over training subjects), so the decoder never
sees its own fold. g is StandardScaler plus multinomial logistic on
[log q, onehot(T), log q (x) onehot(T)], a class that contains the constant
predictor (hence >= majority) and can represent the exact cell lookup
(hence >= plug-in SIA). The ceiling A_bn is the test accuracy of the class member selected on
training data only (fitted decoder with T, without T, plug-in lookup,
majority). Picking the winner from test-fold accuracies, whether once from
means or per fold, uses the same labels as the credit and is not identified.

We report A_bn both with and without T, because handing the decoder the
elapsed window is a deployment fact (a system knows the timeline of content
it is itself playing) but makes the ceiling stronger.

Also computes:
  * cluster bootstrap CI over subjects and a TOST equivalence test on the
    sensor credit, with the smallest margin at which equivalence holds;
  * the doubly-disjoint split, holding out subject s AND clip v jointly, so
    that no window of the test participant and no window of the test clip is
    in training.

Outputs results/bottleneck.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
RF_SEED = 80
N_INNER = 4
BOOT = 10000
BOOT_SEED = 7
EPS = 1e-6

CEAP_PHY = np.arange(12, 24)
CEAP_BEH = np.r_[0:12, 24:49]
CEAP_ALL = np.arange(49)


def rf(seed: int = RF_SEED) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=seed, n_jobs=-1
    )


def onehot(idx: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((len(idx), n), dtype=np.float64)
    m[np.arange(len(idx)), idx] = 1.0
    return m


def bottleneck_design(q: np.ndarray, t: np.ndarray, n_t: int,
                      with_t: bool) -> np.ndarray:
    """[log q, onehot(T), log q (x) onehot(T)] or [log q] alone."""
    lq = np.log(np.clip(q, EPS, 1.0))
    if not with_t:
        return lq
    ht = onehot(t, n_t)
    inter = (lq[:, :, None] * ht[:, None, :]).reshape(len(q), -1)
    return np.hstack([lq, ht, inter])


def fill_clip_posterior(q: np.ndarray, n_clip: int) -> np.ndarray:
    """Rows left all-zero (skipped inner group) become the uniform prior."""
    q = np.asarray(q, dtype=np.float64).copy()
    miss = q.sum(axis=1) < 0.5
    if miss.any():
        q[miss] = 1.0 / n_clip
    return q


def cross_fitted_clip_posterior(X, v, pid, tr_mask, n_clip, rng_seed=0):
    """Out-of-fold p(V|S) for training rows; grouped by participant."""
    subs = np.unique(pid[tr_mask])
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(len(subs))
    groups = np.array_split(subs[order], N_INNER)
    q = np.zeros((tr_mask.sum(), n_clip), dtype=np.float64)
    idx_tr = np.flatnonzero(tr_mask)
    pid_tr = pid[idx_tr]
    for g in groups:
        held = np.isin(pid_tr, g)
        if held.all() or not held.any():
            continue
        clf = rf()
        clf.fit(X[idx_tr[~held]], v[idx_tr[~held]])
        p = clf.predict_proba(X[idx_tr[held]])
        cols = clf.classes_.astype(int)
        q[np.ix_(np.flatnonzero(held), cols)] = p
    return fill_clip_posterior(q, n_clip)


def inner_acc_bottleneck(D, y, pid_sub, seed: int) -> float:
    """Participant-grouped OOF accuracy of the scaled logistic.

    In-sample training accuracy of g always beats majority and the plug-in,
    so nested selection would collapse to bn(q,T) and the ceiling could fall
    below a class member (majority) that uses no extra parameters.
    """
    subs = np.unique(pid_sub)
    rng = np.random.default_rng(seed)
    groups = np.array_split(subs[rng.permutation(len(subs))], N_INNER)
    pred = np.full(len(y), -1, dtype=np.int64)
    for g in groups:
        held = np.isin(pid_sub, g)
        if held.all() or not held.any():
            continue
        pred_te, _, _ = decode_bottleneck(D[~held], y[~held], D[held])
        pred[np.flatnonzero(held)] = pred_te
    ok = pred >= 0
    if not ok.any():
        return 0.0
    return accuracy_score(y[ok], pred[ok]) * 100


def decode_bottleneck(Dtr, ytr, Dte):
    """Train-only StandardScaler, then multinomial logistic.

    Unscaled [log q, onehot(T), interaction] mixes log-probabilities with
    0/1 indicators. CASE has ~890 such columns; without scaling the
    fitted decoder can undershoot the majority class, which is a class
    member and so cannot be a valid plug-in for the supremum.
    """
    scaler = StandardScaler()
    ztr = scaler.fit_transform(Dtr)
    zte = scaler.transform(Dte)
    g = LogisticRegression(max_iter=5000, C=1.0, random_state=RF_SEED)
    g.fit(ztr, ytr)
    nit = int(np.max(np.atleast_1d(g.n_iter_)))
    return g.predict(zte), g.predict(ztr), nit


def decompose(X, y, v, t, pid, n_clip, n_t, label: str):
    """LOSO decomposition. Returns per-fold arrays."""
    subjects = np.unique(pid)
    n_cls = int(y.max()) + 1
    rows = {k: [] for k in ("loso", "bn_qt", "bn_q", "sia_plug", "majority",
                            "clip_id", "prior", "ceiling", "ceiling_pick")}
    for s in subjects:
        te = pid == s
        tr = ~te
        gmaj = int(np.bincount(y[tr], minlength=n_cls).argmax())
        rows["majority"].append(accuracy_score(y[te], np.full(te.sum(), gmaj)) * 100)

        clf_c = rf()
        clf_c.fit(X[tr], v[tr])
        q_te = np.zeros((te.sum(), n_clip))
        p = clf_c.predict_proba(X[te])
        q_te[:, clf_c.classes_.astype(int)] = p
        chat = clf_c.classes_[p.argmax(1)].astype(int)
        rows["clip_id"].append(accuracy_score(v[te], chat) * 100)

        clf_y = rf()
        clf_y.fit(X[tr], y[tr])
        rows["loso"].append(accuracy_score(y[te], clf_y.predict(X[te])) * 100)

        # plug-in cell lookup, at true and at predicted clip
        cell = np.full((n_clip, n_t), gmaj, dtype=np.int64)
        for c in range(n_clip):
            for w in range(n_t):
                m = tr & (v == c) & (t == w)
                if m.any():
                    cell[c, w] = int(np.bincount(y[m], minlength=n_cls).argmax())
        q_te = fill_clip_posterior(q_te, n_clip)
        q_tr = cross_fitted_clip_posterior(
            X, v, pid, tr, n_clip, rng_seed=1000 + int(np.where(subjects == s)[0][0])
        )
        chat = q_te.argmax(1)
        rows["sia_plug"].append(accuracy_score(y[te], cell[chat, t[te]]) * 100)
        rows["prior"].append(accuracy_score(y[te], cell[v[te], t[te]]) * 100)
        te_acc = {}
        tr_acc = {}
        te_acc["majority"] = rows["majority"][-1]
        tr_acc["majority"] = accuracy_score(y[tr], np.full(tr.sum(), gmaj)) * 100
        chat_tr = q_tr.argmax(1)
        te_acc["sia_plug"] = rows["sia_plug"][-1]
        tr_acc["sia_plug"] = accuracy_score(y[tr], cell[chat_tr, t[tr]]) * 100
        for key, with_t in (("bn_qt", True), ("bn_q", False)):
            Dtr = bottleneck_design(q_tr, t[tr], n_t, with_t)
            Dte = bottleneck_design(q_te, t[te], n_t, with_t)
            pred_te, pred_tr, nit = decode_bottleneck(Dtr, y[tr], Dte)
            if nit >= 5000:
                print(f"    warn logistic max_iter {label} subj={s} {key}",
                      flush=True)
            te_acc[key] = accuracy_score(y[te], pred_te) * 100
            tr_acc[key] = inner_acc_bottleneck(Dtr, y[tr], pid[tr], seed=2000 + int(np.where(subjects == s)[0][0]))
            rows[key].append(te_acc[key])
        pick = max(tr_acc, key=tr_acc.get)
        rows["ceiling"].append(te_acc[pick])
        rows["ceiling_pick"].append(
            {"majority": "m", "sia_plug": "p", "bn_q": "q", "bn_qt": "b"}[pick]
        )

    out = {k: np.asarray(vv, dtype=np.float64) for k, vv in rows.items()
           if k != "ceiling_pick"}
    picks = rows["ceiling_pick"]
    print(
        f"  {label:22s} clip {out['clip_id'].mean():5.2f}  maj {out['majority'].mean():5.2f}"
        f"  plug {out['sia_plug'].mean():5.2f}  bn(q) {out['bn_q'].mean():5.2f}"
        f"  bn(q,T) {out['bn_qt'].mean():5.2f}  ceil {out['ceiling'].mean():5.2f}"
        f"  LOSO {out['loso'].mean():5.2f}"
        f"  credit {(out['loso'] - out['ceiling']).mean():+5.2f}"
        f"  picks { {s: picks.count(s) for s in 'bqpm'} }",
        flush=True,
    )
    out["_picks"] = picks
    return out


def tost(d: np.ndarray, margin: float) -> float:
    """Two one-sided t-tests; returns max of the two p-values."""
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    df = n - 1
    t_low = (d.mean() + margin) / se
    t_high = (d.mean() - margin) / se
    p_low = stats.t.sf(t_low, df)
    p_high = stats.t.cdf(t_high, df)
    return float(max(p_low, p_high))


def smallest_equivalence_margin(d: np.ndarray, alpha=0.05) -> float:
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if tost(d, mid) < alpha:
            hi = mid
        else:
            lo = mid
    return hi


def inference(d: np.ndarray) -> dict:
    """Cluster bootstrap over independent units + Wilcoxon + TOST."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    if np.allclose(d, 0) or np.count_nonzero(~np.isclose(d, 0)) == 0:
        p_w = 1.0
    else:
        p_w = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    return {
        "mean": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "wilcoxon_p": p_w,
        "tost_p_2pt": tost(d, 2.0),
        "equiv_margin_95": smallest_equivalence_margin(d),
        "n_folds": int(n),
    }


CAND_KEYS = ("bn_qt", "bn_q", "sia_plug", "majority")
CAND_NAMES = {
    "bn_qt": "bottleneck (q,T)",
    "bn_q": "bottleneck (q)",
    "sia_plug": "plug-in lookup",
    "majority": "marginal",
}


def pack(out: dict) -> dict:
    picks = out.get("_picks", [])
    numeric = {k: v for k, v in out.items() if k != "_picks"}
    res = {k: {"acc": float(v.mean()), "per_fold": [float(x) for x in v]}
           for k, v in numeric.items()}
    src_map = {"b": "bottleneck (q,T)", "q": "bottleneck (q)",
               "p": "plug-in lookup", "m": "marginal"}
    counts = {src_map[k]: int(picks.count(k)) for k in src_map}
    src = max(counts, key=counts.get) if picks else "bottleneck (q,T)"
    res["ceiling_acc"] = float(out["ceiling"].mean())
    res["ceiling_source"] = src
    res["ceiling_rule"] = (
        "train-only argmax among bn(q,T), bn(q), plug-in, majority; "
        "test accuracy of the winner"
    )
    res["ceiling_win_counts"] = counts
    res["credit_selected"] = inference(out["loso"] - out["ceiling"])
    res["credit_qt"] = inference(out["loso"] - out["bn_qt"])
    res["credit_q"] = inference(out["loso"] - out["bn_q"])
    res["vs_majority"] = inference(out["loso"] - out["majority"])
    res["bn_vs_majority"] = inference(out["bn_qt"] - out["majority"])
    res["plug_vs_majority"] = inference(out["sia_plug"] - out["majority"])
    return res


def doubly_disjoint(X, y, v, pid, label: str) -> dict:
    """Hold out subject s and clip v jointly: train on s'!=s and v'!=v.

    Fold accuracies are not iid: every clip-fold of a subject shares the
    same leave-one-subject training pool. Inference is clustered at the
    subject: one mean delta per participant, then bootstrap those.
    """
    n_cls = int(y.max()) + 1
    per_s_acc, per_s_maj = defaultdict(list), defaultdict(list)
    for s in np.unique(pid):
        for c in np.unique(v):
            te = (pid == s) & (v == c)
            if not te.any():
                continue
            tr = (pid != s) & (v != c)
            gmaj = int(np.bincount(y[tr], minlength=n_cls).argmax())
            maj = accuracy_score(y[te], np.full(te.sum(), gmaj)) * 100
            clf = rf()
            clf.fit(X[tr], y[tr])
            acc = accuracy_score(y[te], clf.predict(X[te])) * 100
            per_s_acc[int(s)].append(acc)
            per_s_maj[int(s)].append(maj)
    fold_a = np.array([x for xs in per_s_acc.values() for x in xs], np.float64)
    fold_m = np.array([x for xs in per_s_maj.values() for x in xs], np.float64)
    subj_d = np.array(
        [float(np.mean(per_s_acc[s]) - np.mean(per_s_maj[s])) for s in per_s_acc],
        dtype=np.float64,
    )
    print(
        f"  {label:22s} doubly-disjoint {fold_a.mean():5.2f}  maj {fold_m.mean():5.2f}"
        f"  delta {subj_d.mean():+5.2f}  ({len(fold_a)} folds, "
        f"{len(subj_d)} subjects)",
        flush=True,
    )
    return {
        "acc": float(fold_a.mean()),
        "majority": float(fold_m.mean()),
        "n_folds": int(len(fold_a)),
        "n_subjects": int(len(subj_d)),
        "delta": inference(subj_d),
        "delta_fold_iid": inference(fold_a - fold_m),
    }


def window_index(pid, vid):
    win = np.zeros(len(pid), dtype=np.int64)
    for p in np.unique(pid):
        for v in np.unique(vid[pid == p]):
            idx = np.flatnonzero((pid == p) & (vid == v))
            win[idx] = np.arange(len(idx))
    return win


def remap(a):
    uniq = np.array(sorted(np.unique(a).tolist()))
    inv = {int(u): i for i, u in enumerate(uniq)}
    return np.array([inv[int(x)] for x in a], dtype=np.int64)


def run_ceap(out: dict, full: bool):
    d = np.load(ROOT / "data" / "ceap_official.npz")
    feat, pid = d["feat"], d["pid"]
    v = remap(d["vid"])
    t = window_index(pid, d["vid"])
    n_clip, n_t = int(v.max()) + 1, int(t.max()) + 1
    tasks = {"V": d["y_val"], "A": d["y_aro"]}
    if full:
        tasks.update({"V3": d["y_val3"], "A3": d["y_aro3"], "C5": d["y5"]})
    chans = {"All": CEAP_ALL} if not full else {
        "All": CEAP_ALL, "Behavior": CEAP_BEH, "Physio": CEAP_PHY}
    print("=== CEAP-360VR ===", flush=True)
    bag = {}
    for cname, cols in chans.items():
        X = feat[:, cols]
        for tname, y in tasks.items():
            if cname != "All" and tname not in ("V", "A"):
                continue
            r = decompose(X, y, v, t, pid, n_clip, n_t, f"{cname}/{tname}")
            bag[f"{cname}/{tname}"] = pack(r)
    dd = {}
    for tname, y in tasks.items():
        dd[tname] = doubly_disjoint(feat, y, v, pid, f"All/{tname}")
    out["ceap"] = {"decomposition": bag, "doubly_disjoint": dd,
                   "n_clip": n_clip, "n_windows": n_t}


def run_case(out: dict):
    d = np.load(ROOT / "data" / "case_windows.npz")
    feat, pid = d["feat"], d["pid"]
    v = remap(d["vid"])
    t = d["win"].astype(np.int64)
    n_clip, n_t = int(v.max()) + 1, int(t.max()) + 1
    print("=== CASE ===", flush=True)
    bag = {}
    for tname, y in (("V", d["y_val"]), ("A", d["y_aro"])):
        r = decompose(feat, y, v, t, pid, n_clip, n_t, f"All/{tname}")
        bag[f"All/{tname}"] = pack(r)
    dd = {tname: doubly_disjoint(feat, y, v, pid, f"All/{tname}")
          for tname, y in (("V", d["y_val"]), ("A", d["y_aro"]))}
    out["case"] = {"decomposition": bag, "doubly_disjoint": dd,
                   "n_clip": n_clip, "n_windows": n_t}


def run_vreed(out: dict):
    """Self-report valence/arousal, not designed quadrant.

    The designed quadrant is a deterministic function of the clip, so a
    credit number on that task is not a measurement of affective decoding.
    Self-reports vary within clip (stimulus share 59% valence, 36% arousal
    in the 26-subject official tables) and are the identified analogue of
    the CEAP/CASE binary tasks. Features and the vid join are the same as
    src/run_vreed.py (0 quadrant mismatches on that join).
    """
    from run_vreed import load_tables, swap_jng_bns
    d = load_tables()
    pid, vid = d["pid"], d["vid"]
    X = np.nan_to_num(d["all"])
    v = remap(vid)
    t = np.zeros(len(pid), dtype=np.int64)
    n_clip, n_t = int(v.max()) + 1, 1
    val, aro = d["val"], d["aro"]
    finite = np.isfinite(val) & np.isfinite(aro)
    if not finite.all():
        print(f"  dropping {int((~finite).sum())} rows with missing SAM",
              flush=True)
        X, pid, v, t, val, aro = (X[finite], pid[finite], v[finite],
                                  t[finite], val[finite], aro[finite])
    yv = (val >= 5).astype(np.int64)
    ya = (aro >= 5).astype(np.int64)
    print(f"=== VREED (self-report, {len(yv)} trials, "
          f"{len(np.unique(pid))} subjects) ===", flush=True)
    bag = {}
    for tname, y in (("V", yv), ("A", ya)):
        r = decompose(X, y, v, t, pid, n_clip, n_t, f"All/{tname}")
        bag[f"All/{tname}"] = pack(r)
    dd = {tname: doubly_disjoint(X, y, v, pid, f"All/{tname}")
          for tname, y in (("V", yv), ("A", ya))}
    vid_kept = d["vid"] if finite.all() else d["vid"][finite]
    v_swap = remap(swap_jng_bns(vid_kept))
    bag_swap = {}
    print("=== VREED JNG/BNS swap ===", flush=True)
    for tname, y in (("V", yv), ("A", ya)):
        r = decompose(X, y, v_swap, t, pid, n_clip, n_t, f"swap/{tname}")
        bag_swap[f"All/{tname}"] = pack(r)
    out["vreed"] = {
        "decomposition": bag,
        "doubly_disjoint": dd,
        "n_clip": n_clip,
        "n_trials": int(len(yv)),
        "label": "self-report SAM split at 5; designed quadrant excluded "
                 "because it is a deterministic function of the clip",
        "vid_join_mismatches": int(d["n_mismatch"]),
        "n_sam_nan": int((~np.isfinite(d["val"]) | ~np.isfinite(d["aro"])).sum()),
        "n_feat_nan": d.get("n_feat_nan"),
        "jng_bns_swap": bag_swap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", nargs="?", default="all",
                    choices=("all", "ceap", "ceap_full", "case", "vreed"))
    args = ap.parse_args()
    t0 = time.time()
    cache = OUT / "bottleneck.json"
    out = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    out["protocol"] = {
        "bottleneck": "S -> q=p(V|S) -> g(q,T) -> Y; g is StandardScaler + "
                      "logistic on [log q, onehot(T), log q x onehot(T)], "
                      "cross-fitted q, inner seed 1000+outer_fold",
        "why": "contains the constant predictor (>= majority) and can represent "
               "the exact cell lookup (>= plug-in SIA)",
        "ceiling": "nested: among bn(q,T), bn(q), plug-in, majority, pick by "
                   "training-set scores (OOF acc for g, OOF-q plug-in, "
                   "majority); credit uses that winner's test accuracy",
        "sensor_credit": "A_loso - A_ceiling",
        "doubly_disjoint": "train on subject != s and clip != v; CIs clustered "
                           "by subject (clip-folds of one subject are not iid)",
        "bootstrap": "cluster bootstrap over independent units, 10000 draws; "
                     "resamples unit differences, does not refit models",
        "tost": "two one-sided t-tests; equiv_margin_95 is the smallest "
                "margin at which equivalence holds at alpha=0.05",
    }
    if args.part in ("all", "ceap", "ceap_full"):
        run_ceap(out, full=args.part in ("all", "ceap_full"))
    if args.part in ("all", "case"):
        run_case(out)
    if args.part in ("all", "vreed"):
        run_vreed(out)
    out["elapsed_s"] = time.time() - t0
    OUT.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {cache} in {out['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
