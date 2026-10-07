"""Capacity sensitivity of the stimulus-information ceiling.

Eq. (SIA) is the Bayes accuracy of Y given (q, T). The reported ceiling is
the nested maximum over a finite decoder class G applied to a finite-sample
clip posterior q_n. Any feasible member is a lower bound on the Bayes
ceiling, so estimated credit is an upper bound on residual decoding.

This script asks whether that bound is an artefact of one clip classifier
or one decoder. Clip posteriors: the locked depth-4 forest, multinomial
logistic, and histogram gradient boosting. Decoders: the locked scaled
logistic on the interaction design, an unrestricted forest, and boosting
on a compact (log q, T) design. Nested selection among g(q,T), g(q),
plug-in lookup, and majority is unchanged.

Outputs results/ceiling_capacity.json. Does not overwrite bottleneck.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_bottleneck as bn  # noqa: E402

EPS = bn.EPS
N_INNER = bn.N_INNER
RF_SEED = bn.RF_SEED
OUT = ROOT / "results"


def clip_clf(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=seed, n_jobs=-1
        )
    if kind == "log":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=200,
            random_state=seed, early_stopping=True,
        )
    raise ValueError(kind)


def g_clf(kind: str, seed: int):
    if kind == "log":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=80, max_depth=None, min_samples_leaf=8,
            random_state=seed, n_jobs=-1,
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.1, max_iter=120,
            random_state=seed, early_stopping=True,
        )
    raise ValueError(kind)


def design(q, t, n_t, with_t, g_kind):
    if g_kind == "log":
        return bn.bottleneck_design(q, t, n_t, with_t)
    lq = np.log(np.clip(q, EPS, 1.0))
    if not with_t:
        return lq
    scale = float(max(n_t - 1, 1))
    return np.hstack([lq, t.astype(np.float64)[:, None] / scale])


def fit_predict(kind, Dtr, ytr, Dte, seed):
    clf = g_clf(kind, seed)
    clf.fit(Dtr, ytr)
    return clf.predict(Dte)


def inner_acc(kind, D, y, pid_sub, seed):
    subs = np.unique(pid_sub)
    rng = np.random.default_rng(seed)
    n_g = 2 if kind != "log" else N_INNER
    groups = np.array_split(subs[rng.permutation(len(subs))], n_g)
    pred = np.full(len(y), -1, dtype=np.int64)
    for g in groups:
        held = np.isin(pid_sub, g)
        if held.all() or not held.any():
            continue
        pred[np.flatnonzero(held)] = fit_predict(
            kind, D[~held], y[~held], D[held], seed
        )
    ok = pred >= 0
    if not ok.any():
        return 0.0
    return accuracy_score(y[ok], pred[ok]) * 100


def cross_fitted_q(X, v, pid, tr_mask, n_clip, q_kind, rng_seed):
    subs = np.unique(pid[tr_mask])
    rng = np.random.default_rng(rng_seed)
    groups = np.array_split(subs[rng.permutation(len(subs))], N_INNER)
    q = np.zeros((tr_mask.sum(), n_clip), dtype=np.float64)
    idx_tr = np.flatnonzero(tr_mask)
    pid_tr = pid[idx_tr]
    for g in groups:
        held = np.isin(pid_tr, g)
        if held.all() or not held.any():
            continue
        clf = clip_clf(q_kind, rng_seed)
        clf.fit(X[idx_tr[~held]], v[idx_tr[~held]])
        p = clf.predict_proba(X[idx_tr[held]])
        cols = clf.classes_.astype(int)
        q[np.ix_(np.flatnonzero(held), cols)] = p
    return bn.fill_clip_posterior(q, n_clip)


def decompose_cap(X, y, v, t, pid, n_clip, n_t, label, q_kind, g_kind):
    subjects = np.unique(pid)
    n_cls = int(y.max()) + 1
    rows = {k: [] for k in ("loso", "bn_qt", "bn_q", "sia_plug", "majority",
                            "clip_id", "prior", "ceiling", "ceiling_pick")}
    for s in subjects:
        te = pid == s
        tr = ~te
        fold = int(np.where(subjects == s)[0][0])
        gmaj = int(np.bincount(y[tr], minlength=n_cls).argmax())
        rows["majority"].append(accuracy_score(y[te], np.full(te.sum(), gmaj)) * 100)

        clf_c = clip_clf(q_kind, RF_SEED)
        clf_c.fit(X[tr], v[tr])
        q_te = np.zeros((te.sum(), n_clip))
        p = clf_c.predict_proba(X[te])
        q_te[:, clf_c.classes_.astype(int)] = p
        chat = clf_c.classes_[p.argmax(1)].astype(int)
        rows["clip_id"].append(accuracy_score(v[te], chat) * 100)

        clf_y = bn.rf()
        clf_y.fit(X[tr], y[tr])
        rows["loso"].append(accuracy_score(y[te], clf_y.predict(X[te])) * 100)

        cell = np.full((n_clip, n_t), gmaj, dtype=np.int64)
        for c in range(n_clip):
            for w in range(n_t):
                m = tr & (v == c) & (t == w)
                if m.any():
                    cell[c, w] = int(np.bincount(y[m], minlength=n_cls).argmax())
        q_te = bn.fill_clip_posterior(q_te, n_clip)
        q_tr = cross_fitted_q(X, v, pid, tr, n_clip, q_kind, rng_seed=1000 + fold)
        chat = q_te.argmax(1)
        rows["sia_plug"].append(accuracy_score(y[te], cell[chat, t[te]]) * 100)
        rows["prior"].append(accuracy_score(y[te], cell[v[te], t[te]]) * 100)
        te_acc, tr_acc = {}, {}
        te_acc["majority"] = rows["majority"][-1]
        tr_acc["majority"] = accuracy_score(y[tr], np.full(tr.sum(), gmaj)) * 100
        chat_tr = q_tr.argmax(1)
        te_acc["sia_plug"] = rows["sia_plug"][-1]
        tr_acc["sia_plug"] = accuracy_score(y[tr], cell[chat_tr, t[tr]]) * 100
        for key, with_t in (("bn_qt", True), ("bn_q", False)):
            Dtr = design(q_tr, t[tr], n_t, with_t, g_kind)
            Dte = design(q_te, t[te], n_t, with_t, g_kind)
            pred_te = fit_predict(g_kind, Dtr, y[tr], Dte, RF_SEED)
            te_acc[key] = accuracy_score(y[te], pred_te) * 100
            tr_acc[key] = inner_acc(g_kind, Dtr, y[tr], pid[tr], seed=2000 + fold)
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
        f"  {label:28s} q={q_kind:3s} g={g_kind:3s}  "
        f"clip {out['clip_id'].mean():5.2f}  ceil {out['ceiling'].mean():5.2f}"
        f"  LOSO {out['loso'].mean():5.2f}"
        f"  credit {(out['loso'] - out['ceiling']).mean():+5.2f}"
        f"  picks { {s: picks.count(s) for s in 'bqpm'} }",
        flush=True,
    )
    out["_picks"] = picks
    packed = bn.pack(out)
    packed["q_model"] = q_kind
    packed["g_model"] = g_kind
    packed["label"] = label
    return packed


def ceap_bundle():
    d = np.load(ROOT / "data" / "ceap_official.npz")
    feat, pid = d["feat"], d["pid"]
    v = bn.remap(d["vid"])
    t = bn.window_index(pid, d["vid"])
    return {
        "feat": feat, "pid": pid, "v": v, "t": t,
        "n_clip": int(v.max()) + 1, "n_t": int(t.max()) + 1,
        "y": {"V": d["y_val"], "A": d["y_aro"], "C5": d["y5"]},
        "chan": {"All": bn.CEAP_ALL, "Physio": bn.CEAP_PHY, "Behavior": bn.CEAP_BEH},
    }


def case_bundle():
    d = np.load(ROOT / "data" / "case_windows.npz")
    v = bn.remap(d["vid"])
    t = d["win"].astype(np.int64)
    return {
        "feat": d["feat"], "pid": d["pid"], "v": v, "t": t,
        "n_clip": int(v.max()) + 1, "n_t": int(t.max()) + 1,
        "y": {"V": d["y_val"], "A": d["y_aro"]},
        "chan": {"All": None},
    }


def vreed_bundle():
    from run_vreed import load_tables
    d = load_tables()
    pid, vid = d["pid"], d["vid"]
    X = np.nan_to_num(d["all"])
    v = bn.remap(vid)
    t = np.zeros(len(pid), dtype=np.int64)
    val, aro = d["val"], d["aro"]
    finite = np.isfinite(val) & np.isfinite(aro)
    X, pid, v, t, val, aro = (X[finite], pid[finite], v[finite],
                              t[finite], val[finite], aro[finite])
    return {
        "feat": X, "pid": pid, "v": v, "t": t,
        "n_clip": int(v.max()) + 1, "n_t": 1,
        "y": {"V": (val >= 5).astype(np.int64), "A": (aro >= 5).astype(np.int64)},
        "chan": {"All": None},
    }


# Locked (q=rf, g=log) is already in bottleneck.json. Sweep the others.
CONFIGS = [
    ("log", "log"),
    ("hgb", "log"),
    ("rf", "rf"),
    ("rf", "hgb"),
    ("hgb", "hgb"),
]

TASKS = [
    ("vreed", "All", "V"),
    ("ceap", "All", "C5"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="CEAP All/V and VREED All/V only, two configs")
    args = ap.parse_args()
    t0 = time.time()
    bundles = {
        "ceap": ceap_bundle,
        "case": case_bundle,
        "vreed": vreed_bundle,
    }
    tasks = TASKS
    cfgs = CONFIGS
    if args.quick:
        tasks = [("ceap", "All", "V"), ("vreed", "All", "V")]
        cfgs = [("hgb", "hgb"), ("rf", "rf")]
    cache = OUT / "ceiling_capacity.json"
    out = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {
        "protocol": {
            "q_models": "rf depth-4 (locked), multinomial logistic, HGB",
            "g_models": "scaled logistic on interaction design (locked); "
                        "unrestricted RF and HGB on compact (log q, T)",
            "selection": "nested train-only among g(q,T), g(q), plug-in, majority",
            "loso": "always the locked depth-4 RF on raw sensors, so credit "
                    "isolates ceiling capacity rather than moving both sides",
        },
        "runs": {},
    }
    loaded = {}
    for corpus, chan, task in tasks:
        if corpus not in loaded:
            print(f"loading {corpus}", flush=True)
            loaded[corpus] = bundles[corpus]()
        b = loaded[corpus]
        cols = b["chan"][chan]
        X = b["feat"] if cols is None else b["feat"][:, cols]
        y = b["y"][task]
        for q_kind, g_kind in cfgs:
            key = f"{corpus}/{chan}/{task}/q={q_kind}/g={g_kind}"
            if key in out["runs"]:
                print(f"skip existing {key}", flush=True)
                continue
            label = f"{corpus}/{chan}/{task}"
            if corpus == "case" and g_kind == "log":
                print(f"skip {key} (CASE locked logistic already in bottleneck.json)",
                      flush=True)
                continue
            packed = decompose_cap(
                X, y, b["v"], b["t"], b["pid"], b["n_clip"], b["n_t"],
                label, q_kind, g_kind,
            )
            out["runs"][key] = packed
            OUT.mkdir(exist_ok=True)
            cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["elapsed_s"] = time.time() - t0
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {cache} in {out['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
