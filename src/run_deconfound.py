"""Deconfounding protocol for fixed-stimulus affect recognition.

Four estimands, all leave-one-subject-out except A_lovo:

  A_ann  stimulus prior          Acc(C -> Y), C = (clip, window)
  A_sia  shortcut-implied acc.   Acc(m_train(Ĉ, T) -> Y)
                                 Ĉ = clip predicted from sensors; T is known
  A_loso reported accuracy       Acc(S -> Y) under clip-mixed LOSO
  A_id   identified accuracy     deviation task / clip-disjoint (computed elsewhere)

Credits (percentage points):
  StimulusCredit = A_sia  - majority
  SensorCredit   = A_loso - A_sia

A_sia is not a random embedding. It is the accuracy of the mediated path
S -> clip -> label, using the real sensors' own LOSO clip-confusion and a
training-only cell majority. Under the pure-shortcut hypothesis, a
well-specified decoder of (Ĉ, T) cannot beat A_sia, and A_sia cannot beat
A_ann.

Phantom sweep is a *calibration* of that claim: a sensor that is literally
a noisy code for (clip, window) should land on its own A_sia. Multiple
embedding seeds. Arousal sitting at the majority floor is reported as
such, not as a match to published accuracy.

Outputs results/deconfound.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PHY = np.arange(12, 24)
BEH = np.r_[0:12, 24:49]
ALL = np.arange(49)
SLICES = (("All", ALL), ("Behavior", BEH), ("Physio", PHY))
RF_SEED = 80
N_WIN = 30
N_PHANTOM_SEEDS = 5
PHANTOM_SEEDS = (0, 1, 2, 3, 42)
SIGMAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def rf(seed: int = RF_SEED) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=seed, n_jobs=-1
    )


def window_index(pid: np.ndarray, vid: np.ndarray) -> np.ndarray:
    win = np.zeros(len(pid), dtype=np.int64)
    for p in np.unique(pid):
        for v in np.unique(vid[pid == p]):
            idx = np.flatnonzero((pid == p) & (vid == v))
            win[idx] = np.arange(len(idx))
    return win


def remap(vid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.array(sorted(np.unique(vid).tolist()))
    inv = {int(u): i for i, u in enumerate(uniq)}
    return np.array([inv[int(v)] for v in vid], dtype=np.int64), uniq


def majority_table(y, v, win, tr, n_clip, n_cls) -> tuple[np.ndarray, int]:
    """Train-only majority label per (clip, window). Unseen cells get global maj."""
    gmaj = int(np.bincount(y[tr], minlength=n_cls).argmax())
    table = np.full((n_clip, N_WIN), gmaj, dtype=np.int64)
    for c in range(n_clip):
        for t in range(N_WIN):
            m = tr & (v == c) & (win == t)
            if m.any():
                table[c, t] = int(np.bincount(y[m], minlength=n_cls).argmax())
    return table, gmaj


def clip_majority_table(y, v, tr, n_clip, n_cls) -> np.ndarray:
    gmaj = int(np.bincount(y[tr], minlength=n_cls).argmax())
    table = np.full(n_clip, gmaj, dtype=np.int64)
    for c in range(n_clip):
        m = tr & (v == c)
        if m.any():
            table[c] = int(np.bincount(y[m], minlength=n_cls).argmax())
    return table


def phantom_features(v, win, n_feat, sigma, rng) -> np.ndarray:
    cells = [(int(c), int(t)) for c, t in zip(v, win)]
    uniq = sorted(set(cells))
    table = {cell: rng.standard_normal(n_feat) for cell in uniq}
    base = np.stack([table[cell] for cell in cells])
    if sigma == 0.0:
        return base
    return base + sigma * rng.standard_normal(base.shape)


def summarise(per_fold: list[float]) -> dict:
    a = np.asarray(per_fold, dtype=np.float64)
    return {
        "acc": float(a.mean()),
        "std": float(a.std(ddof=1) if len(a) > 1 else 0.0),
        "per_fold": [float(x) for x in a],
    }


def run_sia(feat, y_tasks, pid, v, win, n_clip):
    """LOS O clip probe + cell lookup. Also records A_loso of the same forest."""
    subjects = sorted(np.unique(pid).tolist())
    out = {}
    for sname, cols in SLICES:
        X = feat[:, cols]
        bag = {
            "clip_id": [],
            "tasks": {t: {"loso": [], "sia_cell": [], "sia_clip": [],
                          "ann_cell": [], "majority": []} for t in y_tasks},
        }
        for s in subjects:
            te = pid == s
            tr = ~te
            clf_c = rf()
            clf_c.fit(X[tr], v[tr])
            chat = clf_c.predict(X[te]).astype(np.int64)
            bag["clip_id"].append(accuracy_score(v[te], chat) * 100)
            t_te = win[te]
            for tname, y in y_tasks.items():
                n_cls = int(y.max()) + 1
                cell, gmaj = majority_table(y, v, win, tr, n_clip, n_cls)
                clip_tab = clip_majority_table(y, v, tr, n_clip, n_cls)
                yte = y[te]
                bag["tasks"][tname]["sia_cell"].append(
                    accuracy_score(yte, cell[chat, t_te]) * 100
                )
                bag["tasks"][tname]["sia_clip"].append(
                    accuracy_score(yte, clip_tab[chat]) * 100
                )
                bag["tasks"][tname]["ann_cell"].append(
                    accuracy_score(yte, cell[v[te], t_te]) * 100
                )
                bag["tasks"][tname]["majority"].append(
                    accuracy_score(yte, np.full(te.sum(), gmaj)) * 100
                )
                clf_y = rf()
                clf_y.fit(X[tr], y[tr])
                bag["tasks"][tname]["loso"].append(
                    accuracy_score(yte, clf_y.predict(X[te])) * 100
                )
        row = {"clip_id": summarise(bag["clip_id"]), "tasks": {}}
        for tname in y_tasks:
            cells = {k: summarise(bag["tasks"][tname][k]) for k in bag["tasks"][tname]}
            sia = cells["sia_cell"]["acc"]
            loso = cells["loso"]["acc"]
            maj = cells["majority"]["acc"]
            ann = cells["ann_cell"]["acc"]
            cells["stimulus_credit"] = sia - maj
            cells["sensor_credit"] = loso - sia
            cells["annotation_gap"] = ann - sia
            d_sia = np.asarray(bag["tasks"][tname]["sia_cell"])
            d_loso = np.asarray(bag["tasks"][tname]["loso"])
            cells["sensor_credit_per_fold"] = (d_loso - d_sia).tolist()
            cells["sensor_credit_std"] = float((d_loso - d_sia).std(ddof=1))
            row["tasks"][tname] = cells
        out[sname] = row
        print(
            f"{sname:9s} clip-id {row['clip_id']['acc']:5.2f}  "
            + "  ".join(
                f"{t} loso={row['tasks'][t]['loso']['acc']:.2f} "
                f"sia={row['tasks'][t]['sia_cell']['acc']:.2f} "
                f"Δsens={row['tasks'][t]['sensor_credit']:+.2f}"
                for t in ("V", "A")
            ),
            flush=True,
        )
    return out


def run_phantom(feat_shape, y_tasks, pid, v, win, n_clip, n_feat):
    subjects = sorted(np.unique(pid).tolist())
    by_sigma = {}
    for sigma in SIGMAS:
        seed_rows = []
        for seed in PHANTOM_SEEDS:
            rng = np.random.default_rng(seed)
            Xp = phantom_features(v, win, n_feat, sigma, rng)
            clip_accs, task_accs, sia_accs = [], {t: [] for t in y_tasks}, {t: [] for t in y_tasks}
            for s in subjects:
                te = pid == s
                tr = ~te
                clf_c = rf(seed)
                clf_c.fit(Xp[tr], v[tr])
                chat = clf_c.predict(Xp[te]).astype(np.int64)
                clip_accs.append(accuracy_score(v[te], chat) * 100)
                t_te = win[te]
                for tname, y in y_tasks.items():
                    n_cls = int(y.max()) + 1
                    cell, _ = majority_table(y, v, win, tr, n_clip, n_cls)
                    sia_accs[tname].append(
                        accuracy_score(y[te], cell[chat, t_te]) * 100
                    )
                    clf_y = rf(seed)
                    clf_y.fit(Xp[tr], y[tr])
                    task_accs[tname].append(
                        accuracy_score(y[te], clf_y.predict(Xp[te])) * 100
                    )
            seed_rows.append({
                "seed": int(seed),
                "clip_id": float(np.mean(clip_accs)),
                "tasks": {
                    t: {
                        "loso": float(np.mean(task_accs[t])),
                        "sia_cell": float(np.mean(sia_accs[t])),
                    }
                    for t in y_tasks
                },
            })
            print(
                f"  sigma={sigma:<5} seed={seed:<3} clip-id {np.mean(clip_accs):5.2f}  "
                + "  ".join(
                    f"{t} {np.mean(task_accs[t]):5.2f}/{np.mean(sia_accs[t]):5.2f}"
                    for t in ("V", "A")
                ),
                flush=True,
            )

        def stack(key_fn):
            vals = np.array([key_fn(r) for r in seed_rows], dtype=np.float64)
            return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                    "min": float(vals.min()), "max": float(vals.max()),
                    "seeds": [float(x) for x in vals]}

        by_sigma[f"sigma_{sigma}"] = {
            "clip_id": stack(lambda r: r["clip_id"]),
            "tasks": {
                t: {
                    "loso": stack(lambda r, tt=t: r["tasks"][tt]["loso"]),
                    "sia_cell": stack(lambda r, tt=t: r["tasks"][tt]["sia_cell"]),
                }
                for t in y_tasks
            },
            "per_seed": seed_rows,
        }
    return by_sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", nargs="?", default="all",
                        choices=("all", "sia", "phantom"))
    args = parser.parse_args()
    t0 = time.time()
    d = np.load(ROOT / "data" / "ceap_official.npz")
    feat, pid = d["feat"], d["pid"]
    v, _ = remap(d["vid"])
    win = window_index(pid, d["vid"])
    assert int(win.max()) == N_WIN - 1
    n_clip = int(v.max()) + 1
    y_tasks = {"V": d["y_val"], "A": d["y_aro"]}
    # 3-class / 5-class are slower and not required for the credit table;
    # keep binary, which is what the overclaim concerns.
    cache = OUT / "deconfound.json"
    out = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    out["protocol"] = {
        "A_ann": "train-only majority at the true (clip, window); equals Prior(v,t)",
        "A_sia": "same majority table indexed by LOSO-predicted clip and true window",
        "A_loso": "depth-4 RF on sensors, clip-mixed LOSO, seed 80",
        "StimulusCredit": "A_sia - majority",
        "SensorCredit": "A_loso - A_sia",
        "phantom_seeds": list(PHANTOM_SEEDS),
        "sigmas": list(SIGMAS),
        "note": "T (window index) is treated as observed without sensors",
    }

    if args.part in ("all", "sia"):
        print("=== shortcut-implied accuracy on real sensors ===", flush=True)
        out["sia"] = run_sia(feat, y_tasks, pid, v, win, n_clip)

    if args.part in ("all", "phantom"):
        print("=== multi-seed phantom calibration ===", flush=True)
        out["phantom"] = run_phantom(
            feat.shape, y_tasks, pid, v, win, n_clip, feat.shape[1]
        )

    out["elapsed_s"] = time.time() - t0
    OUT.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {cache} in {out['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
