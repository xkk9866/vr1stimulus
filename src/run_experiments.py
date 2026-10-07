"""Subject-independent CV: baselines vs FactorVR, plus sickness-confound tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import CNNLSTMBaseline, CNNBaseline, FactorVR, TransformerBaseline, orthogonality_loss

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "windows.npz"
OUT = ROOT / "results"
SEED = 42


def seed_all(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load() -> dict[str, np.ndarray]:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}. Run ceap_prepare.py")
    z = np.load(DATA, allow_pickle=True)
    return {k: z[k] for k in z.files}


def window_features(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    slope = X[:, -1] - X[:, 0]
    return np.concatenate([mean, std, mn, mx, slope], axis=1)


def metrics(y: np.ndarray, logit: np.ndarray) -> dict[str, float]:
    prob = 1.0 / (1.0 + np.exp(-logit))
    pred = (prob >= 0.5).astype(int)
    out = {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def confound_r(prob: np.ndarray, ssq: np.ndarray) -> float:
    if prob.std() < 1e-8 or ssq.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(prob, ssq)[0, 1])


def subject_folds(pid: np.ndarray, n_folds: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    subjects = np.array(sorted(set(pid.tolist())))
    n_folds = max(2, min(n_folds, len(subjects)))
    folds = np.array_split(subjects, n_folds)
    out = []
    for test_sub in folds:
        if len(test_sub) == 0:
            continue
        te = np.isin(pid, test_sub)
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        out.append((np.where(tr)[0], np.where(te)[0]))
    return out


def zscore_train(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)


def fit_sklearn(Xtr, ytr, Xte):
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=SEED
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    pos = proba[:, 1] if proba.shape[1] == 2 else np.zeros(len(Xte))
    logit = np.clip(pos, 1e-4, 1 - 1e-4)
    logit = np.log(logit / (1 - logit))
    return logit


def _loader(X, yv, ya, ssq, ipq, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(yv.astype(np.float32)),
        torch.from_numpy(ya.astype(np.float32)),
        torch.from_numpy(ssq.astype(np.float32)),
        torch.from_numpy(ipq.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


def train_torch(name: str, n_ch: int, loaders, epochs: int, dev: torch.device):
    if name == "cnn":
        model = CNNBaseline(n_ch)
    elif name == "cnnlstm":
        model = CNNLSTMBaseline(n_ch)
    elif name == "transformer":
        model = TransformerBaseline(n_ch)
    else:
        model = FactorVR(n_ch)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    train_loader, _ = loaders
    model.train()
    for ep in range(epochs):
        tot = 0.0
        n = 0
        for xb, yv, ya, ssq, ipq in train_loader:
            xb, yv, ya, ssq, ipq = xb.to(dev), yv.to(dev), ya.to(dev), ssq.to(dev), ipq.to(dev)
            opt.zero_grad()
            out = model(xb) if name != "factorvr" else model(xb, adv_lambda=min(1.0, ep / 8))
            loss = bce(out["logit_v"], yv) + bce(out["logit_a"], ya)
            if name == "factorvr":
                loss = loss + 0.45 * mse(out["rec"], xb)
                loss = loss + 0.25 * mse(out["pred_ssq"], ssq)
                loss = loss + 0.15 * mse(out["pred_ipq"], ipq)
                loss = loss + 0.10 * orthogonality_loss(out["pool_a"], out["pool_s"], out["pool_p"])
                # z_a should not carry sickness: adversarial BCE on median-split SSQ
                ssq_bin = (ssq > 0).float()
                loss = loss + 0.20 * bce(out["adv_ssq"], ssq_bin)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()) * len(xb)
            n += len(xb)
        if ep == 0 or ep == epochs - 1:
            print(f"  {name} epoch {ep+1}/{epochs} loss={tot / max(n,1):.4f}")
    return model


@torch.no_grad()
def predict_torch(model, name: str, loader, dev):
    model.eval()
    lv, la, pools = [], [], []
    for xb, *_rest in loader:
        xb = xb.to(dev)
        out = model(xb)
        lv.append(out["logit_v"].cpu().numpy())
        la.append(out["logit_a"].cpu().numpy())
        key = "pool_a" if name == "factorvr" else "pool"
        pools.append(out[key].cpu().numpy())
    return np.concatenate(lv), np.concatenate(la), np.concatenate(pools)


def main() -> int:
    seed_all()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    X, pid = d["X"], d["pid"]
    print(f"windows={X.shape} subjects={len(set(pid.tolist()))} device={device()}")
    # Standardize SSQ/IPQ globally for regression targets (scale only).
    ssq_z = (d["ssq"] - d["ssq"].mean()) / (d["ssq"].std() + 1e-6)
    ipq_z = (d["ipq"] - d["ipq"].mean()) / (d["ipq"].std() + 1e-6)

    models = ["rf", "logreg", "cnn", "cnnlstm", "transformer", "factorvr"]
    folds = subject_folds(pid)
    all_rows = []
    dev = device()
    epochs = 12 if dev.type == "cuda" else 8

    for fold_i, (tr, te) in enumerate(folds, 1):
        print(f"\n=== fold {fold_i}/{len(folds)} test subjects {sorted(set(pid[te].tolist()))} ===")
        Xtr, Xte = zscore_train(X[tr], X[te])
        yv_tr, yv_te = d["y_val"][tr], d["y_val"][te]
        ya_tr, ya_te = d["y_aro"][tr], d["y_aro"][te]
        feat_tr, feat_te = window_features(Xtr), window_features(Xte)
        scaler = StandardScaler()
        feat_tr = scaler.fit_transform(feat_tr)
        feat_te = scaler.transform(feat_te)

        for name in models:
            if name == "rf":
                lv = fit_sklearn(feat_tr, yv_tr, feat_te)
                la = fit_sklearn(feat_tr, ya_tr, feat_te)
            elif name == "logreg":
                lv_m = LogisticRegression(max_iter=400, class_weight="balanced")
                la_m = LogisticRegression(max_iter=400, class_weight="balanced")
                lv_m.fit(feat_tr, yv_tr)
                la_m.fit(feat_tr, ya_tr)
                lv = lv_m.decision_function(feat_te)
                la = la_m.decision_function(feat_te)
            else:
                tr_loader = _loader(Xtr, yv_tr, ya_tr, ssq_z[tr], ipq_z[tr], 256, True)
                te_loader = _loader(Xte, yv_te, ya_te, ssq_z[te], ipq_z[te], 256, False)
                model = train_torch(name, X.shape[-1], (tr_loader, te_loader), epochs, dev)
                lv, la, _pool = predict_torch(model, name, te_loader, dev)

            mv = metrics(yv_te, lv)
            ma = metrics(ya_te, la)
            pv = 1 / (1 + np.exp(-lv))
            pa = 1 / (1 + np.exp(-la))
            row = {
                "fold": fold_i,
                "model": name,
                "val": mv,
                "aro": ma,
                "r_val_ssq": confound_r(pv, d["ssq"][te]),
                "r_aro_ssq": confound_r(pa, d["ssq"][te]),
                "r_val_hm": confound_r(pv, d["hm_energy"][te]),
                "r_aro_hm": confound_r(pa, d["hm_energy"][te]),
            }
            all_rows.append(row)
            print(
                f"  {name:12s} V f1={mv['f1']:.3f} auc={mv['auc']:.3f} | "
                f"A f1={ma['f1']:.3f} auc={ma['auc']:.3f} | r(A,SSQ)={row['r_aro_ssq']:.3f}"
            )

    summary = {}
    for name in models:
        rows = [r for r in all_rows if r["model"] == name]
        def agg(path):
            xs = [r[path[0]][path[1]] if len(path) == 2 else r[path[0]] for r in rows]
            xs = [x for x in xs if x == x]
            return {"mean": float(np.mean(xs)), "std": float(np.std(xs))}

        summary[name] = {
            "val_f1": agg(("val", "f1")),
            "val_auc": agg(("val", "auc")),
            "val_acc": agg(("val", "acc")),
            "aro_f1": agg(("aro", "f1")),
            "aro_auc": agg(("aro", "auc")),
            "aro_acc": agg(("aro", "acc")),
            "r_aro_ssq": agg(("r_aro_ssq",)),
            "r_val_ssq": agg(("r_val_ssq",)),
        }

    payload = {"folds": all_rows, "summary": summary, "n_windows": int(X.shape[0])}
    (OUT / "cv_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# FactorVR CEAP-360VR subject-independent CV", "", "| Model | Val F1 | Val AUC | Aro F1 | Aro AUC | r(Aro, SSQ) |", "|---|---:|---:|---:|---:|---:|"]
    for name in models:
        s = summary[name]
        lines.append(
            f"| {name} | {s['val_f1']['mean']:.3f}±{s['val_f1']['std']:.3f} | "
            f"{s['val_auc']['mean']:.3f}±{s['val_auc']['std']:.3f} | "
            f"{s['aro_f1']['mean']:.3f}±{s['aro_f1']['std']:.3f} | "
            f"{s['aro_auc']['mean']:.3f}±{s['aro_auc']['std']:.3f} | "
            f"{s['r_aro_ssq']['mean']:.3f}±{s['r_aro_ssq']['std']:.3f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
