"""Reproduce Xue et al. TMM Table III protocol + TwoClock.

SI = LOSOCV. Metrics = Accuracy and weighted F1 (paper).
Baselines cloned from CEAP-360VR official scripts: RF max_depth=4,
SVM, NB, KNN; 1D-CNN / LSTM architectures from the paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ceap_protocol.npz"
OUT = ROOT / "results"
SEED = 42

# Published SI 2s numbers (Xue et al., IEEE TMM 2021, Table III)
PUBLISHED_SI = {
    "RF": {"V_acc": 66.80, "V_wf1": 0.6238, "A_acc": 64.26, "A_wf1": 0.5298, "V3_acc": 49.92, "A3_acc": 52.20},
    "1D-CNN": {"V_acc": 64.27, "V_wf1": 0.5828, "A_acc": 67.64, "A_wf1": 0.5808, "V3_acc": 45.51, "A3_acc": 47.17},
    "LSTM": {"V_acc": 65.00, "V_wf1": 0.6349, "A_acc": 66.30, "A_wf1": 0.5934, "V3_acc": 44.62, "A3_acc": 43.79},
}


class CEAPCNN(nn.Module):
    """Paper 1D-CNN: filters (4,64),(16,32),(64,16),(128,8),(128,32) + GMP + dense."""

    def __init__(self, n_ch: int, n_cls: int):
        super().__init__()
        layers = []
        c = n_ch
        for n, k in ((4, 64), (16, 32), (64, 16), (128, 8), (128, 32)):
            layers += [nn.Conv1d(c, n, k, padding=k // 2), nn.ReLU(inplace=True)]
            c = n
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_cls)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))
        h = torch.amax(h, dim=-1)
        return self.head(h)


class CEAPLSTM(nn.Module):
    def __init__(self, n_ch: int, n_cls: int):
        super().__init__()
        self.rnn = nn.LSTM(n_ch, 100, batch_first=True)
        self.head = nn.Linear(100, n_cls)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1])


def scores(y, pred, prob=None, ssq=None):
    out = {
        "acc": float(accuracy_score(y, pred) * 100),
        "wf1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    if prob is not None and ssq is not None and len(np.unique(y)) == 2 and prob.ndim == 2 and prob.shape[1] >= 2:
        p = prob[:, 1]
        if p.std() > 1e-8 and np.std(ssq) > 1e-8:
            out["r_ssq"] = float(np.corrcoef(p, ssq)[0, 1])
        else:
            out["r_ssq"] = 0.0
    return out


def ml_model(name: str):
    if name == "RF":
        return RandomForestClassifier(max_depth=4, random_state=SEED, n_jobs=-1)
    if name == "SVM":
        return make_pipeline(StandardScaler(), LinearSVC(max_iter=4000, dual="auto"))
    if name == "NB":
        return GaussianNB()
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    raise ValueError(name)


def fit_ml(name, Xtr, ytr, Xte):
    clf = ml_model(name)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = None
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(Xte)
    elif hasattr(clf, "decision_function"):
        d = clf.decision_function(Xte)
        if d.ndim == 1:
            p = 1 / (1 + np.exp(-d))
            prob = np.stack([1 - p, p], axis=1)
    return pred, prob


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zscore(tr, te):
    mu = tr.mean(axis=(0, 1), keepdims=True)
    sd = tr.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((tr - mu) / sd).astype(np.float32), ((te - mu) / sd).astype(np.float32)


def train_dl(kind, Xtr, ytr, Xte, n_cls, epochs=12):
    dev = device()
    n_ch = Xtr.shape[-1]
    model = CEAPCNN(n_ch, n_cls) if kind == "CNN" else CEAPLSTM(n_ch, n_cls)
    model.to(dev)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    ds = TensorDataset(torch.from_numpy(Xtr), ytr_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = []
        te = DataLoader(TensorDataset(torch.from_numpy(Xte)), batch_size=256)
        for (xb,) in te:
            logits.append(model(xb.to(dev)).cpu().numpy())
    logit = np.concatenate(logits)
    pred = logit.argmax(1)
    e = np.exp(logit - logit.max(1, keepdims=True))
    prob = e / e.sum(1, keepdims=True)
    return pred, prob


def loso(pid):
    subs = sorted(set(pid.tolist()))
    for s in subs:
        te = pid == s
        tr = ~te
        yield s, np.where(tr)[0], np.where(te)[0]


def main():
    d = np.load(DATA)
    pid = d["pid"]
    print(f"N={len(pid)} subjects={len(set(pid.tolist()))} device={device()}")
    tasks = {
        "V": d["y_val"],
        "A": d["y_aro"],
        "V3": d["y_val3"],
        "A3": d["y_aro3"],
    }
    rows = []

    def add_row(model, feat_key, x_key, kind):
        cell = {"model": model}
        for task, y in tasks.items():
            accs, f1s, rs = [], [], []
            for _s, tr, te in loso(pid):
                ytr, yte = y[tr], y[te]
                n_cls = int(y.max()) + 1
                if kind == "ml":
                    pred, prob = fit_ml(model.replace("TwoClock-", ""), d[feat_key][tr], ytr, d[feat_key][te])
                else:
                    Xtr, Xte = zscore(d[x_key][tr], d[x_key][te])
                    dl = "CNN" if "CNN" in model else "LSTM"
                    pred, prob = train_dl(dl, Xtr, ytr, Xte, n_cls)
                sc = scores(yte, pred, prob, d["ssq"][te])
                accs.append(sc["acc"])
                f1s.append(sc["wf1"])
                if "r_ssq" in sc:
                    rs.append(sc["r_ssq"])
            cell[f"{task}_acc"] = float(np.mean(accs))
            cell[f"{task}_wf1"] = float(np.mean(f1s))
            if rs:
                cell[f"{task}_r"] = float(np.mean(rs))
            print(f"  {model:16s} {task:3s} acc={cell[f'{task}_acc']:.2f} wf1={cell[f'{task}_wf1']:.4f}")
        rows.append(cell)

    print("\n=== official-style ML on raw features ===")
    for name in ("RF", "SVM", "NB", "KNN"):
        add_row(name, "feat", "X", "ml")

    print("\n=== TwoClock + same ML (fast residual features) ===")
    add_row("TwoClock-RF", "feat_fast", "Xfast", "ml")
    add_row("TwoClock-SVM", "feat_fast", "Xfast", "ml")

    print("\n=== paper 1D-CNN / LSTM on raw vs TwoClock residual ===")
    add_row("1D-CNN", "feat", "X", "dl")
    add_row("LSTM", "feat", "X", "dl")
    add_row("TwoClock-CNN", "feat_fast", "Xfast", "dl")
    add_row("TwoClock-LSTM", "feat_fast", "Xfast", "dl")

    OUT.mkdir(exist_ok=True)
    (OUT / "table3_protocol.json").write_text(json.dumps({"ours": rows, "published_si": PUBLISHED_SI, "n": int(len(pid))}, indent=2), encoding="utf-8")
    cols = ["model", "V_acc", "V_wf1", "A_acc", "A_wf1", "V3_acc", "V3_wf1", "A3_acc", "A3_wf1", "A_r"]
    lines = [
        "# Table: CEAP-360VR subject-independent (LOSO, 2 s), matching Xue et al. TMM 2021",
        "",
        f"N_windows={len(pid)}, N_subjects={len(set(pid.tolist()))}. Metrics: Accuracy (%) and weighted F1.",
        "",
        "| Model | V Acc | V wF1 | A Acc | A wF1 | V3 Acc | A3 Acc | r(A, SSQ) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| RF (published) | 66.80 | 0.6238 | 64.26 | 0.5298 | 49.92 | 52.20 | — |",
        "| 1D-CNN (published) | 64.27 | 0.5828 | 67.64 | 0.5808 | 45.51 | 47.17 | — |",
        "| LSTM (published) | 65.00 | 0.6349 | 66.30 | 0.5934 | 44.62 | 43.79 | — |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r.get('V_acc', float('nan')):.2f} | {r.get('V_wf1', float('nan')):.4f} | "
            f"{r.get('A_acc', float('nan')):.2f} | {r.get('A_wf1', float('nan')):.4f} | "
            f"{r.get('V3_acc', float('nan')):.2f} | {r.get('A3_acc', float('nan')):.2f} | "
            f"{r.get('A_r', float('nan')):.3f} |"
        )
    (OUT / "table3_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main() or 0)
