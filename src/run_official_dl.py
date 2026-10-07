"""Official CEAP deep baselines on the 12-channel tensors.

Architectures ported verbatim from the released 4_Classifer_models.py
(github.com/cwi-dis/CEAP-360VR-Dataset), not from the paper text, which
lists a different five-layer CNN. Training follows the released code:
RMSprop 1e-3, batch 256, 50 epochs, early stop on training loss
(patience 5) and ReduceLROnPlateau (factor 0.1, patience 3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
WINDOW = 100
EPOCHS = 50


class OfficialCNN(nn.Module):
    def __init__(self, n_ch: int, n_cls: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch, 4, 256, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, 8, 128, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 32, 64, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Linear(128, n_cls))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))
        return self.head(torch.amax(h, dim=-1))


class OfficialLSTM(nn.Module):
    def __init__(self, n_ch: int, n_cls: int):
        super().__init__()
        self.rnn = nn.LSTM(n_ch, WINDOW, batch_first=True)
        self.head = nn.Linear(WINDOW, n_cls)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1])


def train_eval(kind, Xtr, ytr, Xte, n_cls, dev):
    model = (OfficialCNN if kind == "1DCNN" else OfficialLSTM)(Xtr.shape[-1], n_cls).to(dev)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=3)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))),
        batch_size=256,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss()
    best, bad = float("inf"), 0
    model.train()
    for _ in range(EPOCHS):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
        total /= len(loader.dataset)
        sched.step(total)
        if total < best - 1e-4:
            best, bad = total, 0
        else:
            bad += 1
            if bad >= 5:
                break
    model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(Xte)), batch_size=512):
            preds.append(model(xb.to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "both"
    d = np.load(ROOT / "data" / "ceap_official.npz")
    X, pid = d["X"], d["pid"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("X", X.shape, "device", dev, flush=True)
    tasks = {
        "V": d["y_val"],
        "A": d["y_aro"],
        "V3": d["y_val3"],
        "A3": d["y_aro3"],
        "C5": d["y5"],
    }
    kinds = ("1DCNN", "LSTM") if only == "both" else (only,)
    out_path = OUT / "official_dl.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for kind in kinds:
        out.setdefault(kind, {})
        for task, y in tasks.items():
            n_cls = int(y.max()) + 1
            accs, f1s, subj = [], [], {}
            for s in sorted(set(pid.tolist())):
                te = pid == s
                tr = ~te
                pred = train_eval(kind, X[tr], y[tr], X[te], n_cls, dev)
                accs.append(accuracy_score(y[te], pred) * 100)
                f1s.append(f1_score(y[te], pred, average="weighted", zero_division=0))
                subj[int(s)] = float(accs[-1])
            out[kind][task] = {
                "acc": float(np.mean(accs)),
                "wf1": float(np.mean(f1s)),
                "subj": subj,
            }
            print(f"{kind:6s} {task:3s} acc={np.mean(accs):.2f} wf1={np.mean(f1s):.4f}", flush=True)
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/official_dl.json", flush=True)


if __name__ == "__main__":
    main()
