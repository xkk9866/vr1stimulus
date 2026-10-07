"""Delay-aware residual model on CEAP-360VR.

The stimulus prior already knows how the crowd feels at this second of
this clip. A wearable is useful only if it can predict the *departure*
from that crowd, and autonomic responses arrive with a lag of several
seconds. This script tests exactly that claim:

    yhat_{s,v,t} = c_hat_{v,t} + f_theta(S_{s, v, t-L:t})

c_hat is the mean continuous rating of the *training* participants at
the same clip and window. f_theta is a causal TCN or a GRU reading the
past L seconds of the 12 released channels (physiology and behaviour).
The training target is the residual, not the raw affect label.

Even a null here is a strong negative: the architecture is built to
exploit delay and to see the prior, so a failure cannot be dismissed as
"you used a weak classifier on the wrong target."
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 80
N_WIN = 30
FS = 50  # Hz inside each released 2 s window
WIN_SEC = 2.0
SAMPLES = 100  # 2 s * 50 Hz

# Lookbacks in seconds. 2 s = current window only; 10 s covers typical EDA lag.
LOOKBACKS = (2, 10)
MODELS = ("tcn", "gru")
EPOCHS = 40
PATIENCE = 6
BATCH = 128
LR = 1e-3
N_VAL_SUBJ = 4


def rank_biserial(delta: np.ndarray) -> float:
    nz = delta[delta != 0]
    if nz.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(nz))
    return float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())


def paired(a, b) -> dict:
    d = np.asarray(b, float) - np.asarray(a, float)
    if np.allclose(d, 0):
        return {"delta": 0.0, "p": 1.0, "p_perm": 1.0, "rb": 0.0, "wins": 0, "n": len(d)}
    p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    rng = np.random.default_rng(SEED)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(20000, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return {
        "delta": float(d.mean()),
        "p": p,
        "p_perm": float(((null >= obs).sum() + 1) / (len(null) + 1)),
        "rb": rank_biserial(d),
        "wins": int((d > 0).sum()),
        "n": int(len(d)),
    }


class CausalConv1d(nn.Module):
    """Left-padded convolution: output at t depends only on inputs <= t."""

    def __init__(self, n_in: int, n_out: int, k: int, dilation: int):
        super().__init__()
        self.left = (k - 1) * dilation
        self.conv = nn.Conv1d(n_in, n_out, k, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.left, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, k: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, k, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, k, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualTCN(nn.Module):
    """Causal TCN -> last timestep -> scalar residual."""

    def __init__(self, n_ch: int, hidden: int = 64, k: int = 5):
        super().__init__()
        self.proj = CausalConv1d(n_ch, hidden, 1, 1)
        # receptive field: 1 + 2*(k-1)*sum(dilations) ≈ 10 s at 50 Hz for k=5
        self.blocks = nn.Sequential(
            *[TCNBlock(hidden, k, d) for d in (1, 2, 4, 8, 16, 32)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, C
        h = self.proj(x.transpose(1, 2))
        h = self.blocks(h)
        return self.head(h[:, :, -1]).squeeze(-1)


class ResidualGRU(nn.Module):
    def __init__(self, n_ch: int, hidden: int = 64):
        super().__init__()
        self.rnn = nn.GRU(n_ch, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.head(h[:, -1]).squeeze(-1)


def make_model(kind: str, n_ch: int) -> nn.Module:
    if kind == "tcn":
        return ResidualTCN(n_ch)
    if kind == "gru":
        return ResidualGRU(n_ch)
    raise ValueError(kind)


def window_index(pid: np.ndarray, vid: np.ndarray) -> np.ndarray:
    """0..29 within each (subject, clip), preserving file order."""
    win = np.zeros(len(pid), dtype=np.int64)
    for p in np.unique(pid):
        for v in np.unique(vid[pid == p]):
            idx = np.flatnonzero((pid == p) & (vid == v))
            win[idx] = np.arange(len(idx))
    return win


def lookback_tensor(X: np.ndarray, pid: np.ndarray, vid: np.ndarray, win: np.ndarray, k_win: int) -> np.ndarray:
    """Causal stack of the last k_win two-second windows. Zero-pad at clip start."""
    n, t, c = X.shape
    out = np.zeros((n, k_win * t, c), dtype=np.float32)
    index = {(int(p), int(v), int(w)): i for i, (p, v, w) in enumerate(zip(pid, vid, win))}
    for i, (p, v, w) in enumerate(zip(pid, vid, win)):
        chunks = []
        for dw in range(k_win - 1, -1, -1):
            src = index.get((int(p), int(v), int(w) - dw))
            chunks.append(X[src] if src is not None else np.zeros((t, c), np.float32))
        out[i] = np.concatenate(chunks, axis=0)
    return out


def crowd_mean(y: np.ndarray, vid: np.ndarray, win: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LOSO crowd trajectory: mean of training subjects at each (clip, window)."""
    out = np.zeros(len(y), dtype=np.float64)
    g = float(y[mask].mean())
    keys = {}
    for v, w, val in zip(vid[mask], win[mask], y[mask]):
        keys.setdefault((int(v), int(w)), []).append(float(val))
    cmap = {k: float(np.mean(vs)) for k, vs in keys.items()}
    for i in range(len(y)):
        out[i] = cmap.get((int(vid[i]), int(win[i])), g)
    return out


def train_regressor(kind: str, Xtr, ytr, Xva, yva, Xte, device) -> np.ndarray:
    torch.manual_seed(SEED)
    model = make_model(kind, Xtr.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=BATCH,
        shuffle=True,
        drop_last=False,
    )
    xva = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva.astype(np.float32)).to(device)
    best, bad, best_state = float("inf"), 0, None
    model.train()
    for _ in range(EPOCHS):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(xb)
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(xva), yva_t))
        model.train()
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(Xte)), batch_size=256):
            preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds)


def standardize_time(Xtr, Xte):
    """Channel-wise z-score over training time and samples. No test leakage."""
    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)


def eval_task(name, y_cont, y_bin, X_hist, pid, vid, win, kind, device):
    """LOSO residual regression, then derived deviation and reconstructed labels."""
    subjects = np.unique(pid)
    per = {
        "r": [],
        "rmse": [],
        "rmse0": [],
        "dev_acc": [],
        "dev_maj": [],
        "recon_acc": [],
        "prior_acc": [],
    }
    for i, s in enumerate(subjects):
        te = pid == s
        tr_all = pid != s
        tr_sub = np.unique(pid[tr_all])
        rng = np.random.default_rng(SEED + int(s))
        n_val = min(N_VAL_SUBJ, max(0, len(tr_sub) - 1))
        if n_val > 0:
            val_ids = set(rng.choice(tr_sub, size=n_val, replace=False))
            va = np.array([p in val_ids for p in pid]) & tr_all
            tr = tr_all & ~va
        else:
            # tiny split: last 15% of training windows as validation
            tr_idx = np.flatnonzero(tr_all)
            cut = max(1, int(0.85 * len(tr_idx)))
            tr = np.zeros(len(pid), dtype=bool)
            va = np.zeros(len(pid), dtype=bool)
            tr[tr_idx[:cut]] = True
            va[tr_idx[cut:]] = True
        c = crowd_mean(y_cont, vid, win, tr)
        r_tr = y_cont[tr] - c[tr]
        r_va = y_cont[va] - c[va]
        r_te = y_cont[te] - c[te]
        Xtr, Xte_s = standardize_time(X_hist[tr], X_hist[te])
        _, Xva = standardize_time(X_hist[tr], X_hist[va])
        pred_r = train_regressor(kind, Xtr, r_tr, Xva, r_va, Xte_s, device)
        yhat = c[te] + pred_r
        # residual quality
        if np.std(pred_r) < 1e-12 or np.std(r_te) < 1e-12:
            per["r"].append(0.0)
        else:
            per["r"].append(float(np.corrcoef(pred_r, r_te)[0, 1]))
        per["rmse"].append(float(np.sqrt(np.mean((pred_r - r_te) ** 2))))
        per["rmse0"].append(float(np.sqrt(np.mean(r_te ** 2))))  # predict-zero
        # deviation sign
        d_true = (r_te > 0).astype(np.int64)
        d_pred = (pred_r > 0).astype(np.int64)
        d_tr = (r_tr > 0).astype(np.int64)
        maj = int(np.bincount(d_tr, minlength=2).argmax())
        per["dev_acc"].append(accuracy_score(d_true, d_pred) * 100)
        per["dev_maj"].append(accuracy_score(d_true, np.full(len(d_true), maj)) * 100)
        # reconstructed binary affect vs stimulus prior
        yhat_bin = (yhat >= 5.0).astype(np.int64)
        prior_bin = (c[te] >= 5.0).astype(np.int64)
        per["recon_acc"].append(accuracy_score(y_bin[te], yhat_bin) * 100)
        per["prior_acc"].append(accuracy_score(y_bin[te], prior_bin) * 100)
        print(
            f"  {name:8s} {kind:3s} subj {int(s):02d}  r={per['r'][-1]:+.3f}  "
            f"dev {per['dev_acc'][-1]:5.1f}/{per['dev_maj'][-1]:5.1f}  "
            f"recon {per['recon_acc'][-1]:5.1f} prior {per['prior_acc'][-1]:5.1f}",
            flush=True,
        )
    bag = {k: float(np.mean(v)) for k, v in per.items()}
    bag["per_fold"] = {k: [float(x) for x in v] for k, v in per.items()}
    bag["tests"] = {
        "dev vs majority": paired(per["dev_maj"], per["dev_acc"]),
        "recon vs prior": paired(per["prior_acc"], per["recon_acc"]),
        "rmse vs predict-zero": paired(per["rmse0"], per["rmse"]),
    }
    return bag


def main():
    import sys

    t0 = time.time()
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    off = np.load(ROOT / "data" / "ceap_official.npz")
    cont = np.load(ROOT / "data" / "ceap_va_cont.npz")
    X, pid, vid = off["X"], off["pid"], off["vid"]
    win = window_index(pid, vid)
    assert int(win.max()) == N_WIN - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"X {X.shape} device {device}", flush=True)

    LOOKBACKS_RUN = (10,) if smoke else LOOKBACKS
    MODELS_RUN = ("tcn",) if smoke else MODELS
    tasks = {
        "valence": (cont["y_val_c"], off["y_val"]),
        "arousal": (cont["y_aro_c"], off["y_aro"]),
    }
    if smoke:
        tasks = {"valence": tasks["valence"]}
        keep = pid <= 2
        X, pid, vid, win = X[keep], pid[keep], vid[keep], win[keep]
        tasks = {"valence": (cont["y_val_c"][keep], off["y_val"][keep])}
        print("SMOKE: subjects 1-2, TCN L=10 valence only", flush=True)

    hist = {}
    for lb in LOOKBACKS_RUN:
        k_win = max(1, int(round(lb / WIN_SEC)))
        print(f"building lookback {lb}s  k={k_win}  T={k_win * SAMPLES}", flush=True)
        hist[lb] = lookback_tensor(X, pid, vid, win, k_win)
    out = {
        "protocol": {
            "lookbacks_s": list(LOOKBACKS),
            "models": list(MODELS),
            "equation": "yhat = crowd_mean(v,t) + f(S[t-L:t])",
            "target": "continuous residual, not raw affect label",
        },
        "runs": {},
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "residual_seq.json"
    for lb in LOOKBACKS_RUN:
        for kind in MODELS_RUN:
            for tname, (yc, yb) in tasks.items():
                key = f"{kind}_L{lb}_{tname}"
                print(f"\n=== {key} ===", flush=True)
                out["runs"][key] = eval_task(
                    tname, yc, yb, hist[lb], pid, vid, win, kind, device
                )
                bag = out["runs"][key]
                print(
                    f"MEAN {key}: r={bag['r']:+.3f}  "
                    f"dev {bag['dev_acc']:.2f} vs maj {bag['dev_maj']:.2f}  "
                    f"recon {bag['recon_acc']:.2f} vs prior {bag['prior_acc']:.2f}  "
                    f"p_dev={bag['tests']['dev vs majority']['p']:.3f}  "
                    f"p_recon={bag['tests']['recon vs prior']['p']:.3f}",
                    flush=True,
                )
                path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["elapsed_s"] = time.time() - t0
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {path} in {out['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
