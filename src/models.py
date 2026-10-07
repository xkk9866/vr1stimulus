"""FactorVR: timescale-identifiable factorization of VR psychophysiology.

One generative model, not a stack of unrelated modules.

x_t ≈ D_a z^a_t + D_s z^s_t + D_p z^p_t
z^k_t = α_k z^k_{t-1} + (1-α_k) h_t     (distinct AR timescales)
with auxiliary alignment of z^a→affect, z^s→SSQ, z^p→presence.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


def ema_filter(h: torch.Tensor, alpha: float) -> torch.Tensor:
    """Causal exponential moving average along time. h: (B, T, D)."""
    out = []
    state = torch.zeros(h.size(0), h.size(2), device=h.device, dtype=h.dtype)
    a = h.new_tensor(alpha)
    for t in range(h.size(1)):
        state = a * state + (1.0 - a) * h[:, t]
        out.append(state)
    return torch.stack(out, dim=1)


class SharedEncoder(nn.Module):
    def __init__(self, n_ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, C -> B, T, H
        h = self.net(x.transpose(1, 2))
        return h.transpose(1, 2)


class FactorVR(nn.Module):
    def __init__(self, n_ch: int = 9, hidden: int = 64, z_dim: int = 8):
        super().__init__()
        self.encoder = SharedEncoder(n_ch, hidden)
        self.to_a = nn.Linear(hidden, z_dim)
        self.to_s = nn.Linear(hidden, z_dim)
        self.to_p = nn.Linear(hidden, z_dim)
        self.dec = nn.Linear(z_dim * 3, n_ch)
        self.head_v = nn.Linear(z_dim, 1)
        self.head_a = nn.Linear(z_dim, 1)
        self.head_ssq = nn.Linear(z_dim, 1)
        self.head_ipq = nn.Linear(z_dim, 1)
        self.adv_ssq = nn.Sequential(nn.Linear(z_dim, 16), nn.GELU(), nn.Linear(16, 1))
        self.alpha_a = 0.35
        self.alpha_s = 0.82
        self.alpha_p = 0.93

    def factors(self, x: torch.Tensor):
        h = self.encoder(x)
        z_a = ema_filter(self.to_a(h), self.alpha_a)
        z_s = ema_filter(self.to_s(h), self.alpha_s)
        z_p = ema_filter(self.to_p(h), self.alpha_p)
        return z_a, z_s, z_p

    def forward(self, x: torch.Tensor, adv_lambda: float = 1.0) -> dict[str, torch.Tensor]:
        z_a, z_s, z_p = self.factors(x)
        zcat = torch.cat([z_a, z_s, z_p], dim=-1)
        rec = self.dec(zcat)
        pa = z_a.mean(dim=1)
        ps = z_s.mean(dim=1)
        pp = z_p.mean(dim=1)
        return {
            "z_a": z_a,
            "z_s": z_s,
            "z_p": z_p,
            "rec": rec,
            "logit_v": self.head_v(pa).squeeze(-1),
            "logit_a": self.head_a(pa).squeeze(-1),
            "pred_ssq": self.head_ssq(ps).squeeze(-1),
            "pred_ipq": self.head_ipq(pp).squeeze(-1),
            "adv_ssq": self.adv_ssq(grad_reverse(pa, adv_lambda)).squeeze(-1),
            "pool_a": pa,
            "pool_s": ps,
            "pool_p": pp,
        }


class CNNBaseline(nn.Module):
    def __init__(self, n_ch: int = 9, hidden: int = 64):
        super().__init__()
        self.encoder = SharedEncoder(n_ch, hidden)
        self.head_v = nn.Linear(hidden, 1)
        self.head_a = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x).mean(dim=1)
        return {"logit_v": self.head_v(h).squeeze(-1), "logit_a": self.head_a(h).squeeze(-1), "pool": h}


class CNNLSTMBaseline(nn.Module):
    def __init__(self, n_ch: int = 9, hidden: int = 64):
        super().__init__()
        self.encoder = SharedEncoder(n_ch, hidden)
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
        self.head_v = nn.Linear(hidden, 1)
        self.head_a = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        h, _ = self.rnn(h)
        h = h.mean(dim=1)
        return {"logit_v": self.head_v(h).squeeze(-1), "logit_a": self.head_a(h).squeeze(-1), "pool": h}


class TransformerBaseline(nn.Module):
    def __init__(self, n_ch: int = 9, hidden: int = 64, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(n_ch, hidden)
        layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head_v = nn.Linear(hidden, 1)
        self.head_a = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.enc(self.proj(x)).mean(dim=1)
        return {"logit_v": self.head_v(h).squeeze(-1), "logit_a": self.head_a(h).squeeze(-1), "pool": h}


def orthogonality_loss(z_a, z_s, z_p) -> torch.Tensor:
    def cos(u, v):
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        return (u * v).sum(dim=-1).pow(2).mean()

    return cos(z_a, z_s) + cos(z_a, z_p) + cos(z_s, z_p)
