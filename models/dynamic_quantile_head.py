"""Dynamic low-rank quantile head: P_step(t) generates per-step U_head(t), V_head(t) (HeadHyperMLP).

[u~, beta, s~](t) = (H_top(t) U_head(t)) V_head(t)
Y_0.5 = Y_0.5,base * exp(m u) + m beta sigma_X,              u = 2.5 tanh(u~)
Y_q   = Y_0.5 + (Y_q,base - Y_0.5,base) * exp(m u) * (1 + m (s - 1)),   s = Softplus(s~) + 0.1
m(t) = 0 gives Y = Y_base exactly. Non-crossing: exp(m u) > 0 and 1 + m (s - 1) >= 0.1.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from configs.model_config import ModelConfig


@dataclass
class HeadOutput:
    quantiles: Tensor  # [B, H, |Q|]
    u_median: Tensor  # [B, H]


class DynamicQuantileHead(nn.Module):
    def __init__(self, cfg: ModelConfig, d_ts: int, d_llm: int):
        super().__init__()
        self.d_ts, self.r, self.alpha_max = d_ts, cfg.head_rank, cfg.alpha_max
        n_u = d_ts * self.r
        self.hyper = nn.Sequential(nn.Linear(d_llm, d_llm), nn.GELU(), nn.Linear(d_llm, n_u + self.r * 3))  # HeadHyperMLP
        out = self.hyper[-1]
        nn.init.normal_(out.weight, std=1e-3)
        with torch.no_grad():  # U ~ N(0, 1/d_ts), V ~ N(0, 1e-2): small, non-zero head output at step 1
            out.bias[:n_u].normal_(std=1 / math.sqrt(d_ts))
            out.bias[n_u:].normal_(std=1e-2)

    def forward(self, p_step: Tensor, h_top: Tensor, base: Tensor, m: Tensor, sigma: Tensor, median_idx: int) -> HeadOutput:
        """p_step: [B, H, d_llm], h_top: [B, H, d_ts], base: [B, H, Q], m: [B, H], sigma: [B, 1]."""
        B, H, _ = h_top.shape
        w = self.hyper(p_step)
        U = w[..., : self.d_ts * self.r].view(B, H, self.d_ts, self.r)
        V = w[..., self.d_ts * self.r :].view(B, H, self.r, 3)
        u_raw, beta, s_raw = (h_top[..., None, :] @ U @ V).squeeze(-2).unbind(-1)
        u = self.alpha_max * torch.tanh(u_raw)
        spread = F.softplus(s_raw) + 0.1

        base_med = base[..., median_idx]
        scale = torch.exp(m * u)
        y_med = base_med * scale + m * beta * sigma
        y = y_med[..., None] + (base - base_med[..., None]) * (scale * (1 + m * (spread - 1)))[..., None]
        # Same value as the formula at m = 0, but without float rounding of base_med + (base - base_med).
        return HeadOutput(torch.where(m[..., None] > 0, y, base), u)
