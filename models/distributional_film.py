"""Cross-modal attention and non-crossing quantile FiLM on the output distribution.

Y_0.5 = Y_0.5,base * exp(m u) + m beta sigma_X
Y_q   = Y_0.5 + (Y_q,base - Y_0.5,base) * exp(m u) * (1 + m (s - 1))
m(t) = 0 gives Y = Y_base exactly.
"""
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from configs.model_config import ModelConfig

LOCALITY_SLOPE = 2.0  # attention logit bias -slope * |t - t'|: step t attends mostly to its own Z_qwen(t)


def event_lifecycle(start: Tensor, end: Tensor, valid: Tensor, horizon: int, cfg: ModelConfig) -> Tuple[Tensor, Tensor]:
    """start, end, valid: [B, N]. Returns per-event envelope [B, N, H] in [0, 1] and phase one-hot [B, N, H, 4].

    Envelope: 1 in [start, end); exp(-|dt| / tau_buf) in the pre and post buffers; 0 elsewhere.
    Phases: warm-up [start - pre, start), burst [start, mid), decay [mid, end), overdraft [end, end + post).
    """
    t = torch.arange(horizon, device=start.device, dtype=start.dtype)
    s, e = start[..., None], end[..., None]
    inside = (t >= s) & (t < e)
    before = (t >= s - cfg.pre_buffer) & (t < s)
    after = (t >= e) & (t < e + cfg.post_buffer)
    dist = torch.where(before, s - t, t - e + 1)
    buffer = torch.exp(-dist.abs() / cfg.tau_buf)
    env = torch.where(inside, torch.ones_like(buffer), torch.where(before | after, buffer, torch.zeros_like(buffer)))
    env = env * valid[..., None]
    mid = (s + e) / 2
    phase = torch.stack([before, inside & (t < mid), inside & (t >= mid), after], -1) & valid[..., None, None]
    return env, phase.to(start.dtype)


@dataclass
class FiLMOutput:
    quantiles: Tensor  # [B, H, |Q|]
    u_median: Tensor  # [B, H] log-scale ratio u_0.5
    beta: Tensor  # [B, H]
    spread: Tensor  # [B, H]


class DistributionalFiLM(nn.Module):
    def __init__(self, cfg: ModelConfig, d_ts: int, d_llm: int):
        super().__init__()
        self.alpha_max = cfg.alpha_max
        # kdim/vdim != embed_dim gives separate W_Q, W_K, W_V.
        self.attn = nn.MultiheadAttention(d_ts, cfg.ts_heads, kdim=d_llm, vdim=d_llm, batch_first=True)
        self.norm = nn.LayerNorm(d_ts)
        self.lin_u, self.lin_beta, self.lin_s = (nn.Linear(d_ts, 1) for _ in range(3))
        for lin in (self.lin_u, self.lin_beta, self.lin_s):
            nn.init.normal_(lin.weight, std=1e-3)
            nn.init.zeros_(lin.bias)
        nn.init.constant_(self.lin_s.bias, math.log(math.expm1(0.9)))  # spread = 1.0 at init

    def forward(self, h_fut: Tensor, z_qwen: Tensor, base: Tensor, m: Tensor, sigma: Tensor, median_idx: int) -> FiLMOutput:
        """h_fut: [B, H, d_ts], z_qwen: [B, H, d_llm], base: [B, H, Q], m: [B, H], sigma: [B, 1]."""
        t = torch.arange(h_fut.shape[1], device=h_fut.device, dtype=h_fut.dtype)
        locality = -LOCALITY_SLOPE * (t[:, None] - t[None, :]).abs()
        f_ctx = self.norm(h_fut + self.attn(h_fut, z_qwen, z_qwen, attn_mask=locality, need_weights=False)[0])
        u = self.alpha_max * torch.tanh(self.lin_u(f_ctx)).squeeze(-1)
        beta = self.lin_beta(f_ctx).squeeze(-1)
        spread = F.softplus(self.lin_s(f_ctx)).squeeze(-1) + 0.1

        base_med = base[..., median_idx]
        scale = torch.exp(m * u)
        y_med = base_med * scale + m * beta * sigma
        width = (scale * (1 + m * (spread - 1)))[..., None]
        y = y_med[..., None] + (base - base_med[..., None]) * width
        # Same value as the formula at m = 0, but without float rounding of base_med + (base - base_med).
        y = torch.where(m[..., None] > 0, y, base)
        return FiLMOutput(y, u, beta, spread)
