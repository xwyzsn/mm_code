"""TimesFM backbone. Gives shallow patch states, future top-level states and the base quantile forecast.

prod:      official TimesFM 3.0 from the `timesfm3` package (frozen) + a trainable horizon adapter.
mac_debug: MockTimesFM3Backbone, a small causal patch Transformer with the same outputs.
Both support any history length L, any horizon H and any quantile set Q.
"""
import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from configs.model_config import ModelConfig


def masked_mean_std(x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """x, mask: [B, L]. Returns mu, sigma: [B, 1], sigma = max(std, 1e-5)."""
    m = mask.to(x.dtype)
    n = m.sum(-1, keepdim=True).clamp(min=1)
    mu = (x * m).sum(-1, keepdim=True) / n
    var = (((x - mu) * m) ** 2).sum(-1, keepdim=True) / n
    return mu, var.sqrt().clamp(min=1e-5)


def sinusoidal(pos: Tensor, dim: int) -> Tensor:
    """pos: [...] float. Returns [..., dim]."""
    freqs = torch.exp(-torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32) * (9.2103 / dim))  # ln(1e4)
    ang = pos[..., None].float() * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


@dataclass
class TimesFMOutput:
    shallow_hidden: Tensor  # [B, L, d_ts] shallow patch states, repeated to each time step
    fut_hidden: Tensor  # [B, H, d_ts] top-level future states
    base_quantiles: Tensor  # [B, H, |Q|] original scale


class MockTimesFM3Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, p = cfg.d_ts, cfg.ts_patch_len
        self.hidden_size, self.patch_len = d, p
        self.input_proj = nn.Sequential(nn.Linear(2 * p, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, cfg.ts_heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.ts_layers, enable_nested_tensor=False)
        self.horizon_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.quantile_head = nn.Linear(d, 2)  # normalized median, raw spread

    def forward(self, x: Tensor, mask: Tensor, horizon: int, quantiles: Sequence[float]) -> TimesFMOutput:
        B, L = x.shape
        p, d = self.patch_len, self.hidden_size
        mu, sigma = masked_mean_std(x, mask)
        m = mask.to(x.dtype)
        pad = (-L) % p
        xn, m = F.pad((x - mu) / sigma * m, (pad, 0)), F.pad(m, (pad, 0))
        N = xn.shape[1] // p
        patches = torch.cat([xn.view(B, N, p), m.view(B, N, p)], -1)
        shallow = self.input_proj(patches) + sinusoidal(torch.arange(N, device=x.device), d)
        causal = nn.Transformer.generate_square_subsequent_mask(N, device=x.device)
        top = self.encoder(shallow, mask=causal, is_causal=True)

        steps = torch.arange(horizon, device=x.device, dtype=x.dtype)
        fut = top[:, -1:] + self.horizon_mlp(sinusoidal(steps, d))
        med, spread = self.quantile_head(fut).unbind(-1)
        z = torch.special.ndtri(torch.tensor(quantiles)).to(x)  # ndtri has no MPS kernel
        base = mu[..., None] + sigma[..., None] * (med[..., None] + z * F.softplus(spread)[..., None])
        return TimesFMOutput(shallow.repeat_interleave(p, 1)[:, pad:], fut, base)


def interp_quantiles(pred: Tensor, grid: Tensor, q: Tensor) -> Tensor:
    """Linear interpolation of native quantiles pred [..., G] on grid [G] to levels q [Q]."""
    q = q.clamp(grid[0], grid[-1])
    hi = torch.searchsorted(grid, q).clamp(1, len(grid) - 1)
    lo = hi - 1
    w = (q - grid[lo]) / (grid[hi] - grid[lo])
    return pred[..., lo] * (1 - w) + pred[..., hi] * w


class ProdTimesFM3Backbone(nn.Module):
    """Frozen official TimesFM 3.0 (`timesfm3` package, `TimesFM3Torch.decode`) + a trainable horizon adapter."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from timesfm3 import TimesFM3Torch

        self.model = TimesFM3Torch.from_pretrained(cfg.timesfm_model_id)
        self.model.requires_grad_(False)
        d = self.model.transformer_config.transformer.model_dims
        self.hidden_size, self.patch_len = d, self.model.input_patch_len
        self.register_buffer("native_q", torch.tensor(self.model.quantiles), persistent=False)
        self.horizon_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))  # trainable top adapter

    def forward(self, x: Tensor, mask: Tensor, horizon: int, quantiles: Sequence[float]) -> TimesFMOutput:
        L, p = x.shape[1], self.patch_len
        logits, aux = self.model.decode(
            x[:, None].float(), horizon=horizon, mask=~mask.bool(), return_aux_outputs=True
        )
        n_ctx = math.ceil(L / p)  # decode left-pads the context to a multiple of p
        shallow = aux["__call__:transformer_input"][:, 0, :n_ctx].repeat_interleave(p, 1)[:, -L:]
        # Patch token n_ctx - 1 + k forecasts steps [k * p, (k + 1) * p).
        top = aux["__call__:transformer_output"][:, 0, n_ctx - 1 : n_ctx - 1 + math.ceil(horizon / p)]
        steps = torch.arange(horizon, device=x.device, dtype=torch.float32)
        fut = top.repeat_interleave(p, 1)[:, :horizon] + self.horizon_mlp(sinusoidal(steps, self.hidden_size))
        native = logits[:, 0].sort(-1).values  # [B, H, 9]; sorted like TimesFM3Forecaster
        q = torch.tensor(quantiles, device=x.device, dtype=torch.float32)
        return TimesFMOutput(shallow, fut, interp_quantiles(native, self.native_q, q))


class TimesFMBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.core = ProdTimesFM3Backbone(cfg) if cfg.mode == "prod" else MockTimesFM3Backbone(cfg)
        self.hidden_size = self.core.hidden_size

    def forward(self, history_x: Tensor, history_mask: Tensor, horizon: int, quantiles: Sequence[float]) -> TimesFMOutput:
        return self.core(history_x, history_mask, horizon, quantiles)
