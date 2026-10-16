"""TimesFM 3.0 with HyperDoRALinear in the last N layers. Gives the top states H_top(t) and the base quantiles.

prod:      official TimesFM 3.0 (`timesfm3.TimesFM3Torch`, frozen). `decode` is called without its no_grad wrapper,
           so that gradients reach the HyperDoRA parameters.
mac_debug: MockTimesFM3Core, a small causal patch Transformer with the TimesFM 3 module names
           (transformer_stack.layers[i].{seq_attn,var_attn}.{query,key,value,out}_proj, ff0, ff1).
Both: any history length L, any horizon H, any quantile set Q, one variate.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from configs.model_config import ModelConfig
from models.hyper_dora_linear import HyperCondition, hyper_condition, inject_hyper_dora


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


def interp_quantiles(pred: Tensor, grid: Tensor, q: Tensor) -> Tensor:
    """Linear interpolation of native quantiles pred [..., G] on grid [G] to levels q [Q]."""
    q = q.clamp(grid[0], grid[-1])
    hi = torch.searchsorted(grid, q).clamp(1, len(grid) - 1)
    lo = hi - 1
    w = (q - grid[lo]) / (grid[hi] - grid[lo])
    return pred[..., lo] * (1 - w) + pred[..., hi] * w


class MockAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.query_proj, self.key_proj, self.value_proj, self.out_proj = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x: Tensor, causal: bool) -> Tensor:
        b, n, d = x.shape
        q, k, v = (p(x).view(b, n, self.n_heads, -1).transpose(1, 2) for p in (self.query_proj, self.key_proj, self.value_proj))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.out_proj(o.transpose(1, 2).reshape(b, n, d))


class MockMixingLayer(nn.Module):
    """Sequence attention over patches, variate attention over the single variate, ReLU feed-forward."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.pre_seq_attn_ln, self.pre_var_attn_ln, self.pre_ff_ln = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(d)
        self.seq_attn, self.var_attn = MockAttention(d, n_heads), MockAttention(d, n_heads)
        self.ff0, self.ff1 = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        x = x + self.seq_attn(self.pre_seq_attn_ln(x), causal=True)
        x = x + self.var_attn(self.pre_var_attn_ln(x).reshape(b * n, 1, d), causal=False).view(b, n, d)  # TimesFM layout
        return x + self.ff1(F.relu(self.ff0(self.pre_ff_ln(x))))


class MockTimesFM3Core(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, p = cfg.d_ts, cfg.ts_patch_len
        self.hidden_size, self.patch_len, self.span = d, p, p
        self.pre_transformer_resblock = nn.Sequential(nn.Linear(2 * p, d), nn.SiLU(), nn.Linear(d, d))
        self.transformer_stack = nn.Module()
        self.transformer_stack.layers = nn.ModuleList(MockMixingLayer(d, cfg.ts_heads) for _ in range(cfg.ts_layers))
        self.output_head = nn.Linear(d, 2 * p)  # per step of the next patch: normalized median, raw spread

    @property
    def layers(self) -> nn.ModuleList:
        return self.transformer_stack.layers

    def forward(self, x: Tensor, mask: Tensor, horizon: int, quantiles: Sequence[float]) -> Tuple[Tensor, Tensor]:
        B, L = x.shape
        p, d = self.patch_len, self.hidden_size
        mu, sigma = masked_mean_std(x, mask)
        m = mask.to(x.dtype)
        pad, n_hor = (-L) % p, math.ceil(horizon / p)
        xn = F.pad((x - mu) / sigma * m, (pad, n_hor * p))  # horizon patches: value 0, valid flag 0
        m = F.pad(m, (pad, n_hor * p))
        N = xn.shape[1] // p
        n_ctx = N - n_hor
        h = self.pre_transformer_resblock(torch.cat([xn.view(B, N, p), m.view(B, N, p)], -1))
        h = h + sinusoidal(torch.arange(N, device=x.device), d)
        for layer in self.layers:
            h = layer(h)

        tok = h[:, n_ctx - 1 : n_ctx - 1 + n_hor]  # token n_ctx - 1 + k forecasts patch k
        med, spread = self.output_head(tok).view(B, n_hor * p, 2)[:, :horizon].unbind(-1)
        z = torch.special.ndtri(torch.tensor(quantiles)).to(x)  # ndtri has no MPS kernel
        base = mu[..., None] + sigma[..., None] * (med[..., None] + z * F.softplus(spread)[..., None])
        return tok.repeat_interleave(p, 1)[:, :horizon], base


class ProdTimesFM3Core(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from timesfm3 import TimesFM3Torch

        self.model = TimesFM3Torch.from_pretrained(cfg.timesfm_model_id)
        self.model.requires_grad_(False)
        assert self.model.use_stitching, "token-to-step mapping assumes use_stitching=True"
        self.hidden_size = self.model.transformer_config.transformer.model_dims
        self.patch_len, self.span = self.model.input_patch_len, self.model._stitching_extract_len
        self.register_buffer("native_q", torch.tensor(self.model.quantiles), persistent=False)

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.transformer_stack.layers

    def forward(self, x: Tensor, mask: Tensor, horizon: int, quantiles: Sequence[float]) -> Tuple[Tensor, Tensor]:
        L, p = x.shape[1], self.patch_len
        decode = type(self.model).decode.__wrapped__  # without @torch.no_grad
        logits, aux = decode(self.model, x[:, None].float(), horizon=horizon, mask=~mask.bool(), return_aux_outputs=True)
        n_ctx = math.ceil(L / p)  # decode left-pads the context to a multiple of p
        tok = aux["__call__:transformer_output"][:, 0, n_ctx - 1 : n_ctx - 1 + math.ceil(horizon / p)]
        native = logits[:, 0].sort(-1).values  # [B, H, 9]; sorted like TimesFM3Forecaster
        q = torch.tensor(quantiles, device=x.device, dtype=torch.float32)
        return tok.repeat_interleave(p, 1)[:, :horizon], interp_quantiles(native, self.native_q, q)


@dataclass
class TimesFMOutput:
    top_hidden: Tensor  # [B, H, d_ts] H_top(t): top state of the token that forecasts step t
    base_quantiles: Tensor  # [B, H, |Q|] original scale


class ModularTimesFM3Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig, d_cond: int):
        super().__init__()
        self.core = ProdTimesFM3Core(cfg) if cfg.mode == "prod" else MockTimesFM3Core(cfg)
        self.hidden_size = self.core.hidden_size
        self.hyper_layers = inject_hyper_dora(self.core.layers, cfg.n_last_layers, cfg.hyper_rank, d_cond)  # plain list

    def forward(
        self,
        history_x: Tensor,
        history_mask: Tensor,
        horizon: int,
        quantiles: Sequence[float],
        c_event: Optional[Tensor] = None,  # [B, d_cond]; None = native TimesFM
        m_steps: Optional[Tensor] = None,  # [B, H]
    ) -> TimesFMOutput:
        cond = None
        if c_event is not None:
            n_ctx = math.ceil(history_x.shape[1] / self.core.patch_len)
            cond = HyperCondition(c_event, m_steps, n_ctx, self.core.patch_len, self.core.span)
        with hyper_condition(self.hyper_layers, cond):
            return TimesFMOutput(*self.core(history_x, history_mask, horizon, quantiles))
