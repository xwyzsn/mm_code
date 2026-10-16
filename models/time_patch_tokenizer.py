"""Shallow jump features of the history and their projection to LLM soft tokens.

R(t) = (X_t - MA(X_t)) / sigma_X                                   (additive residual)
U(t) = ln((max(X_t, 0) + 0.1 sigma_X) / (max(MA(X_t), 0) + 0.1 sigma_X))  (local log ratio)
Each history event slice [start - pre, end + post) gives P <TS_HIST_PATCH> tokens; the recent window gives P <TS_RECENT_PATCH> tokens.
"""
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from configs.model_config import ModelConfig

SHALLOW_CH = 16  # TimesFM shallow patch states are compressed to this many channels


def moving_average(x: Tensor, mask: Tensor, window: int) -> Tensor:
    """Trailing masked mean over `window` steps. x, mask: [B, L]."""
    m = mask.to(x.dtype)
    cx = F.pad((x * m).cumsum(-1), (window, 0))
    cm = F.pad(m.cumsum(-1), (window, 0))
    return (cx[:, window:] - cx[:, :-window]) / (cm[:, window:] - cm[:, :-window]).clamp(min=1)


def jump_features(x: Tensor, mask: Tensor, sigma: Tensor, window: int) -> Tuple[Tensor, Tensor]:
    """Returns R, U: [B, L]. sigma: [B, 1]."""
    ma = moving_average(x, mask, window)
    r = (x - ma) / sigma
    u = torch.log((x.clamp(min=0) + 0.1 * sigma) / (ma.clamp(min=0) + 0.1 * sigma))
    m = mask.to(x.dtype)
    return r * m, u * m


class TimeToLLMProjector(nn.Module):
    """Multi-scale 1D conv + adaptive time pooling + 2-layer MLP. [N, T, C] -> [N, P, d_llm]."""

    def __init__(self, in_ch: int, d_llm: int, n_tokens: int, width: int = 64):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(in_ch, width, k, padding=k // 2) for k in (3, 5, 9))
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.mlp = nn.Sequential(nn.Linear(3 * width, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))

    def forward(self, x: Tensor) -> Tensor:
        h = x.transpose(1, 2)
        h = torch.cat([F.gelu(conv(h)) for conv in self.convs], 1)
        return self.mlp(self.pool(h).transpose(1, 2))


class TimePatchTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig, d_ts: int, d_llm: int):
        super().__init__()
        self.cfg = cfg
        self.shallow_proj = nn.Linear(d_ts, SHALLOW_CH)
        in_ch = 2 + SHALLOW_CH + 3 + 1  # [R, U, shallow, phase one-hot (pre/in/post), valid]
        self.hist_proj = TimeToLLMProjector(in_ch, d_llm, cfg.n_patch_tokens)
        self.recent_proj = TimeToLLMProjector(in_ch, d_llm, cfg.n_patch_tokens)
        self.stat_proj = nn.Sequential(nn.Linear(4, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))

    def forward(
        self,
        x: Tensor,  # [B, L]
        mask: Tensor,  # [B, L]
        mu: Tensor,  # [B, 1]
        sigma: Tensor,  # [B, 1]
        shallow: Tensor,  # [B, L, d_ts]
        hist_start: Tensor,  # [B, N]
        hist_end: Tensor,  # [B, N]
        hist_ok: Tensor,  # [B, N] bool
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns stat token [B, 1, d], history patch tokens [B, N, P, d], recent tokens [B, P, d]."""
        cfg = self.cfg
        B, L = x.shape
        r, u = jump_features(x, mask, sigma, cfg.ma_window)
        feats = torch.cat([r[..., None], u[..., None], self.shallow_proj(shallow)], -1)  # [B, L, C]

        # History event slices on a fixed grid of S samples over [start - pre, end + post).
        lo, hi = hist_start - cfg.pre_buffer, hist_end + cfg.post_buffer
        frac = (torch.arange(cfg.slice_len, device=x.device, dtype=x.dtype) + 0.5) / cfg.slice_len
        pos = lo[..., None] + (hi - lo)[..., None] * frac  # [B, N, S] offsets
        idx = pos.floor().long() + L
        idx_c = idx.clamp(0, L - 1)
        b = torch.arange(B, device=x.device)[:, None, None]
        ok = (idx >= 0) & (idx < L) & mask.bool()[b, idx_c] & hist_ok[..., None]
        s, e = hist_start[..., None], hist_end[..., None]
        phase = torch.stack([pos < s, (pos >= s) & (pos < e), pos >= e], -1).to(x.dtype)
        ev = torch.cat([feats[b, idx_c] * ok[..., None], phase, ok[..., None].to(x.dtype)], -1)
        N = ev.shape[1]
        hist_tok = self.hist_proj(ev.flatten(0, 1)).view(B, N, cfg.n_patch_tokens, -1)
        hist_tok = hist_tok * hist_ok[..., None, None]

        recent = feats[:, -cfg.recent_len :]
        rm = mask[:, -cfg.recent_len :, None].to(x.dtype)
        recent = torch.cat([recent * rm, torch.zeros(*recent.shape[:2], 3, device=x.device), rm], -1)
        recent_tok = self.recent_proj(recent)

        stat = torch.cat([mu / sigma, sigma.log(), r[:, -1:], u[:, -1:]], -1)
        return self.stat_proj(stat)[:, None], hist_tok, recent_tok
