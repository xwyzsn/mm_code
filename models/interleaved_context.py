"""Causal event selection and the interleaved multimodal context for one Qwen prefill.

Sequence: [stat, (EventSem(e_hist_1), TSPatch(e_hist_1)), ..., TSRecent, EventSem(e_fut_1), ..., Q_hor_1..H]
Padding and unpublished events are masked tokens (attention_mask = 0), so their values cannot change the output.
"""
from typing import Callable, Tuple

import math
import torch
from torch import Tensor, nn

from backbones.timesfm_backbone import sinusoidal
from configs.model_config import ModelConfig
from data.schema import EventBatch
from models.distributional_film import event_lifecycle


def take(t: Tensor, idx: Tensor) -> Tensor:
    """t: [B, E, ...], idx: [B, N] -> [B, N, ...]."""
    return t[torch.arange(t.shape[0], device=t.device)[:, None], idx]


def _pick(score: Tensor, valid: Tensor, n: int, start: Tensor) -> Tuple[Tensor, Tensor]:
    """Top-n valid events by score, then in time order. Returns idx [B, n], ok [B, n]."""
    k = min(n, score.shape[1])
    top, idx = score.masked_fill(~valid, float("-inf")).topk(k, dim=1)
    ok = torch.isfinite(top)
    order = take(start, idx).masked_fill(~ok, float("inf")).argsort(1)
    idx, ok = idx.gather(1, order), ok.gather(1, order)
    if k < n:
        idx = torch.cat([idx, idx.new_zeros(idx.shape[0], n - k)], 1)
        ok = torch.cat([ok, ok.new_zeros(ok.shape[0], n - k)], 1)
    return idx, ok


def select_events(ev: EventBatch, hist_len: int, horizon: int, cfg: ModelConfig) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Keeps only published events (publish_offset < 0).

    E_hist: start in [-L, 0), the N_hist most recent.
    E_fut:  [start - pre, end + post] meets [0, H), the N_fut earliest.
    An event that starts before t0 and lasts into the forecast window is in both sets.
    """
    s, e = ev.start_offset, ev.end_offset
    visible = ev.event_mask & (ev.publish_offset < 0)
    is_hist = visible & (s >= -hist_len) & (s < 0)
    is_fut = visible & (s - cfg.pre_buffer < horizon) & (e + cfg.post_buffer > 0)
    hist_idx, hist_ok = _pick(s, is_hist, cfg.n_hist_events, s)
    fut_idx, fut_ok = _pick(-s, is_fut, cfg.n_fut_events, s)
    return hist_idx, hist_ok, fut_idx, fut_ok


def _mlp(d_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))


class EventProjector(nn.Module):
    """Event -> semantic tokens: [time token, label token?, text tokens?, vector tokens?]."""

    def __init__(self, cfg: ModelConfig, d_llm: int):
        super().__init__()
        self.time_mlp = _mlp(4 * 16, d_llm)
        self.label_emb = nn.Embedding(cfg.n_event_labels, d_llm)
        self.vec_mlp = _mlp(cfg.event_vec_dim, d_llm)

    def forward(self, ev: EventBatch, idx: Tensor, ok: Tensor, embed_tokens: Callable[[Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
        """Returns tokens [B, N, Te, d] and token mask [B, N, Te]."""
        s, e, p = take(ev.start_offset, idx), take(ev.end_offset, idx), take(ev.publish_offset, idx)
        times = torch.stack([s, e, e - s, p], -1).masked_fill(~ok[..., None], 0.0)
        tokens = [self.time_mlp(sinusoidal(times, 16).flatten(-2))[:, :, None]]
        masks = [ok[..., None]]
        if ev.labels is not None:
            tokens.append(self.label_emb(take(ev.labels, idx).masked_fill(~ok, 0))[:, :, None])
            masks.append(ok[..., None])
        if ev.text_ids is not None:
            tokens.append(embed_tokens(take(ev.text_ids, idx).masked_fill(~ok[..., None], 0)))
            masks.append(take(ev.text_mask, idx) & ok[..., None])
        if ev.sem_vecs is not None:
            tokens.append(self.vec_mlp(take(ev.sem_vecs, idx).masked_fill(~ok[..., None, None], 0.0)))
            masks.append(ok[..., None].expand(-1, -1, ev.sem_vecs.shape[2]))
        return torch.cat(tokens, 2), torch.cat(masks, 2)


class HorizonQueries(nn.Module):
    """Q_hor [B, H, d]: learnable base + step / intraday encoding + lifecycle encoding of each future event."""

    def __init__(self, cfg: ModelConfig, d_llm: int):
        super().__init__()
        self.cfg = cfg
        self.q0 = nn.Parameter(torch.randn(d_llm) * 0.02)
        self.step_mlp = _mlp(16 + 2, d_llm)
        self.life_mlp = _mlp(8 + 8 + 4 + 1, d_llm)

    def forward(self, fut_start: Tensor, fut_end: Tensor, fut_ok: Tensor, horizon: int) -> Tensor:
        t = torch.arange(horizon, device=fut_start.device, dtype=fut_start.dtype)
        ang = 2 * math.pi * t / self.cfg.period
        step = self.step_mlp(torch.cat([sinusoidal(t, 16), ang.sin()[:, None], ang.cos()[:, None]], -1))  # [H, d]
        env, phase = event_lifecycle(fut_start, fut_end, fut_ok, horizon, self.cfg)
        s, e = fut_start.masked_fill(~fut_ok, 0.0)[..., None], fut_end.masked_fill(~fut_ok, 0.0)[..., None]
        life = torch.cat([sinusoidal(t - s, 8), sinusoidal(t - e, 8), phase, env[..., None]], -1)  # [B, N, H, 21]
        life = (self.life_mlp(life) * fut_ok[..., None, None]).sum(1)
        return self.q0 + step + life


class InterleavedContextBuilder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_llm: int):
        super().__init__()
        self.events = EventProjector(cfg, d_llm)
        self.queries = HorizonQueries(cfg, d_llm)

    def forward(
        self,
        ev: EventBatch,
        hist_idx: Tensor,
        hist_ok: Tensor,
        fut_idx: Tensor,
        fut_ok: Tensor,
        stat_tok: Tensor,  # [B, 1, d]
        hist_tok: Tensor,  # [B, N_hist, P, d]
        recent_tok: Tensor,  # [B, P, d]
        horizon: int,
        embed_tokens: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Returns inputs_embeds [B, S, d] and attention_mask [B, S]. The last H positions are the queries."""
        hist_sem, hist_sem_mask = self.events(ev, hist_idx, hist_ok, embed_tokens)
        fut_sem, fut_sem_mask = self.events(ev, fut_idx, fut_ok, embed_tokens)
        q_hor = self.queries(take(ev.start_offset, fut_idx), take(ev.end_offset, fut_idx), fut_ok, horizon)

        B, P = recent_tok.shape[:2]
        hist_pairs = torch.cat([hist_sem, hist_tok], 2).flatten(1, 2)
        hist_mask = torch.cat([hist_sem_mask, hist_ok[..., None].expand(-1, -1, P)], 2).flatten(1, 2)
        ones = lambda n: torch.ones(B, n, dtype=torch.bool, device=recent_tok.device)  # noqa: E731
        embeds = torch.cat([stat_tok, hist_pairs, recent_tok, fut_sem.flatten(1, 2), q_hor], 1)
        mask = torch.cat([ones(1), hist_mask, ones(P), fut_sem_mask.flatten(1, 2), ones(horizon)], 1)
        return embeds * mask[..., None], mask
