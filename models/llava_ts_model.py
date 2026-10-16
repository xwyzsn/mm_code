"""Scheme 1 — LLaVA-TS end-to-end model: TimesFM base forecast, interleaved Qwen prefill, quantile FiLM."""
from dataclasses import dataclass
from typing import Optional, Sequence

from torch import Tensor, nn

from backbones.qwen_backbone import QwenBackbone
from backbones.timesfm_backbone import TimesFMBackbone, masked_mean_std
from configs.model_config import ModelConfig
from data.schema import MultimodalTSBatch
from models.distributional_film import DistributionalFiLM, event_lifecycle
from models.interleaved_context import InterleavedContextBuilder, select_events, take
from models.time_patch_tokenizer import TimePatchTokenizer


@dataclass
class LlavaTSOutput:
    quantiles: Tensor  # [B, H, |Q|]
    base_quantiles: Tensor  # [B, H, |Q|] TimesFM output
    u_median: Tensor  # [B, H]
    envelope: Tensor  # [B, H] m(t)
    mu: Tensor  # [B, 1]
    sigma: Tensor  # [B, 1]
    median_idx: int


class LlavaTSModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qwen = QwenBackbone(cfg)
        self.timesfm = TimesFMBackbone(cfg)
        d_llm, d_ts = self.qwen.hidden_size, self.timesfm.hidden_size
        self.tokenizer = TimePatchTokenizer(cfg, d_ts, d_llm)
        self.context = InterleavedContextBuilder(cfg, d_llm)
        self.film = DistributionalFiLM(cfg, d_ts, d_llm)

    def forward(
        self, batch: MultimodalTSBatch, horizon: Optional[int] = None, quantiles: Optional[Sequence[float]] = None
    ) -> LlavaTSOutput:
        """H defaults to future_y's length, else cfg.horizon. Q defaults to cfg.quantiles and must contain 0.5."""
        cfg, ev = self.cfg, batch.events
        H = horizon or (batch.future_y.shape[1] if batch.future_y is not None else cfg.horizon)
        Q = tuple(quantiles or cfg.quantiles)
        x, mask = batch.history_x, batch.history_mask
        mu, sigma = masked_mean_std(x, mask)

        ts = self.timesfm(x, mask, H, Q)
        hist_idx, hist_ok, fut_idx, fut_ok = select_events(ev, x.shape[1], H, cfg)
        stat_tok, hist_tok, recent_tok = self.tokenizer(
            x, mask, mu, sigma, ts.shallow_hidden, take(ev.start_offset, hist_idx), take(ev.end_offset, hist_idx), hist_ok
        )
        embeds, attn_mask = self.context(
            ev, hist_idx, hist_ok, fut_idx, fut_ok, stat_tok, hist_tok, recent_tok, H, self.qwen.embed_tokens
        )
        z_qwen = self.qwen(embeds, attn_mask)[:, -H:]

        env, _ = event_lifecycle(take(ev.start_offset, fut_idx), take(ev.end_offset, fut_idx), fut_ok, H, cfg)
        m = env.amax(1)
        median_idx = Q.index(0.5)
        film = self.film(ts.fut_hidden, z_qwen, ts.base_quantiles, m, sigma, median_idx)
        return LlavaTSOutput(film.quantiles, ts.base_quantiles, film.u_median, m, mu, sigma, median_idx)
