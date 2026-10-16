"""Scheme 2 — Event-Conditioned HyperLoRA: Qwen hypernetwork -> HyperDoRA TimesFM 3.0 -> dynamic quantile head."""
from dataclasses import dataclass
from typing import Optional, Sequence

from torch import Tensor, nn

from backbones.qwen_hyper_backbone import EventContextCache, QwenHyperBackbone, event_lifecycle, select_events, take
from backbones.timesfm_modular import ModularTimesFM3Backbone, masked_mean_std
from configs.model_config import ModelConfig
from data.schema import MultimodalTSBatch
from models.dynamic_quantile_head import DynamicQuantileHead


@dataclass
class HyperTSOutput:
    quantiles: Tensor  # [B, H, |Q|]
    base_quantiles: Tensor  # [B, H, |Q|] TimesFM with HyperDoRA (= native TimesFM where m = 0)
    u_median: Tensor  # [B, H]
    envelope: Tensor  # [B, H] m(t)
    mu: Tensor  # [B, 1]
    sigma: Tensor  # [B, 1]
    median_idx: int
    hyper_reg: Tensor  # mean over samples of sum over HyperDoRA layers of ||s(E)||^2 + ||dm(E)||^2


class HyperTimesFMModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qwen = QwenHyperBackbone(cfg)
        self.timesfm = ModularTimesFM3Backbone(cfg, d_cond=self.qwen.hidden_size)
        self.head = DynamicQuantileHead(cfg, self.timesfm.hidden_size, self.qwen.hidden_size)

    def forward(
        self,
        batch: MultimodalTSBatch,
        horizon: Optional[int] = None,
        quantiles: Optional[Sequence[float]] = None,
        cache: Optional[EventContextCache] = None,
    ) -> HyperTSOutput:
        """H defaults to future_y's length, else cfg.horizon. Q defaults to cfg.quantiles and must contain 0.5."""
        cfg, ev = self.cfg, batch.events
        H = horizon or (batch.future_y.shape[1] if batch.future_y is not None else cfg.horizon)
        Q = tuple(quantiles or cfg.quantiles)
        x, mask = batch.history_x, batch.history_mask
        mu, sigma = masked_mean_std(x, mask)

        sel = select_events(ev, x.shape[1], H, cfg)
        fut_idx, fut_ok = sel[2], sel[3]
        env, _ = event_lifecycle(take(ev.start_offset, fut_idx), take(ev.end_offset, fut_idx), fut_ok, H, cfg)
        m = env.amax(1)

        c_event, p_step = self.qwen(batch, mu, sigma, sel, H, cache)
        ts = self.timesfm(x, mask, H, Q, c_event=c_event, m_steps=m)
        median_idx = Q.index(0.5)
        head = self.head(p_step, ts.top_hidden, ts.base_quantiles, m, sigma, median_idx)

        reg = sum(sum(p.pow(2).sum(-1) for p in layer.hyper_params(c_event)) for layer in self.timesfm.hyper_layers)
        return HyperTSOutput(head.quantiles, ts.base_quantiles, head.u_median, m, mu, sigma, median_idx, reg.mean())
