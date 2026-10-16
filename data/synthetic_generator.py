"""Synthetic multi-stage impulse series with entity-specific elasticity.

All events use the same text tokens. Only the true response in the history tells the model the elasticity.
Response shape: warm-up dip (ramp down) -> onset spike -> exponential decay in the event -> post-event overdraft dip.
"""
import math
from typing import Sequence, Tuple

import torch
from torch import Tensor

from data.schema import EventBatch, MultimodalTSBatch

EVENT_TEXT_IDS = (11, 12, 13, 14)


def event_factor(t: Tensor, start: Tensor, end: Tensor, pre: float, post: float, amp: Tensor) -> Tensor:
    """Multiplicative factor of one event. t: [T]; start, end, amp: [B, 1]. Returns [B, T]."""
    in_pre = (t >= start - pre) & (t < start)
    dip = in_pre * (t - (start - pre) + 1) / pre
    in_event = (t >= start) & (t < end)
    spike = torch.exp(-(t - start).clamp(min=0) / ((end - start) / 3)) * in_event  # 1 at onset, decays in the event
    in_post = (t >= end) & (t < end + post)
    overdraft = torch.exp(-(t - end).clamp(min=0) / 1.5) * in_post
    return 1 - 0.15 * amp * dip + amp * spike - 0.1 * amp * overdraft


def make_elasticity_batch(
    batch_size: int,
    hist_len: int = 192,
    horizon: int = 24,
    elasticities: Sequence[float] = (3.5, 1.0),
    n_hist_events: int = 3,
    pre_buffer: float = 3.0,
    post_buffer: float = 3.0,
    period: int = 24,
    seed: int = 0,
) -> Tuple[MultimodalTSBatch, Tensor]:
    """Returns the batch and the elasticity of each sample [B]. Samples alternate over `elasticities`."""
    g = torch.Generator().manual_seed(seed)
    B = batch_size
    amp = torch.tensor(elasticities)[torch.arange(B) % len(elasticities)][:, None]

    # One history event per segment; one future event in the forecast window.
    seg = hist_len // n_hist_events
    dur_hist = torch.randint(4, 9, (B, n_hist_events), generator=g).float()
    jitter = torch.randint(0, seg - 8 - 2 * int(pre_buffer), (B, n_hist_events), generator=g).float()
    start_hist = -hist_len + seg * torch.arange(n_hist_events).float() + pre_buffer + jitter
    start_fut = torch.randint(int(pre_buffer), horizon - 8, (B, 1), generator=g).float()
    dur_fut = torch.randint(4, 8, (B, 1), generator=g).float()
    start = torch.cat([start_hist, start_fut], 1)
    end = start + torch.cat([dur_hist, dur_fut], 1)
    publish = torch.minimum(start, torch.zeros(())) - 1 - torch.randint(0, 6, start.shape, generator=g).float()

    t = torch.arange(-hist_len, horizon).float()
    phase = torch.rand(B, 1, generator=g) * 2 * math.pi
    y = 10 + 2 * torch.sin(2 * math.pi * t / period + phase)
    for i in range(start.shape[1]):
        y = y * event_factor(t, start[:, i : i + 1], end[:, i : i + 1], pre_buffer, post_buffer, amp)
    y = y * (1 + 0.05 * torch.randn(y.shape, generator=g))

    E, T = start.shape[1], len(EVENT_TEXT_IDS)
    events = EventBatch(
        start_offset=start,
        end_offset=end,
        publish_offset=publish,
        event_mask=torch.ones(B, E, dtype=torch.bool),
        text_ids=torch.tensor(EVENT_TEXT_IDS).expand(B, E, T).clone(),
        text_mask=torch.ones(B, E, T, dtype=torch.bool),
        labels=torch.ones(B, E, dtype=torch.long),
    )
    batch = MultimodalTSBatch(
        history_x=y[:, :hist_len],
        history_mask=torch.ones(B, hist_len, dtype=torch.bool),
        events=events,
        future_y=y[:, hist_len:],
        future_mask=torch.ones(B, horizon, dtype=torch.bool),
        series_id=[f"entity-{b}-x{float(amp[b]):g}" for b in range(B)],
    )
    return batch, amp.squeeze(1)
