"""Tensor contract for multimodal time-series + event batches. All offsets are relative to the forecast start t0 = 0."""
from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class EventBatch:
    """E events per sample (padded). Give `text_ids` or `sem_vecs` (or both); `labels` is optional."""

    start_offset: Tensor  # [B, E] float
    end_offset: Tensor  # [B, E] float, exclusive
    publish_offset: Tensor  # [B, E] float; the model keeps only publish_offset < 0
    event_mask: Tensor  # [B, E] bool, False = padding
    text_ids: Optional[Tensor] = None  # [B, E, T] long
    text_mask: Optional[Tensor] = None  # [B, E, T] bool
    sem_vecs: Optional[Tensor] = None  # [B, E, K, D] float, K semantic vectors per event
    labels: Optional[Tensor] = None  # [B, E] long


@dataclass
class MultimodalTSBatch:
    history_x: Tensor  # [B, L_hist]
    history_mask: Tensor  # [B, L_hist] bool
    events: EventBatch
    future_y: Optional[Tensor] = None  # [B, H], training only
    future_mask: Optional[Tensor] = None  # [B, H] bool
