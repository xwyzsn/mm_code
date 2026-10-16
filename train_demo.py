"""Minimal LLaVA-TS training loop (Lightning) on synthetic impulse data, plus an in-context elasticity report.

All events have the same text. High (3.5x) and low (1.0x) elasticity series differ only in their history response,
so the model can tell them apart only through the interleaved <TS_HIST_PATCH> tokens.
Run: .venv/bin/python train_demo.py --steps 50
"""
import argparse
from typing import Dict, Optional, Tuple

import lightning as L
import torch
from lightning.pytorch.utilities import move_data_to_device
from torch.utils.data import DataLoader, IterableDataset

from configs.model_config import ModelConfig
from data.synthetic_generator import make_elasticity_batch
from losses.dynamics_loss import dynamics_loss
from models.llava_ts_model import LlavaTSModel

EVAL_SEED = 10_000  # held-out batch, never used for training


class SyntheticStream(IterableDataset):
    """A new synthetic batch at each step."""

    def __init__(self, batch_size: int, horizon: int, seed: int):
        self.batch_size, self.horizon, self.seed = batch_size, horizon, seed

    def __iter__(self):
        step = 0
        while True:
            yield make_elasticity_batch(self.batch_size, horizon=self.horizon, seed=self.seed + step)[0]
            step += 1


class LlavaTSLightning(L.LightningModule):
    def __init__(self, cfg: ModelConfig, lr: float):
        super().__init__()
        self.cfg, self.lr = cfg, lr
        self.model = LlavaTSModel(cfg)

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        losses = dynamics_loss(out, batch.future_y, batch.future_mask, self.cfg.quantiles, self.cfg)
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, batch_size=batch.history_x.shape[0])
        return losses["total"]

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)


@torch.no_grad()
def elasticity_report(model: LlavaTSModel, cfg: ModelConfig, batch_size: int = 64) -> Dict[str, float]:
    """Metrics on the held-out batch. hi = 3.5x series, lo = 1.0x series; the future event is the last event."""
    was_training = model.training
    model.eval()
    batch, amp = make_elasticity_batch(batch_size, horizon=cfg.horizon, seed=EVAL_SEED)
    batch = move_data_to_device(batch, next(model.parameters()).device)
    out = model(batch)
    total = dynamics_loss(out, batch.future_y, batch.future_mask, cfg.quantiles, cfg)["total"]

    k = out.median_idx
    s, e = batch.events.start_offset[:, -1:], batch.events.end_offset[:, -1:]
    t = torch.arange(cfg.horizon, device=s.device)
    in_event, warmup = (t >= s) & (t < e), (t >= s - cfg.pre_buffer) & (t < s)
    hi, lo = (amp == amp.max()).to(s.device), (amp == amp.min()).to(s.device)
    y, base = out.quantiles, out.base_quantiles
    shift = (y[..., k] - base[..., k]) / out.sigma  # median shift in sigma units
    width_ratio = (y[..., -1] - y[..., 0]) / (base[..., -1] - base[..., 0])
    abs_err, base_abs_err = (y[..., k] - batch.future_y).abs(), (base[..., k] - batch.future_y).abs()
    model.train(was_training)
    return {
        "loss": total.item(),
        "u_event_hi": out.u_median[hi][in_event[hi]].mean().item(),
        "u_event_lo": out.u_median[lo][in_event[lo]].mean().item(),
        "warmup_shift_hi": shift[hi][warmup[hi]].mean().item(),
        "width_ratio_event_hi": width_ratio[hi][in_event[hi]].mean().item(),
        "event_mae_ratio_hi": (abs_err[hi][in_event[hi]].mean() / base_abs_err[hi][in_event[hi]].mean()).item(),
    }


def train(
    cfg: Optional[ModelConfig] = None,
    steps: int = 50,
    batch_size: int = 32,
    lr: float = 4e-3,
    seed: int = 0,
    accelerator: str = "auto",
) -> Tuple[LlavaTSModel, Dict[str, float], Dict[str, float]]:
    L.seed_everything(seed, verbose=False)
    cfg = cfg or ModelConfig()
    module = LlavaTSLightning(cfg, lr)
    before = elasticity_report(module.model, cfg)
    trainer = L.Trainer(
        max_steps=steps,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, DataLoader(SyntheticStream(batch_size, cfg.horizon, seed), batch_size=None))
    return module.model, before, elasticity_report(module.model, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args()
    _, before, after = train(steps=args.steps, accelerator=args.accelerator)
    for name in before:
        print(f"{name:>22}: {before[name]:9.4f} -> {after[name]:9.4f}")
