"""Generic dynamics loss in the linear normalized space Y~ = (Y - mu_X) / sigma_X (no arcsinh).

L_total = L_pinball + lambda_1 L_diff1 + lambda_2 L_diff2 + lambda_3 L_cf + lambda_reg (||s(E)||^2 + ||dm(E)||^2)
L_pinball: pinball over all quantiles, linear for |e| <= 15 (sigma units), log growth above (soft transition).
L_diff1:   pinball on first differences (all quantiles).
L_diff2:   Smooth-L1 on the median second difference.
L_cf:      Smooth-L1 between u_0.5 and the clipped true log ratio u*, where m(t) > 0.
"""
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from configs.model_config import ModelConfig
from models.hyper_timesfm_model import HyperTSOutput


def robust_pinball(pred: Tensor, target: Tensor, q: Tensor, delta: float) -> Tensor:
    """pred [B, T, Q], target [B, T]. Element loss [B, T, Q]; value and slope are continuous at |e| = delta."""
    e = target[..., None] - pred
    a = e.abs()
    rho = torch.where(a <= delta, a, delta * (1 + torch.log(a.clamp(min=delta) / delta)))
    return rho * torch.where(e >= 0, q, 1 - q)


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    while mask.dim() < x.dim():
        mask = mask[..., None]
    mask = mask.expand_as(x).to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1)


def dynamics_loss(
    out: HyperTSOutput, future_y: Tensor, future_mask: Tensor, quantiles: Sequence[float], cfg: ModelConfig
) -> Dict[str, Tensor]:
    q = torch.tensor(quantiles, device=future_y.device, dtype=future_y.dtype)
    mu, sigma, k = out.mu, out.sigma, out.median_idx
    y = (future_y - mu) / sigma
    pred = (out.quantiles - mu[..., None]) / sigma[..., None]
    mask = future_mask.bool()

    l_quantile = _masked_mean(robust_pinball(pred, y, q, cfg.huber_sigma), mask)

    dy, dpred = y.diff(dim=1), pred.diff(dim=1)
    dmask = mask[:, 1:] & mask[:, :-1]
    l_diff1 = _masked_mean(robust_pinball(dpred, dy, q, cfg.huber_sigma), dmask)
    d2mask = dmask[:, 1:] & dmask[:, :-1]
    l_diff2 = _masked_mean(F.smooth_l1_loss(dpred[..., k].diff(dim=1), dy.diff(dim=1), reduction="none"), d2mask)

    base_med = out.base_quantiles[..., k].detach()
    u_star = torch.log((future_y.clamp(min=0) + 0.1 * sigma) / (base_med.clamp(min=0) + 0.1 * sigma))
    u_star = u_star.clamp(-cfg.alpha_max, cfg.alpha_max)
    l_cf = _masked_mean(F.smooth_l1_loss(out.u_median, u_star, reduction="none"), mask & (out.envelope > 0))

    total = (
        l_quantile + cfg.lambda_diff1 * l_diff1 + cfg.lambda_diff2 * l_diff2 + cfg.lambda_cf * l_cf
        + cfg.lambda_reg * out.hyper_reg
    )
    return {"total": total, "quantile": l_quantile, "diff1": l_diff1, "diff2": l_diff2, "cf": l_cf, "reg": out.hyper_reg}
