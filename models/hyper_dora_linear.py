"""HyperDoRALinear: frozen W0 + shared low-rank basis A, B with per-sample singular values s(E) and row gains dm(E).

y = (1 + m_act * tanh(dm(E))) * y0 + m_act * ((h A^T) * s(E)) B^T,   y0 = h W0^T + b0
m_act = 0 gives y = y0 exactly (zero drift without events).
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch
from torch import Tensor, nn


@dataclass
class HyperCondition:
    """Conditioning for one forward pass of the TimesFM stack."""

    c_event: Tensor  # [B, d_llm] sequence-level event code C_event
    m_steps: Tensor  # [B, H] event envelope m(t) of the forecast steps
    n_ctx: int  # number of context patch tokens
    patch_len: int
    span: int  # steps that one forecasting token predicts

    def token_activation(self, n_tokens: int) -> Tensor:
        """m_act [B, n_tokens]: token n_ctx - 1 + k forecasts steps [k p, k p + span); the max m(t) of those steps."""
        dev = self.m_steps.device
        k = torch.arange(n_tokens, device=dev) - (self.n_ctx - 1)
        t = torch.arange(self.m_steps.shape[1], device=dev)
        covers = (t >= k[:, None] * self.patch_len) & (t < k[:, None] * self.patch_len + self.span)
        return (self.m_steps[:, None, :] * covers).amax(-1)


class HyperDoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, d_cond: int):
        super().__init__()
        self.base = base.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.rank = rank
        self.A = nn.Parameter(torch.empty(rank, d_in))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.randn(d_out, rank) * 0.02)  # not zero: the hypernetwork gets a gradient at step 1
        self.proj = nn.Linear(d_cond, rank + d_out)  # HyperParamProjector -> [s(E), dm(E)]
        nn.init.normal_(self.proj.weight, std=1e-3)
        nn.init.zeros_(self.proj.bias)
        self.cond: Optional[HyperCondition] = None

    def hyper_params(self, c_event: Tensor):
        """Returns s(E) [B, r] and dm(E) [B, d_out]."""
        return self.proj(c_event).split([self.rank, self.B.shape[0]], -1)

    def forward(self, x: Tensor) -> Tensor:
        y0 = self.base(x)
        if self.cond is None:
            return y0
        c = self.cond
        b = c.c_event.shape[0]
        h = x.reshape(b, -1, x.shape[-1])  # TimesFM layouts with one variate flatten to [B, tokens, d]
        m = c.token_activation(h.shape[1])[..., None].to(x.dtype)
        s, dm = self.hyper_params(c.c_event.to(x.dtype))
        dy = ((h @ self.A.T) * s[:, None]) @ self.B.T
        y = (1 + m * torch.tanh(dm)[:, None]) * y0.reshape(b, -1, y0.shape[-1]) + m * dy
        return y.reshape(y0.shape)


def inject_hyper_dora(layers: nn.ModuleList, n_last: int, rank: int, d_cond: int) -> List[HyperDoRALinear]:
    """Replaces every nn.Linear in the last n_last layers with HyperDoRALinear. Returns the new modules."""
    new = []
    for layer in layers[-n_last:]:
        for name, mod in list(layer.named_modules()):
            if isinstance(mod, nn.Linear):
                parent, _, child = name.rpartition(".")
                hyper = HyperDoRALinear(mod, rank, d_cond).to(mod.weight.device)
                setattr(layer.get_submodule(parent) if parent else layer, child, hyper)
                new.append(hyper)
    return new


@contextmanager
def hyper_condition(modules: List[HyperDoRALinear], cond: Optional[HyperCondition]) -> Iterator[None]:
    for mod in modules:
        mod.cond = cond
    try:
        yield
    finally:
        for mod in modules:
            mod.cond = None
