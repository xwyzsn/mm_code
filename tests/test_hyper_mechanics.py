"""HyperLoRA mechanics tests (mac_debug mode, CPU). The prod test runs only if the TimesFM 3.0 checkpoint is local."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbones.qwen_hyper_backbone import EventContextCache  # noqa: E402
from backbones.timesfm_modular import ModularTimesFM3Backbone  # noqa: E402
from configs.model_config import ModelConfig  # noqa: E402
from data.synthetic_generator import make_elasticity_batch  # noqa: E402
from losses.dynamics_loss import dynamics_loss  # noqa: E402
from models.hyper_dora_linear import HyperCondition, HyperDoRALinear, hyper_condition  # noqa: E402
from models.hyper_timesfm_model import HyperTimesFMModel  # noqa: E402
from train_demo import train  # noqa: E402

TIMESFM_SNAPSHOTS = Path.home() / ".cache/huggingface/hub/models--google--timesfm-3.0-pytorch/snapshots"


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig()


def new_model(cfg):
    torch.manual_seed(0)
    return HyperTimesFMModel(cfg)


def has_grad(p):
    return p.grad is not None and not p.grad.isnan().any() and p.grad.abs().sum() > 0


@pytest.mark.parametrize("horizon", [12, 24, 96])
def test_zero_event_strict_weight_identity(cfg, horizon):
    model = new_model(cfg)
    batch, _ = make_elasticity_batch(4, horizon=horizon)
    batch.events.event_mask[:, -1] = False  # remove the only future event; history events stay
    out = model(batch)
    native = model.timesfm(batch.history_x, batch.history_mask, horizon, cfg.quantiles)  # no HyperDoRA condition
    assert out.quantiles.shape == (4, horizon, len(cfg.quantiles))
    assert out.envelope.max().item() == 0.0
    assert (out.quantiles - native.base_quantiles).abs().max().item() == 0.0


def per_token_condition(c_event, m_tokens):
    """n_ctx = patch = span = 1: token j gets m_act = m_tokens[:, j]."""
    return HyperCondition(c_event, m_tokens, n_ctx=1, patch_len=1, span=1)


def test_batched_hyper_dora_equivalence():
    torch.manual_seed(0)
    B, T, d_in, d_out, r, d_cond = 3, 5, 12, 7, 4, 6
    layer = HyperDoRALinear(nn.Linear(d_in, d_out), r, d_cond).double()
    nn.init.normal_(layer.proj.weight, std=0.5)  # make s(E) and dm(E) clearly non-zero
    h, c = torch.randn(B, T, d_in, dtype=torch.float64), torch.randn(B, d_cond, dtype=torch.float64)
    W0, b0 = layer.base.weight, layer.base.bias
    s, dm = layer.hyper_params(c)

    # task.md form: m_act = 1, dm = 0  ->  W_eff = W0 + B Diag(s_b) A
    with torch.no_grad():
        layer.proj.weight[r:] = 0
        layer.proj.bias[r:] = 0
    s, dm = layer.hyper_params(c)
    with hyper_condition([layer], per_token_condition(c, torch.ones(B, T, dtype=torch.float64))):
        y = layer(h)
    for b in range(B):
        w_eff = W0 + layer.B @ torch.diag(s[b]) @ layer.A
        assert (y[b] - (h[b] @ w_eff.T + b0)).abs().max().item() < 1e-6

    # Full form with row gains and a per-token envelope: W_eff(b, t) = (1 + m tanh dm_b) W0 + m B Diag(s_b) A
    nn.init.normal_(layer.proj.weight, std=0.5)
    s, dm = layer.hyper_params(c)
    m = torch.rand(B, T, dtype=torch.float64)
    with hyper_condition([layer], per_token_condition(c, m)):
        y = layer(h)
    for b in range(B):
        for t in range(T):
            gain = 1 + m[b, t] * torch.tanh(dm[b])
            w_eff = gain[:, None] * W0 + m[b, t] * layer.B @ torch.diag(s[b]) @ layer.A
            assert (y[b, t] - (h[b, t] @ w_eff.T + gain * b0)).abs().max().item() < 1e-6


def check_backbone_identity(backbone, batch, quantiles, d_cond, horizon, event_steps):
    """m = 0 -> exact native output; m > 0 -> different output. Returns the active output."""
    x, mask = batch.history_x, batch.history_mask
    B = x.shape[0]
    c = torch.randn(B, d_cond)
    native = backbone(x, mask, horizon, quantiles)
    zero = backbone(x, mask, horizon, quantiles, c_event=c, m_steps=torch.zeros(B, horizon))
    assert (zero.base_quantiles - native.base_quantiles).abs().max().item() == 0.0
    assert (zero.top_hidden - native.top_hidden).abs().max().item() == 0.0

    m = torch.zeros(B, horizon)
    m[:, event_steps] = 1.0
    active = backbone(x, mask, horizon, quantiles, c_event=c, m_steps=m)
    assert (active.base_quantiles - native.base_quantiles).abs().max().item() > 0
    return native, active


def test_modular_backbone_zero_event_identity():
    torch.manual_seed(0)
    cfg = ModelConfig()
    backbone = ModularTimesFM3Backbone(cfg, d_cond=16)
    assert len(backbone.hyper_layers) == 10 * cfg.n_last_layers
    batch, _ = make_elasticity_batch(4)
    native, active = check_backbone_identity(backbone, batch, cfg.quantiles, 16, 24, range(16, 24))
    # Mock span = patch = 8: tokens that forecast steps 0..15 have m_act = 0 and see only unchanged tokens.
    assert (active.base_quantiles[:, :16] - native.base_quantiles[:, :16]).abs().max().item() == 0.0


@pytest.mark.skipif(not TIMESFM_SNAPSHOTS.is_dir(), reason="TimesFM 3.0 checkpoint is not local")
def test_prod_timesfm_zero_event_identity_and_gradient():
    torch.manual_seed(0)
    cfg = ModelConfig(mode="prod", timesfm_model_id=str(next(TIMESFM_SNAPSHOTS.iterdir())))
    backbone = ModularTimesFM3Backbone(cfg, d_cond=16)  # loads only TimesFM, not Qwen
    assert len(backbone.hyper_layers) == 40
    batch, _ = make_elasticity_batch(2)
    _, active = check_backbone_identity(backbone, batch, cfg.quantiles, 16, 96, range(40, 48))
    active.base_quantiles.sum().backward()
    for layer in backbone.hyper_layers:
        for p in (layer.A, layer.B, layer.proj.weight):
            assert p.grad is not None and p.grad.abs().sum() > 0 and not p.grad.isnan().any()
        assert layer.base.weight.grad is None


def test_hypernetwork_gradient_flow(cfg):
    model = new_model(cfg)
    batch, _ = make_elasticity_batch(8)
    out = model(batch)
    (dynamics_loss(out, batch.future_y, batch.future_mask, cfg.quantiles, cfg)["total"] + out.hyper_reg).backward()

    lora_b = [p for n, p in model.qwen.named_parameters() if "lora_B" in n]  # lora_A is 0 at step 1 (B = 0)
    assert lora_b and all(has_grad(p) for p in lora_b)
    assert len(model.timesfm.hyper_layers) == 40
    for layer in model.timesfm.hyper_layers:
        assert all(has_grad(p) for p in (layer.A, layer.B, layer.proj.weight, layer.proj.bias))
    assert all(has_grad(p) for p in model.head.parameters())


def test_context_cache_and_quantile_monotonicity(cfg, monkeypatch):
    model = new_model(cfg)
    batch, _ = make_elasticity_batch(6)
    batch.series_id = [f"series-{b}" for b in range(6)]
    calls = []
    encode = model.qwen.encode
    monkeypatch.setattr(model.qwen, "encode", lambda *a: calls.append(1) or encode(*a))
    cache = EventContextCache()
    with torch.no_grad():
        miss, hit, no_cache = model(batch, cache=cache), model(batch, cache=cache), model(batch)
    assert len(calls) == 2  # miss + no-cache run; the hit does not run Qwen
    assert (hit.quantiles - miss.quantiles).abs().max().item() == 0.0
    # Cached values are contiguous copies of strided views: only float summation order can differ.
    assert torch.allclose(hit.quantiles, no_cache.quantiles, atol=1e-5)

    # Large hypernetwork weights: strong positive and negative impulses must not cross quantiles.
    with torch.no_grad():
        nn.init.normal_(model.head.hyper[-1].weight, std=0.5)
        for layer in model.timesfm.hyper_layers:
            nn.init.normal_(layer.proj.weight, std=0.5)
        out = model(batch)
    u = out.u_median[out.envelope > 0]
    assert (u > 1.0).any() and (u < -1.0).any()
    assert (out.quantiles.diff(dim=-1) > 0).all()


def test_dynamic_operator_impulse_learning():
    """50 steps (task.md). Same event text; the 3.5x and 1.0x series must get clearly different impulses."""
    _, before, after = train(steps=50, seed=0, accelerator="cpu")
    assert after["loss"] < 0.5 * before["loss"]
    assert after["u_event_hi"] > 1.5 * after["u_event_lo"] > 0  # larger lift for the high-elasticity history
    assert after["warmup_shift_hi"] < 0  # warm-up dip below the base median
    assert after["overdraft_shift_hi"] < 0  # post-event overdraft below the base median
    assert after["width_ratio_event_hi"] > 1.0  # wider quantile range in the event
