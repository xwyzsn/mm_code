"""Qwen3.8-27B as an event-conditioned hypernetwork encoder.

Context: [stat, (EventSem(e_hist_i), Response(e_hist_i))..., EventSem(e_fut_j)..., <EVENT_SUMMARY>, Q_1..Q_H]
Outputs: C_event [B, d_llm] (top state of <EVENT_SUMMARY>) and P_step(t) [B, H, d_llm] (top states of Q_t).
prod:      transformers AutoModelForCausalLM (real Qwen3.8-27B weights).
mac_debug: MockQwenCausalLM, pure PyTorch, same module names and forward interface.
LoRA is applied with peft in both modes. EventContextCache reuses (C_event, P_step) for the same context.
"""
import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn

from backbones.timesfm_modular import sinusoidal
from configs.model_config import ModelConfig
from data.schema import EventBatch, MultimodalTSBatch


class MockQwenConfig:
    model_type = "mock_qwen"

    def __init__(self, vocab_size: int, hidden_size: int, num_hidden_layers: int, num_attention_heads: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

    def to_dict(self) -> dict:
        return {**self.__dict__, "model_type": self.model_type}


@dataclass
class MockCausalLMOutput:
    logits: Tensor
    hidden_states: Optional[Tuple[Tensor, ...]] = None


def apply_rope(x: Tensor, pos: Tensor) -> Tensor:
    """x: [B, h, S, d], pos: [B, S]."""
    d = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos[:, None, :, None].float() * inv_freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class MockAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x: Tensor, pos: Tensor, allowed: Tensor) -> Tensor:
        B, S, d = x.shape
        q, k, v = (p(x).view(B, S, self.n_heads, -1).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        o = F.scaled_dot_product_attention(apply_rope(q, pos), apply_rope(k, pos), v, attn_mask=allowed)
        return self.o_proj(o.transpose(1, 2).reshape(B, S, d))


class MockDecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d)
        self.self_attn = MockAttention(d, n_heads)
        self.post_attention_layernorm = nn.RMSNorm(d)
        self.gate_proj, self.up_proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, 3 * d, bias=False)
        self.down_proj = nn.Linear(3 * d, d, bias=False)

    def forward(self, x: Tensor, pos: Tensor, allowed: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), pos, allowed)
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class MockQwenCausalLM(nn.Module):
    """Small causal LM (RoPE, RMSNorm, SwiGLU) with the forward interface of a transformers CausalLM."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = MockQwenConfig(cfg.llm_vocab_size, cfg.d_llm, cfg.llm_layers, cfg.llm_heads)
        self.embed_tokens = nn.Embedding(cfg.llm_vocab_size, cfg.d_llm)
        self.layers = nn.ModuleList(MockDecoderLayer(cfg.d_llm, cfg.llm_heads) for _ in range(cfg.llm_layers))
        self.norm = nn.RMSNorm(cfg.d_llm)
        self.lm_head = nn.Linear(cfg.d_llm, cfg.llm_vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> MockCausalLMOutput:
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        B, S, _ = x.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, S, dtype=torch.long, device=x.device)
        if position_ids is None:
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
        eye = torch.eye(S, dtype=torch.bool, device=x.device)  # keeps padded rows finite
        allowed = ((causal & attention_mask.bool()[:, None, :]) | eye)[:, None]

        hidden_states = [x]
        for layer in self.layers:
            x = layer(x, position_ids, allowed)
            hidden_states.append(x)
        x = self.norm(x)
        hidden_states[-1] = x
        logits = self.lm_head(x[:, -logits_to_keep:])
        return MockCausalLMOutput(logits, tuple(hidden_states) if output_hidden_states else None)


class QwenBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.mode == "prod":
            from transformers import AutoModelForCausalLM

            base = AutoModelForCausalLM.from_pretrained(
                cfg.qwen_model_id, dtype=torch.bfloat16 if cfg.bf16 else torch.float32
            )
            if cfg.gradient_checkpointing:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            base = MockQwenCausalLM(cfg)
        self.hidden_size = base.config.hidden_size
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
        )
        self.model = get_peft_model(base, lora)

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        return self.model.get_input_embeddings()(input_ids).float()

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor) -> Tensor:
        """inputs_embeds: [B, S, d_llm] (left padded), attention_mask: [B, S]. Returns top hidden [B, S, d_llm]."""
        dtype = self.model.get_input_embeddings().weight.dtype
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        out = self.model(
            inputs_embeds=inputs_embeds.to(dtype),
            attention_mask=attention_mask.long(),
            position_ids=position_ids,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        return out.hidden_states[-1].float()


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
    """Published events only (publish_offset < 0). E_hist: start in [-L, 0), N_hist most recent.
    E_fut: [start - pre, end + post] meets [0, H), N_fut earliest. One event can be in both sets."""
    s, e = ev.start_offset, ev.end_offset
    visible = ev.event_mask & (ev.publish_offset < 0)
    is_hist = visible & (s >= -hist_len) & (s < 0)
    is_fut = visible & (s - cfg.pre_buffer < horizon) & (e + cfg.post_buffer > 0)
    hist_idx, hist_ok = _pick(s, is_hist, cfg.n_hist_events, s)
    fut_idx, fut_ok = _pick(-s, is_fut, cfg.n_fut_events, s)
    return hist_idx, hist_ok, fut_idx, fut_ok


def event_lifecycle(start: Tensor, end: Tensor, valid: Tensor, horizon: int, cfg: ModelConfig) -> Tuple[Tensor, Tensor]:
    """start, end, valid: [B, N]. Returns envelope [B, N, H] in [0, 1] and phase one-hot [B, N, H, 5].

    Envelope: 1 in [start, end); exp(-d / tau_buf) in the pre and post buffers; 0 elsewhere.
    Phases: warm-up (pre-buffer), burst / sustain / tail (thirds of the event), overdraft (post-buffer).
    """
    t = torch.arange(horizon, device=start.device, dtype=start.dtype)
    s, e = start[..., None], end[..., None]
    inside = (t >= s) & (t < e)
    before = (t >= s - cfg.pre_buffer) & (t < s)
    after = (t >= e) & (t < e + cfg.post_buffer)
    buffer = torch.exp(-torch.where(before, s - t, t - e + 1).abs() / cfg.tau_buf)
    env = torch.where(inside, torch.ones_like(buffer), torch.where(before | after, buffer, torch.zeros_like(buffer)))
    third = (t - s) / (e - s) * 3
    phase = torch.stack([before, inside & (third < 1), inside & (third >= 1) & (third < 2), inside & (third >= 2), after], -1)
    return env * valid[..., None], (phase & valid[..., None, None]).to(start.dtype)


def moving_average(x: Tensor, mask: Tensor, window: int) -> Tensor:
    """Trailing masked mean over `window` steps. x, mask: [B, L]."""
    m = mask.to(x.dtype)
    cx = F.pad((x * m).cumsum(-1), (window, 0))
    cm = F.pad(m.cumsum(-1), (window, 0))
    return (cx[:, window:] - cx[:, :-window]) / (cm[:, window:] - cm[:, :-window]).clamp(min=1)


def response_features(x: Tensor, mask: Tensor, mu: Tensor, sigma: Tensor, start: Tensor, end: Tensor, ok: Tensor, cfg: ModelConfig):
    """History responses on U(t) = ln((max(X,0) + 0.1 sigma) / (max(MA,0) + 0.1 sigma)).

    Returns stat features [B, 4] and per-event features [B, N, 4]: dip depth (min U in the pre-buffer),
    peak ratio (max exp(U) in the event), integrated gain (sum U in the event), recovery slope (mean dU in the post-buffer).
    """
    L = x.shape[1]
    ma = moving_average(x, mask, cfg.ma_window)
    m = mask.bool()
    u = torch.log((x.clamp(min=0) + 0.1 * sigma) / (ma.clamp(min=0) + 0.1 * sigma)).masked_fill(~m, 0.0)
    r = ((x - ma) / sigma).masked_fill(~m, 0.0)
    du = F.pad(u.diff(dim=1), (1, 0))

    t = torch.arange(-L, 0, device=x.device, dtype=x.dtype)  # offsets of the history steps
    s, e = start[..., None], end[..., None]
    valid = m[:, None] & ok[..., None]
    pre = valid & (t >= s - cfg.pre_buffer) & (t < s)
    inside = valid & (t >= s) & (t < e)
    post = valid & (t >= e) & (t < e + cfg.post_buffer)
    uu, dd = u[:, None].expand_as(pre), du[:, None].expand_as(pre)
    dip = uu.masked_fill(~pre, float("inf")).amin(-1)
    peak = uu.masked_fill(~inside, float("-inf")).amax(-1)
    feats = torch.stack(
        [
            torch.where(pre.any(-1), dip, torch.zeros_like(dip)),
            torch.where(inside.any(-1), peak.exp(), torch.ones_like(peak)),
            (uu * inside).sum(-1),
            (dd * post).sum(-1) / post.sum(-1).clamp(min=1),
        ],
        -1,
    )
    stat = torch.cat([mu / sigma, sigma.log(), r[:, -1:], u[:, -1:]], -1)
    return stat, feats * ok[..., None]


def _mlp(d_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))


class EventProjector(nn.Module):
    """Event -> semantic tokens: [time topology token, label token?, Qwen text embeddings?, vector tokens?]."""

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


class EventContextCache:
    """(series_id, event_context_hash) -> detached (C_event, P_step). Valid only while the Qwen side is frozen."""

    def __init__(self):
        self.store: Dict[Tuple[str, str], Tuple[Tensor, Tensor]] = {}

    @staticmethod
    def context_hash(batch: MultimodalTSBatch, b: int, horizon: int) -> str:
        ev = batch.events
        parts = [batch.history_x, batch.history_mask, ev.start_offset, ev.end_offset, ev.publish_offset, ev.event_mask,
                 ev.text_ids, ev.text_mask, ev.sem_vecs, ev.labels]
        h = hashlib.sha1(str(horizon).encode())
        for t in parts:
            h.update(b"-" if t is None else t[b].detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


class QwenHyperBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qwen = QwenBackbone(cfg)
        d = self.hidden_size = self.qwen.hidden_size
        self.events = EventProjector(cfg, d)
        self.stat_mlp = _mlp(4, d)
        self.response_mlp = _mlp(4, d)
        self.summary = nn.Parameter(torch.randn(d) * 0.02)  # <EVENT_SUMMARY>
        self.q0 = nn.Parameter(torch.randn(d) * 0.02)  # step query base
        self.step_mlp = _mlp(16 + 2, d)
        self.life_mlp = _mlp(8 + 8 + 5 + 1, d)

    def step_queries(self, fut_start: Tensor, fut_end: Tensor, fut_ok: Tensor, horizon: int) -> Tensor:
        """[B, H, d]: q0 + step / intraday encoding + lifecycle encoding of each future event."""
        t = torch.arange(horizon, device=fut_start.device, dtype=fut_start.dtype)
        ang = 2 * math.pi * t / self.cfg.period
        step = self.step_mlp(torch.cat([sinusoidal(t, 16), ang.sin()[:, None], ang.cos()[:, None]], -1))
        env, phase = event_lifecycle(fut_start, fut_end, fut_ok, horizon, self.cfg)
        s, e = fut_start.masked_fill(~fut_ok, 0.0)[..., None], fut_end.masked_fill(~fut_ok, 0.0)[..., None]
        life = torch.cat([sinusoidal(t - s, 8), sinusoidal(t - e, 8), phase, env[..., None]], -1)
        return self.q0 + step + (self.life_mlp(life) * fut_ok[..., None, None]).sum(1)

    def encode(self, batch: MultimodalTSBatch, mu: Tensor, sigma: Tensor, sel: Tuple[Tensor, ...], horizon: int) -> Tuple[Tensor, Tensor]:
        ev, x, mask = batch.events, batch.history_x, batch.history_mask
        hist_idx, hist_ok, fut_idx, fut_ok = sel
        stat, resp = response_features(x, mask, mu, sigma, take(ev.start_offset, hist_idx), take(ev.end_offset, hist_idx), hist_ok, self.cfg)
        hist_sem, hist_mask = self.events(ev, hist_idx, hist_ok, self.qwen.embed_tokens)
        fut_sem, fut_mask = self.events(ev, fut_idx, fut_ok, self.qwen.embed_tokens)
        queries = self.step_queries(take(ev.start_offset, fut_idx), take(ev.end_offset, fut_idx), fut_ok, horizon)

        B = x.shape[0]
        ones = lambda n: torch.ones(B, n, dtype=torch.bool, device=x.device)  # noqa: E731
        hist_pairs = torch.cat([hist_sem, self.response_mlp(resp)[:, :, None]], 2).flatten(1, 2)
        hist_pair_mask = torch.cat([hist_mask, hist_ok[..., None]], 2).flatten(1, 2)
        summary = self.summary.expand(B, 1, -1)
        embeds = torch.cat([self.stat_mlp(stat)[:, None], hist_pairs, fut_sem.flatten(1, 2), summary, queries], 1)
        attn = torch.cat([ones(1), hist_pair_mask, fut_mask.flatten(1, 2), ones(1 + horizon)], 1)
        top = self.qwen(embeds * attn[..., None], attn)
        return top[:, -horizon - 1], top[:, -horizon:]

    def forward(
        self, batch: MultimodalTSBatch, mu: Tensor, sigma: Tensor, sel: Tuple[Tensor, ...], horizon: int,
        cache: Optional[EventContextCache] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Returns C_event [B, d_llm] and P_step [B, H, d_llm]."""
        if cache is None:
            return self.encode(batch, mu, sigma, sel, horizon)
        assert batch.series_id is not None, "EventContextCache needs batch.series_id"
        keys = [(batch.series_id[b], cache.context_hash(batch, b, horizon)) for b in range(len(batch.series_id))]
        if not all(k in cache.store for k in keys):
            c_event, p_step = self.encode(batch, mu, sigma, sel, horizon)
            for b, k in enumerate(keys):
                cache.store[k] = (c_event[b].detach(), p_step[b].detach())
        c_event, p_step = zip(*(cache.store[k] for k in keys))
        return torch.stack(c_event), torch.stack(p_step)
