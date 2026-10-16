"""Qwen backbone with LoRA. One prefill forward over `inputs_embeds`; returns the top hidden states.

prod:      transformers AutoModelForCausalLM (real Qwen3.8-27B weights).
mac_debug: MockQwenCausalLM, pure PyTorch, same module names and forward interface.
LoRA is applied with peft in both modes.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn

from configs.model_config import ModelConfig


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
