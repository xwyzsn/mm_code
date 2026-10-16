"""Model hyperparameters. `mode="mac_debug"` (default) uses light mock backbones; `mode="prod"` loads real weights."""
from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class ModelConfig:
    mode: Literal["mac_debug", "prod"] = "mac_debug"
    horizon: int = 24  # default H; forward() accepts any H
    quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)  # any set that contains 0.5

    # Qwen backbone. In prod, d_llm comes from the checkpoint.
    qwen_model_id: str = "Qwen/Qwen3.8-27B"  # set to the real Hugging Face repo id
    d_llm: int = 128
    llm_layers: int = 2
    llm_heads: int = 4
    llm_vocab_size: int = 1000
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # TimesFM backbone. In prod, d_ts and patch length come from the checkpoint.
    timesfm_model_id: str = "google/timesfm-3.0-pytorch"
    d_ts: int = 64
    ts_layers: int = 2
    ts_heads: int = 4
    ts_patch_len: int = 8

    bf16: bool = False
    gradient_checkpointing: bool = False

    # Events
    n_hist_events: int = 4  # N_hist
    n_fut_events: int = 4  # N_fut
    pre_buffer: float = 3.0
    post_buffer: float = 3.0
    tau_buf: float = 1.5
    event_vec_dim: int = 32  # D of each pre-extracted semantic vector
    n_event_labels: int = 16

    # Time patch tokenizer and FiLM
    n_patch_tokens: int = 4  # P
    slice_len: int = 32  # samples per [pre, event, post] history slice
    ma_window: int = 24
    recent_len: int = 48
    period: int = 24  # intraday period for sin/cos encoding
    alpha_max: float = 2.5

    # Loss weights
    lambda_dyn: float = 0.5
    lambda_cf: float = 0.5
    huber_sigma: float = 15.0

    def __post_init__(self):
        assert 0.5 in self.quantiles, "quantiles must contain the median 0.5"

    @classmethod
    def prod(cls, **overrides) -> "ModelConfig":
        return cls(**{"mode": "prod", "bf16": True, "gradient_checkpointing": True, **overrides})
