# Change Log

## 2026-10-16 — Stage 1 of 3: configs, data contract, synthetic data, backbones

**Problem:** The LLaVA-TS project in `task.md` starts from an empty folder. The model layers (Stage 2) and the losses, training and tests (Stage 3) need a config, an input contract, synthetic data and two backbones that run on a Mac.

**Solution:**
- Made a `uv` virtual environment `.venv` (Python 3.10, torch 2.14, lightning 2.6, peft 0.21, transformers 5.20) and added `requirements.txt`.
- `configs/model_config.py`: `ModelConfig` with `mac_debug` (default) and `prod` (`ModelConfig.prod()`). It checks that the quantile set contains 0.5.
- `data/schema.py`: `EventBatch` and `MultimodalTSBatch` dataclasses (text ids or multi-vector semantics + labels; start/end/publish offsets).
- `data/synthetic_generator.py`: `make_elasticity_batch()` makes series with the same event text but different elasticity (3.5 vs 1.0), with a pre-event dip and a double peak.
- `backbones/qwen_backbone.py`: `MockQwenCausalLM` (pure PyTorch, RoPE, RMSNorm, SwiGLU, HF module names and forward interface) and `QwenBackbone` (peft LoRA in both modes, single prefill over `inputs_embeds`, left padding, BF16 and gradient checkpointing in prod).
- `backbones/timesfm_backbone.py`: `MockTimesFM3Backbone` (causal patch Transformer) and `ProdTimesFM3Backbone` (frozen transformers `TimesFmModelForPrediction` + trainable horizon adapter, quantile interpolation). Both give shallow step states, future states and non-crossing base quantiles for any L, H and Q.
- Smoke check: shapes for H=12/96 and L=192/37, monotone quantiles, no NaN with left padding, gradients reach LoRA and `inputs_embeds`. The prod paths are not run (no weights on this machine).

## 2026-10-16 — Stage 1 fix: prod TimesFM 3.0 uses the official `timesfm3` loader

**Problem:** The prod TimesFM wrapper used the transformers `TimesFmModelForPrediction` (TimesFM 2.x interface). The real checkpoint `google/timesfm-3.0-pytorch` has a different format (variate attention, CPM iterative RevIN, output patch 64) and loads only with the official `timesfm3` package. `timesfm3` is not on PyPI.

**Solution:**
- Installed `timesfm[torch]` from `git+https://github.com/google-research/timesfm` (contains `timesfm3`) and added it to `requirements.txt`.
- `ProdTimesFM3Backbone` now loads `timesfm3.TimesFM3Torch` (frozen) and calls `decode(..., return_aux_outputs=True)` once. Shallow states = `transformer_input` of the context patches. Future states = `transformer_output` of the patch tokens that forecast each step, plus the trainable horizon adapter. The native 9 quantiles are sorted (as in `TimesFM3Forecaster`) and interpolated to Q. `history_mask` can now have gaps.
- `ModelConfig.timesfm_model_id` = `google/timesfm-3.0-pytorch`.
- Check on the local checkpoint (CPU): d_ts = 1280, patch 32, H = 24/96/130 and Q = 1/3/5 levels work, no NaN, monotone. The quantiles are equal to `TimesFM3Forecaster.predict()` (max abs difference 0.0).
- Started a Git repository and committed Stage 1.
