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

## 2026-10-16 — Stage 2 of 3: model layers (`models/`)

**Problem:** The project needs the LLaVA-TS model layers: time-series soft tokens, the interleaved context with H query tokens, the quantile FiLM layer, and the end-to-end model.

**Solution:**
- `models/time_patch_tokenizer.py`: trailing masked moving average, jump features R(t) and U(t), `TimeToLLMProjector` (conv kernels 3/5/9 + adaptive pooling + 2-layer MLP). Each history event slice `[start - pre, end + post)` (32 samples: R, U, TimesFM shallow states, phase one-hot, valid flag) gives P = 4 `<TS_HIST_PATCH>` tokens. The recent window gives P = 4 `<TS_RECENT_PATCH>` tokens. One series-statistics token.
- `models/interleaved_context.py`: `select_events` (publish_offset < 0 only; E_hist = start in [-L, 0), most recent N_hist; E_fut = extended lifetime meets [0, H), earliest N_fut; both in time order). `EventProjector` (time token + label token + Qwen text-token embeddings + multi-vector tokens). `HorizonQueries` (learnable base + step and intraday sin/cos + per-event lifecycle encoding, any H). `InterleavedContextBuilder` gives `inputs_embeds` and the attention mask. Padded and unpublished events are masked tokens.
- `models/distributional_film.py`: `event_lifecycle` (envelope m(t) and phases warm-up / burst / decay / overdraft). `DistributionalFiLM`: cross-modal attention (separate W_Q, W_K, W_V), heads u, beta, s with N(0, 1e-3) weights. The s bias is set so that s_spread = 1.0 at init. Where m = 0, the output is the base tensor itself (`torch.where`), so the identity is exact with no float rounding.
- `models/llava_ts_model.py`: `LlavaTSModel` (TimesFM → event selection → tokenizer → context → one Qwen prefill → FiLM) with any H and Q.
- `configs/model_config.py`: added `slice_len = 32`.
- `backbones/timesfm_backbone.py`: compute `ndtri` on CPU, because the op has no MPS kernel.
- Smoke check (CPU and MPS): H = 12/24/72/96 shapes, non-crossing, no NaN; zero events give max |y - base| = 0.0; an unpublished event gives max diff 0.0; gradients are non-zero for LoRA B, tokenizer, Q_hor base and all FiLM parameters; `sem_vecs` without text works.

## 2026-10-17 — Stage 3 of 3: losses, training demo, tests

**Problem:** The project needs the dynamics loss, a minimal Lightning training loop, and the 5 model mechanics tests from `task.md`. With full cross-attention in the FiLM layer, 50 training steps did not learn the pre-event dip or the elasticity difference.

**Solution:**
- `losses/dynamics_loss.py`: `dynamics_loss()` in the space (Y − mu_X) / sigma_X. Pinball over all quantiles (linear up to 15σ, log growth above); pinball on first differences; Smooth-L1 on the median second difference; Smooth-L1 between u_0.5 and the clipped u* where m > 0. Total = L_quantile + lambda_dyn · L_dynamics + lambda_cf · L_cf.
- `train_demo.py`: `LlavaTSLightning` (AdamW, lr 4e-3), `SyntheticStream` (a new batch at each step), `elasticity_report()` (held-out batch), `train()`, and a CLI (`--steps`, `--accelerator`).
- `tests/test_model_mechanics.py`: the 5 tests (test 1 for H = 12/24/72/96). Test 5 trains 100 steps (see `check.md` C1).
- `models/distributional_film.py`: soft locality bias −2.0 · |t − t'| in the cross-attention (see `check.md` C2).
- `check.md`: all open questions and decisions (Stages 1–3).
- Result: `pytest` 8 passed in 18 s. Demo on MPS, 50 steps: loss 1.00 → 0.65, warm-up shift −0.31σ, width ratio 2.9, u_hi/u_lo 0.54/0.51. CPU, 100 steps: loss 1.00 → 0.34, u_hi/u_lo 0.73/0.34, warm-up shift −0.32σ, event MAE 0.21 × base.
