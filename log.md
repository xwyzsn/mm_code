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

## 2026-10-17 — Model documentation: original design vs. current implementation

**Problem:** There was no single document that describes the original design in `task.md`, the current implementation, and the differences between them with reasons.

**Solution:**
- Added `model.md`: the original design (idea, requirements, Mermaid data flow, component table), the current implementation (Mermaid data flow with the changed parts in orange, one section per component), and the differences in three tables: Changes D1–D7, Interpretations I1–I22, Additions A1–A6. Each row has the reason and a link to the `check.md` ID. Section 4 lists the verification limits.
- `check.md`: added two new open items found while writing: C6 (σ_X "robust std" vs. plain std) and C7 (Q_hor base q0 + step MLP instead of an H × d_llm matrix).

## 2026-10-17 — Scheme 2 (HyperLoRA), Stage A of 3: HyperDoRALinear and modular TimesFM 3.0

**Problem:** `task.md` now describes Scheme 2. Qwen works as a hypernetwork that modulates the last 4 TimesFM 3.0 layers per sample (Hyper-DoRA). This needs a dynamic linear layer, a way to put it into the real and the mock TimesFM, and gradients through the prod TimesFM (`decode` has `@torch.no_grad`).

**Solution:**
- `models/hyper_dora_linear.py`:
  - `HyperDoRALinear`: frozen W0 and b0; A with Kaiming init; B ~ N(0, 0.02); `HyperParamProjector` with N(0, 1e-3) weights and zero bias.
  - Batched forward y = (1 + m_act · tanh Δm) · y0 + m_act · ((h Aᵀ) ⊙ s) Bᵀ.
  - `HyperCondition` maps m(t) to patch tokens (`check.md` H4).
  - `inject_hyper_dora()` replaces every `nn.Linear` in the last N layers.
  - `hyper_condition()` is a context manager.
- `backbones/timesfm_modular.py`:
  - `MockTimesFM3Core` uses the TimesFM 3 module names (10 linears per layer) and horizon patch tokens.
  - `ProdTimesFM3Core` calls `TimesFM3Torch.decode.__wrapped__` (gradients flow).
  - `ModularTimesFM3Backbone` gives `TimesFMOutput(top_hidden, base_quantiles)`.
- `configs/model_config.py`: `n_last_layers = 4`, `hyper_rank = 8`, `head_rank = 4`, `ts_layers = 4`.
- `tests/test_hyper_mechanics.py`:
  - batched equivalence (task.md form and full form, error < 1e-6),
  - mock backbone zero-event identity and causal locality,
  - prod TimesFM 3.0 zero-event identity and gradients (skipped if the checkpoint is not local).
- `check.md`: Scheme 2 items H1–H16 (H1, H2 confirmed).
- Result: `pytest` 11 passed (3 new + 8 Scheme 1). On the real checkpoint: 40 modules replaced, exact identity at m = 0, non-zero gradients for A, B and the projector of all 40 modules, no gradient for W0.

## 2026-10-17 — Scheme 2 (HyperLoRA), Stage B of 3: Qwen hypernetwork, dynamic quantile head, end-to-end model

**Problem:** Stage A had the dynamic TimesFM layers, but no hypernetwork that generates C_event and P_step(t), no dynamic quantile head, and no end-to-end model. The Scheme 1 files were still on the branch.

**Solution:**
- `backbones/qwen_backbone.py` → `backbones/qwen_hyper_backbone.py` (git mv). Added:
  - event selection and a 5-phase `event_lifecycle` (H9),
  - `response_features` (dip depth, peak ratio, integrated gain, recovery slope; H8),
  - `EventProjector`,
  - `EventContextCache` (key = (series_id, SHA-1 of the sample inputs and H); H10, H20),
  - `QwenHyperBackbone`. Its context is [stat, (event semantics, response) pairs, future events, `<EVENT_SUMMARY>`, H step queries], and it returns C_event and P_step (H7).
- `models/dynamic_quantile_head.py`: `DynamicQuantileHead`. `HeadHyperMLP` generates U_head [B, H, d_ts, r_h] and V_head [B, H, r_h, 3], then the non-crossing quantile composition is applied, with `torch.where(m > 0, ·, base)` for the exact identity (H11, H17).
- `models/hyper_timesfm_model.py`: `HyperTimesFMModel` (Qwen hypernetwork → HyperDoRA TimesFM → dynamic head) and `HyperTSOutput` with `hyper_reg` (H19).
- `data/schema.py`: optional `series_id`.
- Removed the Scheme 1 files: `backbones/timesfm_backbone.py`, `models/{llava_ts_model, interleaved_context, time_patch_tokenizer, distributional_film}.py`, `tests/test_model_mechanics.py`, and the unused config fields `n_patch_tokens`, `slice_len`, `recent_len`.
- `losses/dynamics_loss.py` and `train_demo.py`: import the new model (the field names are the same). Stage C reworks both files.
- `tests/test_hyper_mechanics.py`: added test 1 (`test_zero_event_strict_weight_identity`, H = 12/24/96), test 3 (`test_hypernetwork_gradient_flow`) and test 4 (`test_context_cache_and_quantile_monotonicity`).
- Result: `pytest` 8 passed in 5.5 s (including the prod TimesFM 3.0 test). The MPS forward + backward takes 0.07 s for a batch of 32.

## 2026-10-17 — Scheme 2 (HyperLoRA), Stage C of 3: loss terms, multi-stage pulses, demo, test 5, model.md

**Problem:** The Scheme 2 loss had no separate weights for the first and second differences and no hypernetwork regularizer. The generator had no exponential decay or post-event overdraft. Test 5 was missing. With lr 4e-3 the 50-step demo did not learn the warm-up dip or the elasticity difference. `model.md` still described Scheme 1.

**Solution:**
- `configs/model_config.py`: `lambda_dyn` → `lambda_diff1`, `lambda_diff2` (0.5 each), `lambda_cf` = 0.5, and the new `lambda_reg` = 1e-3.
- `losses/dynamics_loss.py`: total = L_pinball + λ1 L_diff1 + λ2 L_diff2 + λ3 L_cf + λ_reg · `hyper_reg`. The keys are now `diff1`, `diff2` and `reg`.
- `data/synthetic_generator.py`: pulse = warm-up ramp dip + onset spike with exponential decay + post-event overdraft (`check.md` H22). It adds `post_buffer` and `series_id`.
- `train_demo.py`: lr 4e-3 → 1e-3 (H21). The report adds `overdraft_shift_hi`.
- `tests/test_hyper_mechanics.py`: `test_dynamic_operator_impulse_learning` (50 steps; H23).
- `model.md`: rewritten for Scheme 2. It has the original design, the current implementation, Changes D1–D6, Interpretations I1–I20, Additions A1–A5, and the verification limits. The Scheme 1 version stays on `main`.
- `check.md`: H12 resolved; H21–H23 added.
- Result: `pytest` 9 passed in 16 s. 50 steps, lr 1e-3, seeds 0–3: loss × 0.29–0.37, u_hi / u_lo 1.86–2.35, warm-up shift −0.27 to −0.41σ, overdraft shift −0.12 to −0.28σ, width ratio 1.24–1.39. MPS run: loss 1.34 → 0.38.
