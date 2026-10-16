# Open Questions and Decisions

Status: **open** = needs your decision or confirmation. **confirmed** = you accepted it.

## Scheme 2 — HyperLoRA (branch `hyperlora`)

### H1 — confirmed (replace, reuse parts) — What happens to the Scheme 1 code on this branch?
`task.md` says "new empty project", but the branch has the Scheme 1 code. Many parts can be used again: config, schema, generator, mock Qwen + LoRA, event selection, lifecycle, loss. Options: (a) replace Scheme 1 and move the reused parts into the new file names; (b) keep both schemes, with shared modules.

### H2 — confirmed (all 10 per layer) — Which linear modules in the last 4 TimesFM layers become `HyperDoRALinear`?
Each real TimesFM 3.0 layer has 10 `nn.Linear` modules: `seq_attn.{query,key,value,out}_proj`, `var_attn.{query,key,value,out}_proj`, `ff0`, `ff1`. Options: all 10 per layer (40 modules), or only `seq_attn` + `ff` (24 modules).

### H3 — open — Per-sample weights inside the flattened TimesFM layouts
TimesFM calls its linears with flattened inputs: `seq_attn` (b·v, n, d), `var_attn` (b·n, v, d), `ff` (b, v, n, d). With one variate (v = 1, no covariates in the schema), all three can be reshaped to (B, n, d) in the same token order. **Decision:** `HyperDoRALinear` reshapes its input to (B, −1, d_in), and the model supports v = 1 only.

### H4 — open — m_act for each TimesFM patch token
`task.md` gives m_act ∈ [B, T, 1] but does not map m(t) on steps to patch tokens. **Decision:** token n_ctx − 1 + k forecasts the steps from k·p on, so m_act(token) = max m(t) over the steps that the token forecasts. m_act = 0 for all context tokens before n_ctx − 1, so the history processing does not change.

### H5 — open — Gradients through the prod TimesFM
`TimesFM3Torch.decode` has `@torch.no_grad()`. Gradients must flow through the dynamic layers. **Decision:** call the undecorated function (`TimesFM3Torch.decode.__wrapped__`). The frozen TimesFM weights still get no gradient.

### H6 — open — Mock TimesFM structure
**Decision:** a new `ModularTimesFM3Backbone` mock with the TimesFM 3 module names (`transformer_stack.layers[i].seq_attn.query_proj`, …, `ff0`, `ff1`) and horizon patch tokens, so that the same replacement code works in both modes.

### H7 — open — Source of C_event and P_step(t)
**Decision:** the Qwen context ends with one `<EVENT_SUMMARY>` token and H step query tokens. C_event = the top hidden state of the summary token. P_step(t) = the top hidden state of step query t.

### H8 — open — History response features
`task.md` names 4 features per history event. **Decision:** computed on U(t) relative to the moving baseline:
- dip depth = min U in the pre-buffer,
- peak ratio = max exp(U) in the event,
- integrated gain = Σ U over the event,
- recovery slope = mean ΔU over the post-buffer.

### H9 — open — Five phases
`task.md` now names 5 phases (warm-up, burst, sustain, tail, overdraft). **Decision:**
- warm-up = pre-buffer,
- burst = the first third of the event,
- sustain = the middle third,
- tail = the last third,
- overdraft = post-buffer.

### H10 — open — `EventContextCache` key and validity
**Decision:** key = (series_id, SHA-1 of the Qwen input tensors and H). The value is the detached (C_event, P_step). The cache is used only when it is turned on (inference or frozen-feature training). With trainable LoRA, cached values become stale. The schema gets an optional `series_id`.

### H11 — open — Spread start value of the dynamic head
s̃ = H_top · U · V has no bias, so s_spread starts near Softplus(0) + 0.1 = 0.79, and the event quantile range becomes narrower before training. **Stage B decision:** kept as specified (no bias), because there was no answer yet. The alternative is a bias so that s_spread = 1.0 at the start (as in Scheme 1, S7).

### H12 — resolved — Test 5 with 50 steps
Scheme 2 passes test 5 with the 50 steps of `task.md` (lr 1e-3, seeds 0–3 all pass). No extra steps are needed, unlike Scheme 1 (C1).

### H13 — open — The row norm m0 is not stored
`task.md` lists the frozen row norm m0 = ‖W0‖_row. The formula y = (1 + m_act · tanh(Δm)) · y0 + m_act · Δy uses only a relative gain, so m0 is never needed. **Decision:** m0 is not stored. Tell me if you want the DoRA form m0 · (1 + tanh Δm) · W / ‖W‖ instead. That form changes the output even when Δm = 0, if B · Diag(s) · A ≠ 0.

### H14 — open — Mock TimesFM training
As in Scheme 1 (S1): the mock TimesFM is trainable, except W0 and b0 inside the 40 `HyperDoRALinear` modules (frozen, as specified). In prod, all TimesFM weights are frozen.

### H15 — info — Stage A keeps Scheme 1 runnable
`masked_mean_std`, `sinusoidal` and `interp_quantiles` exist in `backbones/timesfm_backbone.py` (Scheme 1) and `backbones/timesfm_modular.py` for one commit. Stage B deletes the Scheme 1 files.

### H16 — info — Test 2 runs in float64
The batched-equivalence test runs in float64, so it checks the algebra (error < 1e-6) without float32 summation-order noise. It checks the `task.md` form (m_act = 1, Δm = 0) and the full form (per-token m_act and Δm).

### H17 — open — HeadHyperMLP initialization
`task.md` gives no init for the head. If both U_head and V_head start near 0, the bilinear output H_top · U · V has almost no gradient at step 1. **Decision:** last-layer weights N(0, 1e-3); the U part of the bias N(0, 1/d_ts) and the V part N(0, 1e-2). Then U is a random projection, V is small, the head output starts near 0, and all head parameters get a gradient at step 1.

### H18 — info — Cache hit vs. no cache
A cache hit equals the cache miss exactly (test 4). Compared with a run without a cache, the difference is ≤ 2e-6, because the cached tensors are contiguous copies of strided views and BLAS can use a different summation order. The test checks this with `allclose(atol=1e-5)`.

### H19 — open — Hypernetwork regularizer
**Decision:** `hyper_reg` = mean over samples of Σ over the 40 HyperDoRA layers of (‖s(E)‖² + ‖Δm(E)‖²). `task.md` does not say whether this is a sum or a mean over the layers. The weight λ_reg comes in Stage C.

### H20 — info — Cache misses return cached values too
On a miss, the whole batch is encoded, stored (detached), and then read from the cache. So a miss and a hit give the same tensors, and no gradient flows into Qwen when a cache is used.

### H21 — open — Learning rate 1e-3
Measured at 50 steps (batch 32): lr 1e-3 passes every test-5 check on seeds 0–3. lr 2e-3 fails on seed 3 (warm-up shift +0.19). lr 3e-3 loses the elasticity separation (u ≈ 0.05), and lr 8e-3 diverges (loss × 5–8). The hypernetwork path is sensitive to the learning rate. A warm-up schedule or per-group learning rates may be needed for prod.

### H22 — open — Synthetic pulse shape
`task.md` names the stages (warm-up dip, onset spike, exponential decay, post-event overdraft) but gives no shape. **Decision:** factor = 1 − 0.15·A·ramp (pre-buffer) + A·exp(−(t − start) / (duration / 3)) in the event − 0.1·A·exp(−(t − end) / 1.5) in the post-buffer, with A ∈ {3.5, 1.0}.

### H23 — open — Test 5 pass conditions
On the held-out batch after 50 steps (seed 0):
- loss < 0.5 × loss before,
- u_event_hi > 1.5 · u_event_lo > 0,
- warm-up shift < 0,
- overdraft shift < 0,
- width ratio > 1.0.

Seeds 0–3 give u_hi / u_lo = 1.86–2.35 and loss ratios 0.29–0.37.

## Stage 3 (losses, demo, tests)

### C1 — open — Test 5 trains 100 steps, not 50
- **Spec:** `task.md` says "train 50 steps" and expects the model to separate the 3.5x and 1.0x elasticity.
- **Measured** (CPU, batch 32, lr 4e-3, seeds 0/1/2):

  | Steps | u_event_hi − u_event_lo | Warm-up shift (σ) | Loss after / before |
  |---|---|---|---|
  | 50 | 0.04 / 0.19 / 0.00 | −0.30 / −0.22 / −0.23 | 0.64 / 0.51 / 0.59 |
  | 100 | 0.39 / 0.31 / 0.58 | −0.32 / −0.24 / −0.04 | 0.34 / 0.33 / 0.35 |

- **Cause:** a linear probe on Z_qwen separates hi and lo with 100% accuracy before training. The context signal is present. In 50 steps, the FiLM readout learns the average lift first and the difference later.
- **Decision:** `TEST5_STEPS = 100` (about 14 s on CPU). `train_demo.py` keeps 50 steps as the default.
- **Options:** (a) accept 100 steps; (b) keep 50 steps with a weak assertion (hi > lo only; seed 0 passes with a margin of 0.04); (c) change the synthetic task or the model so that 50 steps is enough.

### C2 — open — Soft locality bias in the FiLM cross-attention
- **Problem:** with full attention, each future step t attends to the Z_qwen of all H steps. After 50 steps, the warm-up dip was not learned (shift +0.41 σ instead of < 0), and the loss fell only to 0.92.
- **Decision:** add the attention logit bias `−2.0 · |t − t'|` (`LOCALITY_SLOPE` in `models/distributional_film.py`). W_Q, W_K and W_V stay learnable and get gradients.
- **Rejected:** a strict diagonal mask works best, but then W_Q and W_K get zero gradients (the softmax has one key). A band of ±1 to ±3 steps and a residual path are weaker.
- **Question:** is this change to the spec acceptable?

### C3 — open — Soft transition above 15σ in the pinball loss
- **Spec:** "use a Huber soft transition only for |e| > 15σ". A true Huber loss is quadratic near 0, and a linear pinball loss has bounded gradients already, so the wording is not clear.
- **Decision:** linear for |e| ≤ 15; above that, `15 · (1 + ln(|e| / 15))`. The value and the slope are continuous at 15, and the gradient falls as 15/|e|.
- **Question:** is this what you mean, or do you want a different transition (for example, a constant gradient cap)?

### C4 — open — Test 5 pass conditions (relative checks)
You chose "relative checks". These are the exact assertions on a held-out batch (seed 10000, 64 series):
- loss after < 0.6 × loss before
- u_event_hi − u_event_lo > 0.2
- warm-up shift of the median (3.5x series) < 0
- event quantile width / base width (3.5x series) > 1.0

`event_mae_ratio_hi` (event MAE of the model / MAE of the base) is printed but not asserted. At 100 steps it is 0.21.

### C5 — open — Smaller loss choices
- First and second differences are computed only inside the forecast window. The step from the last history value to t = 0 is not used.
- The target u* uses `base_median.detach()`, because u* is a target.
- Learning rate 4e-3 (2e-2 and above diverge). The optimizer is AdamW for all trainable parameters.

## Found while writing model.md

### C6 — open — σ_X: "robust std" or plain std?
- **Spec:** `task.md` says "稳健标准差 σ_X = max(Std(X), 1e-5)". The name says robust, but the formula is the plain std.
- **Decision:** the code follows the formula (masked population std). When there are large events in the history, σ_X becomes larger, and R(t) of the events becomes smaller.
- **Question:** do you want a robust estimator (for example, 1.4826 · MAD) instead?

### C7 — open — Q_hor base is q0 + step MLP, not an H × d_llm matrix
- **Spec:** Q_hor ∈ R^{H×d_llm} is a learnable query token base.
- **Decision:** one learnable vector q0 ∈ R^{d_llm} + an MLP of the step encoding. A fixed H × d matrix cannot support "any H" (for example, an H that training did not use).
- **Question:** is this acceptable? The alternative is a matrix for a maximum H_max, sliced to H.

## Stage 1–2 (already reported)

### S1 — confirmed — Mock TimesFM is trainable in mac_debug; prod TimesFM is frozen with a trainable horizon adapter.
### S2 — confirmed — Prod TimesFM uses the official `timesfm3` package (installed from GitHub).
### S3 — confirmed — peft LoRA is used on the mock Qwen too.
### S4 — confirmed — Test 4 checks `lora_B` gradients (`lora_A` is zero at step 1).

### S5 — open — An event can be in E_hist and E_fut
An event that starts before t0 and lasts into the forecast window is in both sets.

### S6 — open — Phase definitions
Warm-up = [start − pre, start), burst = [start, mid), decay = [mid, end), overdraft = [end, end + post).

### S7 — open — s_spread starts at 1.0
The `Linear_s` bias starts at `ln(e^0.9 − 1)`. With a zero bias, s_spread would start at 0.79.

### S8 — open — Exact identity uses `torch.where(m > 0, y, base)`
This gives the same values as the formula, but without float rounding.

### S9 — open — Intraday sin/cos counts from t0
The schema has no time of day for t0. Add one if t0 is not at the start of a period.

### S10 — open — Hugging Face Hub offline loading
On this machine, the Hub client tries a SOCKS proxy and fails. To load offline, set `timesfm_model_id` to the local snapshot folder (or install `httpx[socks]`).
