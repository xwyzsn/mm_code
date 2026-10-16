# Open Questions and Decisions

Status: **open** = needs your decision or confirmation. **confirmed** = you accepted it.

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
