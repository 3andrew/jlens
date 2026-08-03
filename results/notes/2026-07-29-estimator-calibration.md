# Phase 1 calibration: shared-probe failure, per-position fix

Model: Qwen3.5-0.8B — 24 layers, d_model=1024, all layers hooked. bf16 on MPS,
batch 2 x seq 512, FineWeb sample-10BT.

## Run 1: shared probe per prompt — FAILED (archived: qwen35-0.8b-sharedprobe-FAILED)

1000 prompts, one probe direction per batch element => M = 1000 distinct
probes in d = 1024. Identity check at backward source: 1.015 measured vs
sqrt((d+1)/M) = 1.012 predicted. Split-half r ~ 0.3 flat across layers.
Estimator mathematically correct but probe-starved: cannot resolve a d x d
matrix from M ~ d directions.

## Fix: per-position probes

v ~ randn_like(h_final), one probe per (batch, position). Same-position
pairing is unbiased (E[v_t v_t'^T] = 0 kills cross-terms); cross-position
influence becomes variance, not bias. Probe budget: M = prompts x seq_len.

Estimand change: same-position Jacobian E[dh_final_t/dh_l_t] (tuned-lens
style), not the paper's t' >= t average. Revisit cross-position terms on GPU.

## Run 2: smoke, 100 prompts — mechanism confirmed

M = 51,200. Source identity error: 0.141 measured vs 0.1415 predicted.
Split-half r monotone in depth: 0.41 (L0) -> 0.96 (L23). Early layers are
intrinsically noisier (pullback through 24 layers + cross-position variance).

## Sizing the full run

From smoke r values, r >= 0.95 at L0 needs ~2100 prompts. Full run set to
3000 prompts (~2.5 h on M3 MPS at 0.34 prompts/s).

## Run 3: full, 3000 prompts — GATES PASS (2026-08-03)

M ~ 1.5M probes/layer. Source identity error 0.026 vs 0.0258 predicted —
third consecutive match of sqrt((d+1)/M) across three orders of magnitude.
Split-half r: 0.9555 (L0, tightest) to 0.9995 (L23); every layer passes.
|J-I|/|I| declines smoothly ~1.0 -> 0.5 across depth, then 0.026 at source;
no sharp cliff visible in this statistic. Run survived an overnight laptop
sleep mid-flight (process suspended and resumed; checkpoints intact).
Ops note: cumulative rate print is misleading after stalls; HF stream
timeouts ('timed out', 'Retrying') never matched the monitor grep — widen
filters for future runs.
