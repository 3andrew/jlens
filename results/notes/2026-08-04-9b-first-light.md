# Qwen3.5-9B: gates and first light

Anatomy: 32 layers, d_model=4096, 17 hooked (stride 2 + last). 6000 prompts
at 3.0 prompts/s on an A100 80GB (~33 min, ~$1). Fast-path kernels absent
(CUDA 13.0 toolkit vs cu128 torch; flash-linear-attention alone installed).

## Gates: PASS above L18, FAIL below

Source (L31): identity error 0.036 vs 0.0365 predicted — fourth consecutive
sqrt((d+1)/M) match. Split-half r: >= 0.97 for L18+, 0.65-0.93 for L8-16,
~0.02-0.21 for L0-6 (pure noise; |J-I| ~ 8.8 at L0 is noise magnitude).
Early-layer noise is far worse than the 0.8B: deeper pullback + more
cross-position contamination at d=4096. Brute-force prompts can't rescue
L0-4 (~100x needed). Instrument is calibrated for the back half of the
network; claims below L14 are off-limits for now.

## First light (QA spider prompt, position ' has')

The paper's flagship reading result reproduces:

- L24: ' spiders', ' spider', '蜘蛛', ' webs', ' Spider' — the unspoken
  intermediate, present in neither prompt nor output, multi-script.
- L26: ' legs' 0.27 (logit lens: 0.02)
- L28: ' eight' 0.58 — correct (the 0.8B said four)
- L31: J-lens = logit lens = final distribution, as always.
- Chain across depth: referent -> attribute -> value, in computation order.
- J-lens leads logit lens by ~2 layers at 4-10x confidence through the
  middle; logit lens catches faint '蜘蛛'/' webs' by L22-24.
- Curiosity: L30 logit-lens has ' eight' at 0.74 but the final distribution
  flattens to 0.04 — the last block adds entropy.

## Open items

- Shared-probe control run (paper-faithful estimand) — ~$1 on this pod.
- Early-layer estimation strategy (shorter sequences to cap cross-position
  contamination?) if we ever need L0-12.
- validate_j pearson now float64 (fp32 dot products drifted past |r|=1).
