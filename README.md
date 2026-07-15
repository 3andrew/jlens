# jlens

Reproduction of the Jacobian lens (J-lens) from Anthropic's
["Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace/index.html)
on open Qwen3.5 models (4B for iteration, 9B for results).

## Method sketch

For each layer ℓ, estimate the context-averaged Jacobian of the final-layer
residual stream with respect to the layer-ℓ residual stream:

    J_ℓ = E[ ∂h_final,t' / ∂h_ℓ,t ]

estimated stochastically: one backward pass from vᵀh_final (random Gaussian
probe v) yields gradients at every hooked layer and position at once; the
rank-1 outer products v·gᵀ average to J_ℓ. The lens readout is then

    lens(h_ℓ) = softmax(W_U · norm(J_ℓ · h_ℓ))

## Roadmap

1. J_ℓ estimation over ~1k pretraining-like prompts (FineWeb sample)
2. Sanity checks: J_ℓ ≈ I at late layers; readout converges to logit lens
3. Reading experiments: unspoken intermediates (spider prompt), multi-step
   arithmetic across layers
4. Sparse nonnegative decomposition into J-lens vectors (gradient pursuit)
5. Interventions: steering / ablation / concept swaps
6. Selectivity: verbalizable vs. automatic processing dissociation

## Environment

Targets Colab Pro (A100 for 9B, any GPU for 4B). Run experiments in
non-thinking mode so reasoning stays internal rather than externalized as CoT.
