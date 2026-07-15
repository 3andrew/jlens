# Project plan: reproducing the J-lens / global workspace paper

Target paper: [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) (Anthropic, July 2026).

Goal: reproduce the paper's core results on open models, end to end — the J-lens
itself, the layer-regime structure, the reading and intervention experiments, and
as much of the five workspace properties as small-model competence allows. Stretch
goal: test the "J-lens vectors are a subframe of the feature frame" claim against
Gemma Scope 2 SAEs, which the paper asserts but cannot test.

## Models

| Role | Model | Where it runs |
|---|---|---|
| Pipeline dev / smoke tests | Qwen3.5-0.8B | Local (M3, MPS) |
| Iteration | Qwen3.5-4B | RunPod A100 (or Colab) |
| Headline results | Qwen3.5-9B | RunPod A100 |
| Stretch: SAE cross-validation | Gemma 3 12B + Gemma Scope 2 | RunPod A100 80GB |

All code is model-agnostic (plain HF hooks); switching models is a config change.
Behavioral experiments run in non-thinking mode.

## Repo layout

```
jlens/            # the package — all real logic lives here, importable, testable
  hooks.py        # residual-stream capture, detach-and-releaf
  estimator.py    # stochastic J_l accumulation, checkpoint/resume
  lens.py         # readout: softmax(W_U norm(J_l h)); logit-lens baseline
  decompose.py    # sparse nonnegative decomposition (gradient pursuit)
  interventions.py# steering / ablation / swap in lens coordinates
  tasks.py        # prompt suites: spider-family, arithmetic, category, selectivity
configs/          # one YAML per (model, experiment) — every run is reproducible
scripts/          # thin entrypoints: estimate_j.py, run_reading.py, ...
notebooks/        # exploration and figure-making ONLY; no load-bearing logic
results/
  raw/            # J matrices, per-run outputs (gitignored)
  figures/        # committed — the reproducible outputs of record
  notes/          # committed markdown lab notes, one per experiment day
docs/PLAN.md      # this file — updated as phases complete
tests/            # shape/correctness tests that run on CPU with a tiny model
```

Conventions: every script takes a config path; every output directory is named
`results/raw/<experiment>/<model>-<date>/`; seeds fixed in configs; figures are
regenerated from raw outputs by notebooks, never hand-edited.

## Environment

- **Local (dev):** `uv sync` — pinned Python 3.12 venv in `.venv/`. Torch runs
  MPS/CPU here; only the 0.8B model.
- **RunPod (runs):** PyTorch CUDA template, persistent network volume mounted at
  `/workspace` holding the HF cache (`HF_HOME=/workspace/hf`) and
  `results/raw/`. Clone the repo, `pip install -e . --no-deps` (container torch
  already satisfies the heavy deps), run scripts under `tmux`, `git push`
  results/notes and `rsync` raw artifacts as needed.
- Checkpoint the J accumulator every ~200 prompts; every long-running script must
  be resumable.

## Phases

### Phase 0 — scaffolding (local) ✅ / in progress
uv project, package skeleton, this plan. Gate: `pytest` runs a hook + estimator
shape test on a tiny model, on CPU.

### Phase 1 — J_l estimation (the instrument)
Stochastic estimator: freeze weights; one backward from v^T h_final per probe
yields gradients at every hooked layer/position; accumulate rank-1 v g^T in fp32
on CPU. ~25 hooked layers, ~1k FineWeb prompts, seq 512.
- **[Andrew writes]** the core accumulator update in `estimator.py` — the ~10
  lines that turn probe + gradients into a running J_l estimate.
- Gates: (a) late-layer J_l ≈ I and lens readout ≈ logit lens there; (b)
  split-half convergence — J from prompts 1–500 vs 501–1000 correlates > .95;
  (c) readouts on held-out text are sane.
- Est. compute: ~1 A100-hr (4B), ~3 (9B). Local 0.8B first, overnight on MPS.

### Phase 2 — layer-regime structure
Per-layer readout metrics over held-out prompts: excess kurtosis, rank of the
true next token, agreement with final output. Reproduce the three-regime figure
(illegible → workspace → motor) vs a logit-lens baseline.
- Gate: late layers converge to logit lens; a visible middle band where readouts
  are peaked but disagree with the imminent output token.
- Est. compute: <1 A100-hr. Key open question: does a 36-layer model show a
  sharp onset or a smear?

### Phase 3 — reading experiments
Spider-family prompts (~50, programmatically varied): does the unspoken
intermediate concept appear in middle-layer readouts? Multi-step arithmetic: do
intermediates surface in computation order across layers? Sparse decomposition
to identify workspace contents.
- **[Andrew writes]** the gradient-pursuit inner loop in `decompose.py`.
- Deliverables: hit-rate tables + qualitative gallery. Est: 1–2 A100-hrs.

### Phase 4 — interventions (writing to the workspace)
Steering, ablation, and swap-in-lens-coordinates. Reproduce spider→ant (8→6
legs) and France→China broadcast across ≥4 templates; success-rate vs
intervention strength curves. Define success metrics in the config *before*
running.
- Paper baseline: ~40% swap success on Sonnet 4.5 — expect worse; that's data,
  not failure. Est: 2–4 A100-hrs including strength sweeps.

### Phase 5 — the five workspace properties
The scorecard against the paper: reportability (category swap), top-down
control (focus/ignore instructions), reasoning (from phases 3–4), broadcast
(quantitative swap table), selectivity (language report-vs-continuation
dissociation; count report-vs-use). Selectivity is the crown jewel and the most
likely to be scale-sensitive — design tasks the 9B can actually do.
- Bonus experiment the paper doesn't have: workspace content with Qwen
  thinking-mode on vs off.
- Est: 3–5 A100-hrs.

### Phase 6 — stretch: Gemma Scope cross-validation
Re-estimate J_l on Gemma 3 12B; decompose J-lens vectors in the Gemma Scope 2
SAE/transcoder basis; measure sparsity of the decomposition vs random-direction
controls. First empirical test of the paper's "subframe of the feature frame"
claim. Est: ~5 A100-80GB-hrs.

### Phase 7 — write-up
README results section + a fuller report assembled from `results/notes/`.
Figures regenerated from raw. If phase 6 pans out, this is a publishable note.

## Budget

Rough total: 15–30 A100-hours ≈ **$30–60 on RunPod** (~$1.60–2/hr). Colab Pro
works for phases 1–3 on the 4B if preferred; RunPod for everything else.
