# First light: J-lens readouts on Qwen3.5-0.8B

Lens: ckpt_003000 (3000-prompt J estimates). Readout at the final prompt
position, J-lens vs logit-lens baseline, top-6.

## Instrument checks

- L23 (backward source): J-lens and logit-lens distributions identical, and
  they match the model's actual next-token distribution. The splice vanishes
  at J = I, on the real model.
- L4-9: J-lens reads coherent English tokens at 10-40x the probability where
  the logit lens reads multilingual junk — the paper's core qualitative claim
  (interpretable early/mid-layer content invisible to the logit lens).

## Spider prompt, declarative form ("...the animal that spins webs is")

No spider, no eight. From L10 on the readout converges on '____' (0.38 at
L14): the base model reads the prompt as a worksheet and predicts a blank.
Brief semantically apt content at L19-20 (' equal', ' greater',
' proportional'). Lesson: base model + quiz phrasing = quiz continuation;
the spider->legs computation isn't invoked.

## Spider prompt, QA form ("Q: How many legs ... A: It has")

The headline readout:

- L12-14: faint number content appears (' five')
- L15-17: a candidate cloud — ' four', ' five', ' six', ' seven', ' eight'
  at near-equal ~0.02 — numerosity before selection
- L18: collapse to a decision: ' four' 0.34 (' eight' 0.09 second)
- L19-22: decision holds under rebroadening; L23 shifts to formatting
  (' ' vs digit vs word)
- Model's greedy output: " 4 legs." — the WRONG answer, read correctly by
  the lens five layers early. Logit lens shows nothing legible until L18.

## Takeaways

1. Instrument validated end-to-end on the real model.
2. Candidate-cloud -> collapse dynamics visible in middle layers: the
   workspace-flavored story, at 0.8B.
3. The 0.8B base model fails spider->8 (answers 4); lens fidelity is to the
   model, not the world. Behavioral experiments need better prompts and/or
   the instruct variant.
4. Wanted next: per-token rank tracking across layers (--watch spider,eight),
   position sweeps, then the Phase 2 regime metrics.
