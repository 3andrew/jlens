#!/usr/bin/env python
"""Phase 1 gates for an estimate_j run.

Usage:
    python scripts/validate_j.py results/raw/estimate_j/qwen35-0.8b

Gate 1 (plumbing): the deepest hooked layer is the backward source itself, so
its estimate must converge to the identity — if it doesn't, the hook/estimator
wiring is broken, independent of anything about the model.

Gate 2 (convergence): J from the first half of the corpus must match J from
the second half (Pearson r > 0.95 per layer). The halves come free from
checkpoint additivity:
    second_half = (final_accum - mid_accum) / (final_count - mid_count)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from jlens.estimator import JEstimator

IDENTITY_GATE = 0.1
SPLIT_HALF_GATE = 0.95


def prompts_of(path: Path) -> int:
    return int(path.stem.split("_")[1])


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten() - a.flatten().mean()
    b = b.flatten() - b.flatten().mean()
    return (a @ b / (a.norm() * b.norm())).item()


def main() -> None:
    run_dir = Path(sys.argv[1])
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=prompts_of)
    if len(ckpts) < 2:
        sys.exit(f"need >= 2 checkpoints in {run_dir}, found {len(ckpts)}")

    final_path = ckpts[-1]
    target = prompts_of(final_path) // 2
    mid_path = min(ckpts[:-1], key=lambda p: abs(prompts_of(p) - target))
    mid = JEstimator.load(mid_path)
    fin = JEstimator.load(final_path)
    print(f"first half:  {mid_path.name}")
    print(f"second half: {final_path.name} minus {mid_path.name}\n")

    eye = torch.eye(fin.d)
    last_layer = max(fin.accum)
    ok = True
    print(f"{'layer':>5}  {'|J-I|/|I|':>10}  {'split-half r':>12}")
    for layer in sorted(fin.accum):
        j = fin.estimate(layer)
        ident_dist = ((j - eye).norm() / eye.norm()).item()
        first = mid.estimate(layer)
        second_accum = fin.accum[layer] - mid.accum[layer]
        second = second_accum / (fin.count[layer] - mid.count[layer])
        r = pearson(first, second)

        flags = []
        if r <= SPLIT_HALF_GATE:
            ok = False
            flags.append("r below gate")
        if layer == last_layer and ident_dist > IDENTITY_GATE:
            ok = False
            flags.append("backward source far from identity")
        note = f"  <-- {', '.join(flags)}" if flags else ""
        print(f"{layer:>5}  {ident_dist:>10.3f}  {r:>12.4f}{note}")

    print(f"\nGATES: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
