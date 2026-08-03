#!/usr/bin/env python
"""First light: read a prompt through the J-lens, layer by layer.

Usage:
    python scripts/first_light.py configs/qwen35-0.8b-local.yaml \
        --prompt "The number of legs on the animal that spins webs is" \
        [--position -1] [--topk 6]

Prints, for each estimated layer, the top-k tokens the activation at
--position is disposed to produce — J-lens on the left, logit-lens
baseline on the right.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.estimator import JEstimator
from jlens.hooks import ResidualCapture, freeze
from jlens.lens import JLens


def latest_checkpoint(out_dir: Path) -> Path:
    ckpts = sorted(out_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {out_dir}")
    return ckpts[-1]


def fmt(entries: list[tuple[str, float]]) -> str:
    return " ".join(f"{tok!r}:{p:.2f}" for tok, p in entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--prompt",
        default="The number of legs on the animal that spins webs is",
    )
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--topk", type=int, default=6)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16)
    freeze(model).to(device)

    ckpt = latest_checkpoint(Path(cfg["out_dir"]))
    est = JEstimator.load(ckpt)
    lens = JLens(model, est)
    layers = sorted(est.accum)
    print(f"lens from {ckpt.name}; prompt: {args.prompt!r}\n")

    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad(), ResidualCapture(model, layers, grads=False) as cap:
        model(ids, use_cache=False)

    pos = args.position if args.position >= 0 else ids.shape[1] + args.position
    print(f"reading position {pos} ({tokenizer.decode(ids[0, pos])!r})\n")
    for layer in layers:
        h = cap.captured[layer][0, pos]
        jl = fmt(lens.topk(h, layer, tokenizer, k=args.topk))
        ll = fmt(lens.topk(h, layer, tokenizer, k=args.topk, identity=True))
        print(f"L{layer:>2} J-lens : {jl}")
        print(f"    logit  : {ll}\n")


if __name__ == "__main__":
    main()
