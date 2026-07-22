#!/usr/bin/env python
"""Estimate context-averaged Jacobians J_l = E[dh_final/dh_l] for a causal LM.

Usage:
    python scripts/estimate_j.py configs/qwen35-0.8b-local.yaml [--resume]

Writes JEstimator checkpoints and meta.json under the config's out_dir.
Resumable: --resume picks up from the last checkpoint and fast-forwards the
corpus stream past already-consumed prompts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.corpus import stream_batches
from jlens.driver import probe_step
from jlens.estimator import JEstimator
from jlens.hooks import decoder_layers, freeze


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    torch.manual_seed(cfg["seed"])
    device = pick_device()
    dtype = pick_dtype(cfg["dtype"], device)
    print(f"device={device.type} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype)
    freeze(model).to(device)

    n_layers = len(decoder_layers(model))
    layer_indices = list(range(0, n_layers, cfg["hook_stride"]))
    if layer_indices[-1] != n_layers - 1:
        layer_indices.append(n_layers - 1)  # backward source must be hooked
    d_model = model.config.hidden_size
    print(f"{n_layers} layers, hooking {len(layer_indices)}, d_model={d_model}")

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"

    done = 0
    if args.resume and meta_path.exists():
        done = json.loads(meta_path.read_text())["prompts_done"]
        est = JEstimator.load(out / f"ckpt_{done:06d}.pt")
        print(f"resuming from {done} prompts")
    else:
        est = JEstimator(layer_indices, d_model)

    def checkpoint() -> None:
        est.save(out / f"ckpt_{done:06d}.pt")
        meta_path.write_text(json.dumps({"prompts_done": done}))

    t0 = time.time()
    start_done = done
    last_ckpt = done
    batches = stream_batches(
        tokenizer,
        cfg["dataset"],
        cfg["dataset_config"],
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        skip=done,
        limit=cfg["n_prompts"],
    )
    for batch in batches:
        probe_step(model, layer_indices, batch.to(device), est)
        done += batch.shape[0]
        if done - last_ckpt >= cfg["checkpoint_every"]:
            checkpoint()
            last_ckpt = done
            rate = (done - start_done) / (time.time() - t0)
            print(f"{done}/{cfg['n_prompts']} prompts  ({rate:.2f} prompts/s)")

    checkpoint()
    print(f"done: {done} prompts -> {out}")


if __name__ == "__main__":
    main()
