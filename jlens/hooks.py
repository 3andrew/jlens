"""Residual-stream capture for J-lens estimation.

Hooks the output hidden states of selected decoder layers. The earliest
hooked layer is "releafed": its hidden states are detached and re-marked
requires_grad, so autograd builds a graph only downstream of it — layers
below run graph-free, which is what keeps backward memory bounded with
weights frozen. Later hooked layers retain_grad so their gradients
survive the backward pass.
"""

from __future__ import annotations

import torch
from torch import nn


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The stack of decoder blocks in an HF causal LM (Qwen/Llama-style)."""
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        mod = model
        try:
            for attr in path.split("."):
                mod = getattr(mod, attr)
        except AttributeError:
            continue
        if isinstance(mod, nn.ModuleList):
            return mod
    raise ValueError(f"could not locate decoder layers on {type(model).__name__}")


def freeze(model: nn.Module) -> nn.Module:
    """Freeze all weights and switch to eval mode; gradients then flow only
    through activations."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


class ResidualCapture:
    """Captures the residual stream at `layer_indices` during one forward pass.

    Use as a context manager around a model call. After a backward pass from a
    scalar built on the last captured layer, `grads()` returns the gradient at
    every captured layer. The earliest hooked layer is the releaf point unless
    `releaf=False`.
    """

    def __init__(self, model: nn.Module, layer_indices: list[int], releaf: bool = True):
        self.layers = decoder_layers(model)
        self.indices = sorted(layer_indices)
        self.releaf_at = self.indices[0] if releaf else None
        self.captured: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, idx: int):
        def hook(module, args, output):
            # Decoder layers return either a tensor or a tuple with hidden
            # states first, depending on transformers version.
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            if idx == self.releaf_at:
                hidden = hidden.detach().requires_grad_(True)
                self.captured[idx] = hidden
                return (hidden, *output[1:]) if is_tuple else hidden
            hidden.retain_grad()
            self.captured[idx] = hidden
            return output

        return hook

    def __enter__(self) -> "ResidualCapture":
        self.captured.clear()
        self._handles = [
            self.layers[i].register_forward_hook(self._make_hook(i))
            for i in self.indices
        ]
        return self

    def __exit__(self, *exc) -> bool:
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def grads(self) -> dict[int, torch.Tensor]:
        missing = [i for i, t in self.captured.items() if t.grad is None]
        if missing:
            raise RuntimeError(f"no gradients at layers {missing}; run backward first")
        return {i: t.grad for i, t in self.captured.items()}
