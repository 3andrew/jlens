"""Readout through the J-lens: translate residual-stream activations into
vocabulary distributions via lens(h) = softmax(W_U norm(J_l h)).

The lens borrows the model's own output machinery (final RMSNorm + unembedding)
and splices the estimated J_l in front of it. J = I gives the logit-lens
baseline. All readout math runs on CPU in fp32: J is estimated in fp32, and
readout is analysis, not a hot path.
"""

from __future__ import annotations

import torch
from torch import nn

from jlens.estimator import JEstimator


def final_norm(model: nn.Module) -> nn.Module:
    """The final pre-unembedding norm of an HF causal LM (Qwen/Llama-style)."""
    for path in ("model.norm", "transformer.ln_f"):
        mod = model
        try:
            for attr in path.split("."):
                mod = getattr(mod, attr)
        except AttributeError:
            continue
        return mod
    raise ValueError(f"could not locate final norm on {type(model).__name__}")


class JLens:
    """Vocabulary readout at every estimated layer.

    Holds CPU fp32 copies of everything readout needs: the J matrices, the
    final RMSNorm weight/eps, and the unembedding W_U.
    """

    def __init__(self, model: nn.Module, est: JEstimator):
        self.j = {layer: est.estimate(layer) for layer in est.accum}
        norm = final_norm(model)
        self.norm_w = norm.weight.detach().float().cpu()
        self.norm_eps = norm.variance_epsilon
        self.w_u = model.lm_head.weight.detach().float().cpu()  # (V, d)

    def readout(
        self, h: torch.Tensor, layer: int, identity: bool = False
    ) -> torch.Tensor:
        """Logits (..., V) for residual-stream activations h (..., d) at `layer`.

        identity=True skips J (the logit-lens baseline).

        TODO(andrew): the paper's central equation, ~6 lines.
          1. h to CPU fp32.
          2. Unless identity: h <- h @ J^T  (J rows act on h's last dim).
          3. RMSNorm: x / sqrt(mean(x^2, last dim) + eps) * norm_w.
          4. Logits: x @ W_U^T.
        tests/test_lens.py defines the contract: at the backward-source layer
        (J ~ I) the readout must match the model's own logits.
        """
        raise NotImplementedError

    def topk(
        self,
        h: torch.Tensor,
        layer: int,
        tokenizer,
        k: int = 8,
        identity: bool = False,
    ) -> list[tuple[str, float]]:
        """Top-k (token, prob) for a single activation h of shape (d,)."""
        probs = self.readout(h, layer, identity=identity).softmax(-1)
        top = probs.topk(k)
        return [
            (tokenizer.decode([i]), p.item())
            for i, p in zip(top.indices.tolist(), top.values)
        ]
