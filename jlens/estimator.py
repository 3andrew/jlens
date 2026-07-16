"""Stochastic estimation of the context-averaged Jacobian J_l.

For a random probe v ~ N(0, I_d), one backward pass from (v * h_final).sum()
yields g_l = J_l^T v at every hooked layer at once, and E[v g^T] = J_l
because E[v v^T] = I. The accumulator averages these rank-1 samples in
fp32 on CPU.
"""

from __future__ import annotations

from pathlib import Path

import torch


class JEstimator:
    """Running J_l estimate per hooked layer, accumulated on CPU in fp32."""

    def __init__(self, layer_indices: list[int], d_model: int):
        self.d = d_model
        self.accum = {i: torch.zeros(d_model, d_model) for i in layer_indices}
        self.count = dict.fromkeys(layer_indices, 0)

    def update(self, v: torch.Tensor, grads: dict[int, torch.Tensor]) -> None:
        """Fold one backward pass's gradients into the running estimate.

        v: the probe, shape (d,) or broadcastable to each grad's shape.
        grads: layer index -> gradient tensor of shape (..., d), where every
        leading position (batch element x sequence position) is one rank-1
        sample paired with the probe.
        """
        for layer in grads:
            g = grads[layer]
            V = v.broadcast_to(g.shape).reshape(-1, self.d)
            G = g.reshape(-1, self.d).float()
            N = G.size()[0]
            acc = V.float().transpose(0, 1) @ G
            self.accum[layer] += acc.to(device="cpu")
            self.count[layer] += N

    def estimate(self, layer: int) -> torch.Tensor:
        if self.count[layer] == 0:
            raise RuntimeError(f"no samples accumulated for layer {layer}")
        return self.accum[layer] / self.count[layer]

    def save(self, path: str | Path) -> None:
        torch.save({"accum": self.accum, "count": self.count, "d": self.d}, path)

    @classmethod
    def load(cls, path: str | Path) -> "JEstimator":
        state = torch.load(path, weights_only=True)
        est = cls(list(state["accum"]), state["d"])
        est.accum, est.count = state["accum"], state["count"]
        return est
