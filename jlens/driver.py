"""The J-estimation inner loop, shared by driver scripts and tests."""

from __future__ import annotations

import torch

from jlens.estimator import JEstimator
from jlens.hooks import ResidualCapture


def probe_step(
    model,
    layer_indices: list[int],
    input_ids: torch.Tensor,
    est: JEstimator,
) -> None:
    """One estimation step: forward under capture, one backward, fold into est.

    One probe per batch element, shared across that element's positions: the
    gradient at position t mixes contributions from all downstream positions,
    so distinct per-position probes would scramble the v<->g pairing. Batch
    elements never attend to each other, so per-element probes are independent
    for free.
    """
    with ResidualCapture(model, layer_indices) as cap:
        model(input_ids, use_cache=False)

    h_final = cap.captured[layer_indices[-1]]
    v = torch.randn_like(h_final[:, :1, :])

    (h_final * v).sum().backward()
    est.update(v, cap.grads())
