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

    Per-position probes: backward from sum_t v_t . h_final_t. The gradient at
    position t mixes Jacobian pullbacks of every downstream position's probe,
    but only the same-position pairing survives in expectation
    (E[v_t v_t'^T] = 0 for t' != t), so the estimate converges to the
    same-position Jacobian E[dh_final_t / dh_l_t]; cross-position terms add
    variance, not bias. One probe per (batch, position) makes the probe budget
    scale with prompts x seq_len — at d_model >= 1024, a shared per-prompt
    probe would need M >> d prompts to converge (measured: rel err 1.01 at
    M = 1000, exactly the sqrt((d+1)/M) prediction).
    """
    with ResidualCapture(model, layer_indices) as cap:
        model(input_ids, use_cache=False)

    h_final = cap.captured[layer_indices[-1]]
    v = torch.randn_like(h_final)

    (h_final * v).sum().backward()
    est.update(v, cap.grads())
