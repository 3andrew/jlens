"""End-to-end integration: probe_step drives hooks + estimator against a tiny
random model. At the backward source the gradient is exactly v, so the
estimate there must converge to the identity (E[vv^T] = I) — the same
invariant validate_j.py checks as Gate 1 on real runs.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from jlens.driver import probe_step
from jlens.estimator import JEstimator
from jlens.hooks import freeze

D, LAYERS, VOCAB, B, T = 32, 6, 128, 8, 16


def test_probe_step_end_to_end():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=D,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=VOCAB,
    )
    model = freeze(LlamaForCausalLM(cfg))
    idxs = [1, 3, 5]
    est = JEstimator(idxs, D)

    steps = 150
    for _ in range(steps):
        ids = torch.randint(0, VOCAB, (B, T))
        probe_step(model, idxs, ids, est)

    # Every (batch, position) pair counted, at every layer.
    assert est.count[1] == est.count[5] == steps * B * T

    # Gate 1 invariant: J at the backward source converges to I. Effective
    # sample count is steps*B distinct probes (positions share a probe),
    # so expected rel err ~ sqrt(d/(steps*B)) ~ 0.17.
    eye = torch.eye(D)
    ident_err = ((est.estimate(5) - eye).norm() / eye.norm()).item()
    assert ident_err < 0.25, f"backward-source J far from I: {ident_err:.3f}"

    # Upstream layers: finite, nonzero, and NOT identity (real layers act).
    for layer in (1, 3):
        j = est.estimate(layer)
        assert torch.isfinite(j).all()
        assert ((j - eye).norm() / eye.norm()).item() > ident_err
