"""Contract for JLens.readout, on a tiny random model (CPU, no downloads).

The anchor: at the backward-source layer the estimated J converges to I, and
readout with the TRUE identity must reproduce the model's own logits exactly —
the lens is the model's output pipeline with J spliced in, so at J = I the
splice must vanish.

Remove the xfail markers when implementing readout.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from jlens.estimator import JEstimator
from jlens.hooks import ResidualCapture, freeze
from jlens.lens import JLens

D, LAYERS, VOCAB = 32, 6, 128


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=D,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=VOCAB,
    )
    model = freeze(LlamaForCausalLM(cfg)).float()
    est = JEstimator([1, 5], d_model=D)
    est.update(torch.eye(D), {1: torch.randn(D, D), 5: torch.eye(D)})
    ids = torch.randint(0, VOCAB, (1, 12))
    with torch.no_grad(), ResidualCapture(model, [1, 5], grads=False) as cap:
        out = model(ids, use_cache=False)
    return JLens(model, est), cap.captured, out.logits


def test_identity_readout_matches_model_logits(setup):
    lens, captured, logits = setup
    h_final = captured[5]
    ours = lens.readout(h_final, layer=5, identity=True)
    assert ours.shape == (1, 12, VOCAB)
    assert torch.allclose(ours, logits.float(), atol=1e-4)


def test_j_readout_differs_at_early_layer(setup):
    lens, captured, _ = setup
    h = captured[1]
    with_j = lens.readout(h, layer=1)
    without_j = lens.readout(h, layer=1, identity=True)
    assert with_j.shape == without_j.shape
    assert not torch.allclose(with_j, without_j, atol=1e-2)


def test_readout_accepts_bf16_activations(setup):
    """Production activations arrive as bf16 from MPS/CUDA capture."""
    lens, captured, _ = setup
    ours = lens.readout(captured[1].bfloat16(), layer=1)
    assert ours.dtype == torch.float32
    assert torch.isfinite(ours).all()


def test_topk_returns_decoded_tokens(setup):
    lens, captured, _ = setup

    class FakeTokenizer:
        def decode(self, ids):
            return f"<{ids[0]}>"

    entries = lens.topk(captured[5][0, -1], 5, FakeTokenizer(), k=4)
    assert len(entries) == 4
    assert all(isinstance(t, str) and 0 <= p <= 1 for t, p in entries)
