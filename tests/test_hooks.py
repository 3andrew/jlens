"""Phase 0 gate: hooks capture the residual stream and gradients flow,
on CPU, with a tiny randomly-initialized Llama-style model (no download)."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from jlens.hooks import ResidualCapture, decoder_layers, freeze

D_MODEL, N_LAYERS, VOCAB = 32, 6, 128


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=D_MODEL,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=VOCAB,
    )
    return freeze(LlamaForCausalLM(cfg))


def test_finds_decoder_layers(model):
    assert len(decoder_layers(model)) == N_LAYERS


def test_capture_shapes_and_gradient_flow(model):
    idxs = [1, 3, 5]
    ids = torch.randint(0, VOCAB, (2, 10))

    with ResidualCapture(model, idxs) as cap:
        model(input_ids=ids, use_cache=False)

    assert set(cap.captured) == set(idxs)
    for t in cap.captured.values():
        assert t.shape == (2, 10, D_MODEL)

    v = torch.randn(D_MODEL)
    h_final = cap.captured[5]
    (h_final * v).sum().backward()

    grads = cap.grads()
    for i in idxs:
        assert grads[i].shape == (2, 10, D_MODEL)
    # At the layer we backward from, the gradient is v itself, broadcast.
    assert torch.allclose(grads[5], v.expand(2, 10, D_MODEL))
    # Frozen weights accumulate nothing.
    assert all(p.grad is None for p in model.parameters())


def test_releaf_cuts_graph_below(model):
    ids = torch.randint(0, VOCAB, (1, 8))
    with ResidualCapture(model, [2, 4]) as cap:
        model(input_ids=ids, use_cache=False)
    # Releaf point is a fresh leaf; later captures descend from it.
    assert cap.captured[2].is_leaf and cap.captured[2].requires_grad
    assert cap.captured[4].grad_fn is not None


def test_hooks_removed_on_exit(model):
    ids = torch.randint(0, VOCAB, (1, 8))
    with ResidualCapture(model, [1]) as cap:
        model(input_ids=ids, use_cache=False)
    before = dict(cap.captured)
    model(input_ids=ids, use_cache=False)  # outside the context: no capture
    assert cap.captured.keys() == before.keys()
