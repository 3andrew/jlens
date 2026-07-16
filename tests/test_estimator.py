"""Contract for JEstimator.update: rank-1 samples must average to the true
Jacobian for every shape production will feed it — stacked probes (N, d),
a single probe (d,) shared across (batch, seq) gradients, and bf16 inputs
straight off a GPU backward pass. Simulates a known linear map J so
g = J^T v exactly; estimates must converge as 1/sqrt(N).
"""

import torch

from jlens.estimator import JEstimator

D = 16


def rel_err(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    return ((estimate - truth).norm() / truth.norm()).item()


def test_converges_with_stacked_probes():
    torch.manual_seed(0)
    j_true = torch.randn(D, D)
    est = JEstimator([0], d_model=D)

    for _ in range(200):
        v = torch.randn(100, D)  # 100 independent probes, one per row
        g = v @ j_true           # row-stacked g_i = J^T v_i
        est.update(v, {0: g})

    assert rel_err(est.estimate(0), j_true) < 0.05


def test_converges_with_shared_probe_and_batch_seq_grads():
    """The production shape: one probe (d,), grads (batch, seq, d)."""
    torch.manual_seed(2)
    j_true = torch.randn(D, D)
    est = JEstimator([0], d_model=D)

    for _ in range(1000):
        v = torch.randn(D)
        g = (v @ j_true).expand(4, 10, D)  # every position sees g = J^T v
        est.update(v, {0: g})

    # One probe direction per update converges slower than 100 independent
    # ones, hence the looser threshold.
    assert rel_err(est.estimate(0), j_true) < 0.2


def test_count_is_batch_times_seq():
    est = JEstimator([0], d_model=D)
    est.update(torch.randn(D), {0: torch.randn(2, 3, D)})
    assert est.count[0] == 6


def test_two_updates_equal_one_big_update():
    torch.manual_seed(3)
    v, g = torch.randn(100, D), torch.randn(100, D)

    one = JEstimator([0], d_model=D)
    one.update(v, {0: g})
    two = JEstimator([0], d_model=D)
    two.update(v[:50], {0: g[:50]})
    two.update(v[50:], {0: g[50:]})

    assert torch.allclose(one.estimate(0), two.estimate(0), atol=1e-5)
    assert one.count == two.count


def test_bf16_inputs_accumulate_in_fp32():
    torch.manual_seed(4)
    j_true = torch.randn(D, D)
    est = JEstimator([0], d_model=D)

    for _ in range(200):
        v = torch.randn(100, D)
        g = v @ j_true
        est.update(v.bfloat16(), {0: g.bfloat16()})

    assert est.accum[0].dtype == torch.float32
    assert rel_err(est.estimate(0), j_true) < 0.1


def test_roundtrips_through_checkpoint(tmp_path):
    torch.manual_seed(1)
    est = JEstimator([0, 2], d_model=D)
    v = torch.randn(50, D)
    est.update(v, {0: torch.randn(50, D), 2: torch.randn(50, D)})

    est.save(tmp_path / "j.pt")
    loaded = JEstimator.load(tmp_path / "j.pt")
    assert torch.equal(loaded.estimate(0), est.estimate(0))
    assert loaded.count == est.count
