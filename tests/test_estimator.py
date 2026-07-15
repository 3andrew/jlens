"""Contract for JEstimator.update: the rank-1 samples must average to the
true Jacobian. Simulates a known linear map J so g = J^T v exactly; the
estimate must converge as 1/sqrt(N).

Remove the xfail marker when implementing update() — this test is the gate.
"""

import pytest
import torch

from jlens.estimator import JEstimator

D = 16


@pytest.mark.xfail(reason="JEstimator.update is the [Andrew writes] TODO", strict=False)
def test_converges_to_true_jacobian():
    torch.manual_seed(0)
    j_true = torch.randn(D, D)
    est = JEstimator([0], d_model=D)

    for _ in range(200):
        v = torch.randn(100, D)  # 100 probes, shaped like (positions, d)
        g = v @ j_true           # row-stacked g_i = J^T v_i
        est.update(v, {0: g})

    rel_err = (est.estimate(0) - j_true).norm() / j_true.norm()
    assert rel_err < 0.05, f"relative error {rel_err:.3f}"


@pytest.mark.xfail(reason="JEstimator.update is the [Andrew writes] TODO", strict=False)
def test_roundtrips_through_checkpoint(tmp_path):
    torch.manual_seed(1)
    est = JEstimator([0, 2], d_model=D)
    v = torch.randn(50, D)
    est.update(v, {0: torch.randn(50, D), 2: torch.randn(50, D)})

    est.save(tmp_path / "j.pt")
    loaded = JEstimator.load(tmp_path / "j.pt")
    assert torch.equal(loaded.estimate(0), est.estimate(0))
    assert loaded.count == est.count
