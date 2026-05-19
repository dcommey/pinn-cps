import numpy as np
import torch

from pinncps.models.pinn import PINN, PINNLoss, PINNLossConfigT


def test_forward_pass_shape():
    m = PINN(hidden=16, n_layers=2)
    s = torch.randn(4, 6)
    u = torch.randn(4, 2)
    out = m(s, u)
    assert out.shape == (4, 6)


def test_loss_components_finite():
    loss = PINNLoss(dt=0.1, energy_idle=0.05, energy_lin=0.1, energy_ang=0.05,
                    cfg=PINNLossConfigT())
    s = torch.randn(8, 6)
    s_target = s + 0.01 * torch.randn_like(s)
    s_pred = s + 0.005 * torch.randn_like(s)
    total, comps = loss(s, s_pred, s_target)
    assert torch.isfinite(total)
    for k in ("data", "kin", "energy", "smooth"):
        assert np.isfinite(comps[k])
