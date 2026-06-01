"""Residual one-step predictor and kinematic loss helpers.

The model is a one-step predictor:

    s_{t+1} = f_theta(s_t, u_t)

trained on nominal trajectories with the loss

    L = lambda_data * MSE(s_pred, s_target)
      + lambda_kin  * ||g_kin(s_t, s_pred)||^2
      + lambda_energy * ||g_energy(s_t, s_pred, u_t)||^2
      + lambda_smooth * ||s_pred - s_t||^2

where ``g_kin`` and ``g_energy`` are residual terms derived from the unicycle
model.  Using a one-step predictor lets us compute residuals analytically from
successive states without back-propagating through an integrator, which is
critical for keeping training fast on CPU.

For detection, the per-step anomaly score is

    score_t = w_pred * ||s_obs_t - s_pred_t|| + w_phys * ||phys_residuals||

where the weights ``(w_pred, w_phys)`` are learned on a held-out nominal set
to roughly match the magnitudes of the two terms (see
:meth:`PINNLoss.detection_score`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

CTRL_DIM = 2


class PINN(nn.Module):
    """MLP one-step predictor over (state, command)."""

    def __init__(
        self,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
        obs_dim: int = 6,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        in_dim = self.obs_dim + CTRL_DIM
        layers = []
        d_prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d_prev, hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_prev = hidden
        layers.append(nn.Linear(d_prev, self.obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Predict next state given current (state, command).

        Returns the residual added to ``s`` so the network only models the
        delta -- helps avoid biasing toward identity at init.
        """
        z = torch.cat([s, u], dim=-1)
        delta = self.net(z)
        return s + delta


@dataclass
class PINNLossConfigT:
    lambda_data: float = 1.0
    lambda_kin: float = 1.0
    lambda_energy: float = 0.5
    lambda_smooth: float = 0.05


class PINNLoss(nn.Module):
    """Composite loss with kinematic + energy residuals.

    The kinematic residual penalises violations of

        (s_pred[x] - s[x]) / dt - 0.5*(v_pred*cos(theta_pred) + v*cos(theta)) = 0
        (s_pred[y] - s[y]) / dt - 0.5*(v_pred*sin(theta_pred) + v*sin(theta)) = 0
        (s_pred[theta] - s[theta]) / dt - 0.5*(omega_pred + omega) = 0

    using the trapezoidal rule, which is what RK4 reproduces for affine
    derivatives.  The energy residual penalises violations of the dissipation
    inequality

        s_pred[batt] - s[batt] <= -dt * (e_idle + e_lin |v_mid| + e_ang |w_mid|).
    """

    def __init__(
        self,
        dt: float,
        energy_idle: float,
        energy_lin: float,
        energy_ang: float,
        cfg: PINNLossConfigT | None = None,
        obs_dim: int = 6,
    ):
        super().__init__()
        self.dt = float(dt)
        self.energy_idle = float(energy_idle)
        self.energy_lin = float(energy_lin)
        self.energy_ang = float(energy_ang)
        self.cfg = cfg or PINNLossConfigT()
        self.obs_dim = int(obs_dim)

    # ------------------------------------------------------------------
    # Residual computations
    # ------------------------------------------------------------------
    def _kin_residual(self, s: torch.Tensor, s_pred: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        x, y, th, v, w = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
        xp, yp, thp, vp, wp = (
            s_pred[..., 0], s_pred[..., 1], s_pred[..., 2], s_pred[..., 3], s_pred[..., 4],
        )
        rx = (xp - x) / dt - 0.5 * (v * torch.cos(th) + vp * torch.cos(thp))
        ry = (yp - y) / dt - 0.5 * (v * torch.sin(th) + vp * torch.sin(thp))
        rth = (thp - th) / dt - 0.5 * (w + wp)
        return torch.stack([rx, ry, rth], dim=-1)

    def _energy_residual(self, s: torch.Tensor, s_pred: torch.Tensor) -> torch.Tensor:
        if self.obs_dim < 6 or s.shape[-1] < 6 or s_pred.shape[-1] < 6:
            return torch.zeros_like(s[..., 0])
        dt = self.dt
        v, w = s[..., 3], s[..., 4]
        vp, wp = s_pred[..., 3], s_pred[..., 4]
        v_mid = 0.5 * (v + vp)
        w_mid = 0.5 * (w + wp)
        expected_drain = -dt * (
            self.energy_idle
            + self.energy_lin * torch.abs(v_mid)
            + self.energy_ang * torch.abs(w_mid)
        )
        actual_drain = s_pred[..., 5] - s[..., 5]
        # We penalise both directions: positive deviation (battery grew) and
        # negative deviation (drained faster than physics predicts).
        return actual_drain - expected_drain

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        s: torch.Tensor,
        s_pred: torch.Tensor,
        s_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        data = torch.mean((s_pred - s_target) ** 2)
        kin = torch.mean(self._kin_residual(s, s_pred) ** 2)
        eng = torch.mean(self._energy_residual(s, s_pred) ** 2)
        smooth = torch.mean((s_pred - s) ** 2)
        total = (
            self.cfg.lambda_data * data
            + self.cfg.lambda_kin * kin
            + (self.cfg.lambda_energy if self.obs_dim >= 6 else 0.0) * eng
            + self.cfg.lambda_smooth * smooth
        )
        return total, {
            "data": data.item(),
            "kin": kin.item(),
            "energy": eng.item(),
            "smooth": smooth.item(),
            "total": total.item(),
        }

    # ------------------------------------------------------------------
    # Detection score (no grad)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def detection_score(
        self,
        s: torch.Tensor,
        s_pred: torch.Tensor,
        s_obs_next: torch.Tensor,
        pred_scale: torch.Tensor | None = None,
        kin_scale: torch.Tensor | None = None,
        eng_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_err = torch.linalg.vector_norm(s_pred - s_obs_next, dim=-1)
        kin = torch.linalg.vector_norm(self._kin_residual(s, s_obs_next), dim=-1)
        eng = torch.abs(self._energy_residual(s, s_obs_next))
        def _n(x, scale):
            return x / (scale if scale is not None else (x.std() + 1e-6))
        return _n(pred_err, pred_scale) + _n(kin, kin_scale) + _n(eng, eng_scale)
