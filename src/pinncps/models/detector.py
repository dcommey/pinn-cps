"""Detector adapters that wrap neural predictors / autoencoders into
the unified ``score`` interface used by the evaluator.

Two kinds of neural detectors:

* ``NeuralPredictorDetector`` for one-step predictors (PINN, MLP, LSTM, GRU).
  For PINN we additionally combine kinematic / energy residuals.
* ``ReconstructionDetector`` for the LSTM autoencoder: per-step reconstruction
  error.

Both run a sliding-window forward pass over each trajectory and assemble a
per-timestep score by averaging contributions from each window that contains
the step.  This gives a smoother score than non-overlapping windows.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .pinn import PINN, PINNLoss


def _smooth(scores: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing uniform smoothing along the time axis.

    Smoothing helps separate persistent (small-amplitude) attacks from i.i.d.
    sensor-noise excursions.  Window is in time steps; ``window <= 1`` is a
    no-op.  The output at time ``t`` uses only scores from times ``<= t``.
    """
    if window <= 1:
        return scores
    k = int(window)
    if scores.ndim == 1:
        x = np.asarray(scores)
        cumsum = np.cumsum(np.insert(x, 0, 0.0))
        out = np.empty_like(x, dtype=np.float64)
        for t in range(x.shape[0]):
            start = max(0, t - k + 1)
            out[t] = (cumsum[t + 1] - cumsum[start]) / (t - start + 1)
        return out
    out = np.empty_like(scores)
    for i in range(scores.shape[0]):
        out[i] = _smooth(scores[i], window)
    return out


class NeuralPredictorDetector:
    """Adapter for one-step predictors.

    If a ``physics_loss`` is provided (PINN case) we combine prediction error
    with kinematic and energy residual magnitudes, with weights set so each
    component has unit standard deviation on the nominal validation data.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        kind: str,
        sequence: bool,
        physics_loss: Optional[PINNLoss] = None,
        device: str = "cpu",
        smooth_window: int = 5,
    ):
        self.model = model.to(device).eval()
        self.kind = kind  # "mlp" | "lstm" | "gru" | "pinn"
        self.sequence = sequence
        self.physics_loss = physics_loss
        self.device = device
        self.name = kind
        self.smooth_window = int(smooth_window)
        self._pred_scale: float = 1.0
        self._kin_scale: float = 1.0
        self._eng_scale: float = 1.0

    # --------------------------------------------------------------
    @torch.no_grad()
    def _one_step(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Predict next state from (state, command) batch."""
        if self.sequence:
            # Treat one step as a length-1 sequence.
            return self.model(s.unsqueeze(1), u.unsqueeze(1)).squeeze(1)
        return self.model(s, u)

    # --------------------------------------------------------------
    def calibrate(self, nominal: dict) -> None:
        """Set per-component scale factors from nominal data."""
        obs = nominal["obs"]
        cmd = nominal["commands"]
        N, Tp1, _ = obs.shape
        # Build a flat batch of all (s, u, s_next) triples.
        s = torch.from_numpy(obs[:, :-1].reshape(-1, obs.shape[-1])).float()
        u = torch.from_numpy(cmd.reshape(-1, cmd.shape[-1])).float()
        sn = torch.from_numpy(obs[:, 1:].reshape(-1, obs.shape[-1])).float()
        # Process in chunks to control memory.
        preds = []
        chunk = 4096
        for i in range(0, s.shape[0], chunk):
            preds.append(self._one_step(s[i:i + chunk].to(self.device), u[i:i + chunk].to(self.device)).cpu())
        s_pred = torch.cat(preds, dim=0)
        pred_err = torch.linalg.vector_norm(s_pred - sn, dim=-1)
        self._pred_scale = float(pred_err.std().clamp_min(1e-6))
        if self.physics_loss is not None:
            kin = torch.linalg.vector_norm(self.physics_loss._kin_residual(s, sn), dim=-1)
            eng = torch.abs(self.physics_loss._energy_residual(s, sn))
            self._kin_scale = float(kin.std().clamp_min(1e-6))
            self._eng_scale = float(eng.std().clamp_min(1e-6))
            if sn.shape[-1] < 6:
                self._eng_scale = 1.0

    # --------------------------------------------------------------
    @torch.no_grad()
    def score_batch(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        N, Tp1, D = obs.shape
        T = Tp1 - 1
        s = torch.from_numpy(obs[:, :-1].reshape(-1, D)).float().to(self.device)
        u = torch.from_numpy(commands.reshape(-1, commands.shape[-1])).float().to(self.device)
        sn = torch.from_numpy(obs[:, 1:].reshape(-1, D)).float().to(self.device)
        # Forward in chunks
        s_pred_chunks = []
        chunk = 4096
        for i in range(0, s.shape[0], chunk):
            s_pred_chunks.append(self._one_step(s[i:i + chunk], u[i:i + chunk]))
        s_pred = torch.cat(s_pred_chunks, dim=0)
        pred_err = torch.linalg.vector_norm(s_pred - sn, dim=-1) / self._pred_scale
        if self.physics_loss is not None:
            kin = torch.linalg.vector_norm(self.physics_loss._kin_residual(s, sn), dim=-1) / self._kin_scale
            eng = torch.abs(self.physics_loss._energy_residual(s, sn)) / self._eng_scale
            # The real-data PRM operating score is the analytic kinematic
            # residual. Prediction and energy channels are diagnostic/ablation
            # signals rather than the thresholded deployment score.
            score = kin + (eng if sn.shape[-1] >= 6 else 0.0)
        else:
            score = pred_err
        score = score.cpu().numpy().reshape(N, T)
        # Pad t=0 with the t=1 score.
        out = np.empty((N, Tp1), dtype=np.float64)
        out[:, 0] = score[:, 0]
        out[:, 1:] = score
        return _smooth(out, self.smooth_window)

    def score(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        return self.score_batch(obs[None], commands[None])[0]

    def fit(self, nominal: dict) -> None:  # training is external
        self.calibrate(nominal)


class ReconstructionDetector:
    """Adapter for the LSTM autoencoder.

    We run the AE over each contiguous length-L window (stride = 1) and average
    per-step reconstruction errors across the windows that cover each step.
    """

    name = "lstm_ae"

    def __init__(self, model, window: int, device: str = "cpu", smooth_window: int = 5):
        self.model = model.to(device).eval()
        self.window = window
        self.device = device
        self.smooth_window = int(smooth_window)
        self._scale: float = 1.0

    def calibrate(self, nominal: dict) -> None:
        scores = self._raw_score_batch(nominal["obs"], nominal["commands"])
        self._scale = float(np.std(scores) + 1e-6)

    @torch.no_grad()
    def _raw_score_batch(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        N, Tp1, D = obs.shape
        L = self.window
        out = np.zeros((N, Tp1), dtype=np.float64)
        counts = np.zeros((N, Tp1), dtype=np.float64)
        # Slide windows over each trajectory; align s and u to obs[0..L-1] and cmd[0..L-1].
        for t0 in range(0, max(1, Tp1 - L + 1)):
            t1 = t0 + L
            s_seq = torch.from_numpy(obs[:, t0:t1]).float().to(self.device)
            if t0 < commands.shape[1]:
                u_seq = torch.from_numpy(
                    commands[:, t0:min(t0 + L, commands.shape[1])]
                ).float().to(self.device)
                if u_seq.shape[1] < L:
                    # pad with last command
                    pad = u_seq[:, -1:].expand(-1, L - u_seq.shape[1], -1)
                    u_seq = torch.cat([u_seq, pad], dim=1)
            else:
                u_seq = torch.zeros((N, L, commands.shape[-1]), device=self.device)
            recon = self.model(s_seq, u_seq)
            err = torch.linalg.vector_norm(recon - s_seq, dim=-1)  # (N, L)
            out[:, t0:t1] += err.cpu().numpy()
            counts[:, t0:t1] += 1
        out /= np.maximum(counts, 1)
        return out

    def score_batch(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        raw = self._raw_score_batch(obs, commands)
        return _smooth(raw / self._scale, self.smooth_window)

    def score(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        return self.score_batch(obs[None], commands[None])[0]

    def fit(self, nominal: dict) -> None:
        self.calibrate(nominal)
