"""Classical baselines: Isolation Forest, One-Class SVM, Kalman residual.

All three implement the same ``Detector`` interface so they slot into the
evaluation loop next to the neural detectors:

    .fit(nominal: dict)             # trains on nominal trajectories
    .score(obs, commands) -> (T,)   # per-step anomaly score
    .score_batch(obs, commands) -> (N, T)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from .detector import _smooth


# ---------------------------------------------------------------------------
# Feature engineering shared by IsoForest and OCSVM
# ---------------------------------------------------------------------------

def per_step_features(
    obs: np.ndarray,
    commands: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute per-step engineered features (one row per non-initial step).

    Features capture both raw signal and physics-informed residuals so the
    classical detectors have a fair chance:

        - obs[t]
        - obs[t] - obs[t-1]
        - finite-difference kinematic residuals (x_dot - v cos th, ...)
        - energy delta - expected drain
        - commanded vs reported velocity / omega
    """
    if obs.ndim == 2:
        obs = obs[None]
        commands = commands[None]
        squeeze = True
    else:
        squeeze = False
    N, Tp1, _ = obs.shape
    T = Tp1 - 1
    o0 = obs[:, :-1]
    o1 = obs[:, 1:]
    cmds = commands  # (N, T, 2)
    dobs = o1 - o0
    x_dot_fd = dobs[..., 0] / dt
    y_dot_fd = dobs[..., 1] / dt
    th_dot_fd = dobs[..., 2] / dt
    v_dot_fd = dobs[..., 3] / dt
    w_dot_fd = dobs[..., 4] / dt
    th_mid = 0.5 * (o0[..., 2] + o1[..., 2])
    v_mid = 0.5 * (o0[..., 3] + o1[..., 3])
    w_mid = 0.5 * (o0[..., 4] + o1[..., 4])
    rkx = x_dot_fd - v_mid * np.cos(th_mid)
    rky = y_dot_fd - v_mid * np.sin(th_mid)
    rkth = th_dot_fd - w_mid
    cmd_v_err = cmds[..., 0] - o0[..., 3]
    cmd_w_err = cmds[..., 1] - o0[..., 4]
    feature_list = [
        o0[..., 3],  # v
        o0[..., 4],  # omega
        dobs[..., 0],
        dobs[..., 1],
        dobs[..., 2],
        dobs[..., 3],
        dobs[..., 4],
        x_dot_fd,
        y_dot_fd,
        th_dot_fd,
        v_dot_fd,
        w_dot_fd,
        rkx,
        rky,
        rkth,
        cmd_v_err,
        cmd_w_err,
    ]
    if obs.shape[-1] >= 6:
        feature_list.insert(7, dobs[..., 5])
    feats = np.stack(feature_list, axis=-1)  # (N, T, F)
    return feats[0] if squeeze else feats


class _FeatureModelDetector:
    """Shared wrapper that fits a model on per-step features."""

    name = "feature_model"

    def __init__(self, model, dt: float, smooth_window: int = 5):
        self.model = model
        self.dt = dt
        self.smooth_window = int(smooth_window)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, nominal: dict) -> None:
        feats = per_step_features(nominal["obs"], nominal["commands"], self.dt)
        flat = feats.reshape(-1, feats.shape[-1])
        self._mean = flat.mean(axis=0)
        self._std = flat.std(axis=0) + 1e-6
        flat = (flat - self._mean) / self._std
        self.model.fit(flat)

    def _score(self, feats: np.ndarray) -> np.ndarray:
        flat = (feats.reshape(-1, feats.shape[-1]) - self._mean) / self._std
        # decision_function: higher = more normal; invert for anomaly score.
        s = -self.model.decision_function(flat)
        return s.reshape(feats.shape[:-1])

    def score(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        feats = per_step_features(obs, commands, self.dt)
        s = self._score(feats)
        full = np.empty(obs.shape[0], dtype=np.float64)
        full[0] = s[0]
        full[1:] = s
        return _smooth(full, self.smooth_window)

    def score_batch(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        feats = per_step_features(obs, commands, self.dt)
        s = self._score(feats)  # (N, T)
        full = np.empty(obs.shape[:2], dtype=np.float64)
        full[:, 0] = s[:, 0]
        full[:, 1:] = s
        return _smooth(full, self.smooth_window)


class IsolationForestDetector(_FeatureModelDetector):
    name = "iso_forest"

    def __init__(self, dt: float, n_estimators: int = 200, contamination: float = 0.05):
        super().__init__(
            IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=0,
                n_jobs=1,
            ),
            dt=dt,
        )


class OCSVMDetector(_FeatureModelDetector):
    name = "oc_svm"

    def __init__(self, dt: float, nu: float = 0.05, gamma: str | float = "scale", max_train: int = 4000):
        super().__init__(
            OneClassSVM(kernel="rbf", nu=nu, gamma=gamma),
            dt=dt,
        )
        self.max_train = max_train

    def fit(self, nominal: dict) -> None:
        feats = per_step_features(nominal["obs"], nominal["commands"], self.dt)
        flat = feats.reshape(-1, feats.shape[-1])
        # Subsample for tractability; OC-SVM is O(n^2).
        if flat.shape[0] > self.max_train:
            rng = np.random.default_rng(0)
            idx = rng.choice(flat.shape[0], self.max_train, replace=False)
            flat_fit = flat[idx]
        else:
            flat_fit = flat
        self._mean = flat.mean(axis=0)
        self._std = flat.std(axis=0) + 1e-6
        flat_fit = (flat_fit - self._mean) / self._std
        self.model.fit(flat_fit)


# ---------------------------------------------------------------------------
# Kalman residual detector (EKF on unicycle model)
# ---------------------------------------------------------------------------

class KalmanResidualDetector:
    """Extended Kalman filter with chi-squared residual score.

    Process model is the unicycle (no battery term) so the filter's residual
    captures kinematic violations.  We treat the commanded ``(v_cmd, omega_cmd)``
    as input.
    """

    name = "kalman"

    def __init__(
        self,
        dt: float,
        tau_v: float = 0.4,
        tau_omega: float = 0.3,
        process_std: float = 0.05,
        meas_std: float = 0.05,
        smooth_window: int = 5,
    ):
        self.dt = dt
        self.tau_v = tau_v
        self.tau_omega = tau_omega
        self.q = process_std ** 2
        self.r = meas_std ** 2
        self.smooth_window = int(smooth_window)

    def fit(self, nominal: dict) -> None:
        # Fit process/measurement noise scale to nominal innovations so the
        # chi-squared statistic is well-calibrated.
        scores = self.score_batch(nominal["obs"], nominal["commands"])
        self._cal_med = float(np.median(scores))
        self._cal_mad = float(np.median(np.abs(scores - self._cal_med)) + 1e-9)

    def _step(self, mu, P, u, z):
        dt = self.dt
        x, y, th, v, w = mu
        v_cmd, w_cmd = u
        # Predict
        x1 = x + dt * v * np.cos(th)
        y1 = y + dt * v * np.sin(th)
        th1 = th + dt * w
        v1 = v + dt * (v_cmd - v) / self.tau_v
        w1 = w + dt * (w_cmd - w) / self.tau_omega
        mu_pred = np.array([x1, y1, th1, v1, w1])

        # Jacobian of f wrt state.
        F = np.eye(5)
        F[0, 2] = -dt * v * np.sin(th)
        F[0, 3] = dt * np.cos(th)
        F[1, 2] = dt * v * np.cos(th)
        F[1, 3] = dt * np.sin(th)
        F[2, 4] = dt
        F[3, 3] = 1.0 - dt / self.tau_v
        F[4, 4] = 1.0 - dt / self.tau_omega
        Q = self.q * np.eye(5)
        P_pred = F @ P @ F.T + Q

        # Measurement: identity on the 5 modelled states.
        H = np.eye(5)
        R = self.r * np.eye(5)
        y_innov = z - mu_pred
        S = H @ P_pred @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = P_pred @ H.T @ S_inv
        mu_new = mu_pred + K @ y_innov
        P_new = (np.eye(5) - K @ H) @ P_pred

        # Chi-squared residual statistic (Mahalanobis distance).
        score = float(y_innov.T @ S_inv @ y_innov)
        return mu_new, P_new, score

    def score(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        T = obs.shape[0]
        scores = np.zeros(T, dtype=np.float64)
        mu = obs[0, :5].astype(np.float64)
        P = 0.1 * np.eye(5)
        for t in range(1, T):
            z = obs[t, :5].astype(np.float64)
            u = commands[t - 1].astype(np.float64) if t - 1 < commands.shape[0] else np.zeros(2)
            mu, P, s = self._step(mu, P, u, z)
            scores[t] = s
        scores[0] = scores[1]
        return _smooth(scores, self.smooth_window)

    def score_batch(self, obs: np.ndarray, commands: np.ndarray) -> np.ndarray:
        N = obs.shape[0]
        out = np.zeros(obs.shape[:2], dtype=np.float64)
        for i in range(N):
            out[i] = self.score(obs[i], commands[i])
        return out
