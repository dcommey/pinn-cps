"""Neural baselines: MLP, LSTM, GRU predictors and an LSTM autoencoder."""
from __future__ import annotations

import torch
import torch.nn as nn

CTRL_DIM = 2


class MLPPredictor(nn.Module):
    """Same architecture as PINN but no physics loss."""

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
        return s + self.net(torch.cat([s, u], dim=-1))


class _RNNPredictor(nn.Module):
    def __init__(self, cell: str, hidden: int, n_layers: int, dropout: float, obs_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        in_dim = self.obs_dim + CTRL_DIM
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, self.obs_dim)

    def forward(self, s_seq: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        """Given length-L windows of states + commands, predict the next state at
        each timestep.  Output shape: (B, L, OBS_DIM)."""
        z = torch.cat([s_seq, u_seq], dim=-1)
        h, _ = self.rnn(z)
        delta = self.head(h)
        return s_seq + delta


class LSTMPredictor(_RNNPredictor):
    def __init__(self, hidden: int = 64, n_layers: int = 1, dropout: float = 0.0, obs_dim: int = 6):
        super().__init__("lstm", hidden, n_layers, dropout, obs_dim)


class GRUPredictor(_RNNPredictor):
    def __init__(self, hidden: int = 64, n_layers: int = 1, dropout: float = 0.0, obs_dim: int = 6):
        super().__init__("gru", hidden, n_layers, dropout, obs_dim)


class LSTMAutoencoder(nn.Module):
    """Bottleneck LSTM autoencoder over (obs, command) sequences.

    Implements the standard pattern from time-series anomaly detection (Malhotra
    et al., 2016): the encoder collapses the window into a fixed-size latent,
    and the decoder reconstructs the entire observation sequence from that
    latent vector replicated across timesteps.  No teacher forcing — the
    decoder has nowhere to "cheat", so the reconstruction error is genuinely
    a measure of how well the latent captured the window.

    We standardise the observation inputs by per-channel running statistics
    estimated on the first batch so the network does not have to learn the
    O(10^2) battery scale from zero-initialised weights.
    """

    def __init__(self, hidden: int = 64, latent: int = 16, dropout: float = 0.0, obs_dim: int = 6):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.encoder = nn.LSTM(
            input_size=self.obs_dim + CTRL_DIM,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, self.obs_dim)
        # Running standardisation buffers; updated once on the first batch.
        self.register_buffer("mu", torch.zeros(self.obs_dim))
        self.register_buffer("sd", torch.ones(self.obs_dim))
        self.register_buffer("_init", torch.tensor(0, dtype=torch.long))

    def _maybe_init_stats(self, s_seq: torch.Tensor) -> None:
        if int(self._init.item()) == 0 and self.training:
            with torch.no_grad():
                flat = s_seq.reshape(-1, s_seq.shape[-1])
                self.mu.copy_(flat.mean(dim=0))
                self.sd.copy_(flat.std(dim=0).clamp_min(1e-2))
                self._init.fill_(1)

    def forward(self, s_seq: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        self._maybe_init_stats(s_seq)
        s_norm = (s_seq - self.mu) / self.sd
        z = torch.cat([s_norm, u_seq], dim=-1)
        _, (h, _c) = self.encoder(z)
        latent = self.to_latent(h[-1])
        h0 = self.from_latent(latent).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        B, L, _ = s_seq.shape
        # Decoder input: broadcast the encoder hidden state across all L steps.
        dec_in = h0.transpose(0, 1).expand(B, L, -1).contiguous()
        d, _ = self.decoder(dec_in, (h0, c0))
        recon_norm = self.head(d)
        recon = recon_norm * self.sd + self.mu
        return recon
