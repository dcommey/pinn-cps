"""Training loops for one-step predictors and the LSTM autoencoder."""
from __future__ import annotations

import copy
import math
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import WindowDataset
from ..models.pinn import PINNLoss


def _make_loader(data: dict, window: int, stride: int, batch_size: int, shuffle: bool):
    ds = WindowDataset(data, window=window, stride=stride)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
    return loader


def train_predictor(
    model: torch.nn.Module,
    train_data: dict,
    val_data: dict,
    *,
    sequence: bool,
    physics_loss: Optional[PINNLoss] = None,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 6,
    window: int = 32,
    stride: int = 4,
    device: str = "cpu",
    log_every: int = 0,
) -> Dict:
    """Train a one-step predictor.

    Pointwise (MLP/PINN) and sequence (LSTM/GRU) share this loop -- the only
    difference is whether the per-batch loss is computed over each step in the
    window (sequence) or over the last step only (point).  For PINN we
    additionally weight kinematic + energy residuals via ``physics_loss``.
    """
    model = model.to(device)
    train_loader = _make_loader(train_data, window, stride, batch_size, shuffle=True)
    val_loader = _make_loader(val_data, window, stride, batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = math.inf
    best_state = None
    bad = 0
    history = {"train_loss": [], "val_loss": [], "train_components": [], "val_components": []}
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        train_total = 0.0
        train_components: Dict[str, float] = {}
        n_batches = 0
        for x_obs, x_cmd, target, _ in train_loader:
            x_obs = x_obs.to(device)
            x_cmd = x_cmd.to(device)
            target = target.to(device)
            if sequence:
                # Predict at every step in the window.
                s_pred_seq = model(x_obs, x_cmd)  # (B, L, D)
                pred = s_pred_seq
                s_in = x_obs
                tgt = target
            else:
                # Pointwise: flatten window into independent samples.
                B, L, D = x_obs.shape
                s_in = x_obs.reshape(B * L, D)
                u_in = x_cmd.reshape(B * L, x_cmd.shape[-1])
                tgt = target.reshape(B * L, D)
                pred = model(s_in, u_in)
            if physics_loss is not None:
                loss, comps = physics_loss(s_in.reshape(-1, s_in.shape[-1]),
                                           pred.reshape(-1, pred.shape[-1]),
                                           tgt.reshape(-1, tgt.shape[-1]))
            else:
                loss = torch.mean((pred - tgt) ** 2)
                comps = {"data": loss.item(), "total": loss.item()}
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_total += loss.item()
            for k, v in comps.items():
                train_components[k] = train_components.get(k, 0.0) + float(v)
            n_batches += 1
        sched.step()

        train_loss = train_total / max(n_batches, 1)
        train_components = {k: v / max(n_batches, 1) for k, v in train_components.items()}

        # Validation
        model.eval()
        val_total = 0.0
        val_components: Dict[str, float] = {}
        nv = 0
        with torch.no_grad():
            for x_obs, x_cmd, target, _ in val_loader:
                x_obs = x_obs.to(device); x_cmd = x_cmd.to(device); target = target.to(device)
                if sequence:
                    pred = model(x_obs, x_cmd)
                    s_in = x_obs
                    tgt = target
                else:
                    B, L, D = x_obs.shape
                    s_in = x_obs.reshape(B * L, D)
                    u_in = x_cmd.reshape(B * L, x_cmd.shape[-1])
                    tgt = target.reshape(B * L, D)
                    pred = model(s_in, u_in)
                if physics_loss is not None:
                    loss, comps = physics_loss(s_in.reshape(-1, s_in.shape[-1]),
                                               pred.reshape(-1, pred.shape[-1]),
                                               tgt.reshape(-1, tgt.shape[-1]))
                else:
                    loss = torch.mean((pred - tgt) ** 2)
                    comps = {"data": loss.item(), "total": loss.item()}
                val_total += loss.item()
                for k, v in comps.items():
                    val_components[k] = val_components.get(k, 0.0) + float(v)
                nv += 1
        val_loss = val_total / max(nv, 1)
        val_components = {k: v / max(nv, 1) for k, v in val_components.items()}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_components"].append(train_components)
        history["val_components"].append(val_components)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if log_every and (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"[ep {ep + 1:3d}] train {train_loss:.5f} val {val_loss:.5f} ({elapsed:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val"] = best_val
    history["wall_time"] = time.time() - t0
    return history


def train_autoencoder(
    model: torch.nn.Module,
    train_data: dict,
    val_data: dict,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 6,
    window: int = 32,
    stride: int = 4,
    device: str = "cpu",
) -> Dict:
    model = model.to(device)
    train_loader = _make_loader(train_data, window, stride, batch_size, shuffle=True)
    val_loader = _make_loader(val_data, window, stride, batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = math.inf
    best_state = None
    bad = 0
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        tt = 0.0; nb = 0
        for x_obs, x_cmd, _, _ in train_loader:
            x_obs = x_obs.to(device); x_cmd = x_cmd.to(device)
            recon = model(x_obs, x_cmd)
            loss = torch.mean((recon - x_obs) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tt += loss.item(); nb += 1
        sched.step()
        train_loss = tt / max(nb, 1)

        model.eval()
        vt = 0.0; nv = 0
        with torch.no_grad():
            for x_obs, x_cmd, _, _ in val_loader:
                x_obs = x_obs.to(device); x_cmd = x_cmd.to(device)
                recon = model(x_obs, x_cmd)
                loss = torch.mean((recon - x_obs) ** 2)
                vt += loss.item(); nv += 1
        val_loss = vt / max(nv, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val"] = best_val
    history["wall_time"] = time.time() - t0
    return history
