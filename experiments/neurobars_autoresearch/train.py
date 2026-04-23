from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.neurobars_autoresearch.prepare import (
    CACHE_DIR,
    EVAL_BATCH_SIZE,
    TIME_BUDGET,
    TRAIN_BATCH_SIZE,
    close_delta_mse,
    evaluate_model,
    load_feature_weights,
    load_metadata,
    make_dataloader,
    prepare_cache,
    weighted_next_bar_mse,
)

SEED = 42
LATENT_DIM = 32
HIDDEN_CHANNELS = 64
BASE_LAYERS = 4
FAST_LAYERS = 2
MID_LAYERS = 3
SLOW_LAYERS = 4
FAST_LOOKBACK = 32
MID_LOOKBACK = 64
MID_POOL = 1
SLOW_LOOKBACK = 128
SLOW_POOL = 1
FAST_LATENT_DIM = 12
MID_LATENT_DIM = 12
SLOW_LATENT_DIM = 8
DROPOUT = 0.15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
CHECKPOINT_PATH = Path(
    os.environ.get("NEUROBARS_AUTORESEARCH_CHECKPOINT_PATH", str(ROOT / "checkpoints" / "neurobars_autoresearch_latest.pt"))
)
METRICS_PATH = Path(
    os.environ.get(
        "NEUROBARS_AUTORESEARCH_METRICS_PATH",
        str(ROOT / "checkpoints" / "neurobars_autoresearch_latest_metrics.json"),
    )
)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self._padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        result = super().forward(x)
        if self._padding != 0:
            return result[:, :, :-self._padding]
        return result


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.activation(self.conv1(x))
        out = self.dropout(out)
        out = self.activation(self.conv2(out))
        out = self.dropout(out)
        return out + residual


class NeurobarEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_channels: int, latent_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_channels = input_dim
        for layer_idx in range(num_layers):
            layers.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    dilation=2 ** layer_idx,
                    dropout=dropout,
                )
            )
            in_channels = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.latent_proj = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.tcn(x)
        last_features = features[:, :, -1]
        latent = self.latent_proj(last_features)
        return F.layer_norm(latent, latent.shape[1:])


class MultiScaleNeurobarEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, HIDDEN_CHANNELS, kernel_size=1),
            nn.GELU(),
        )
        self.base_encoder = NeurobarEncoder(
            input_dim=HIDDEN_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            latent_dim=LATENT_DIM,
            num_layers=BASE_LAYERS,
            dropout=DROPOUT,
        )
        self.fast_encoder = NeurobarEncoder(
            input_dim=HIDDEN_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            latent_dim=FAST_LATENT_DIM,
            num_layers=FAST_LAYERS,
            dropout=DROPOUT,
        )
        self.mid_encoder = NeurobarEncoder(
            input_dim=HIDDEN_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            latent_dim=MID_LATENT_DIM,
            num_layers=MID_LAYERS,
            dropout=DROPOUT,
        )
        self.slow_encoder = NeurobarEncoder(
            input_dim=HIDDEN_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            latent_dim=SLOW_LATENT_DIM,
            num_layers=SLOW_LAYERS,
            dropout=DROPOUT,
        )
        self.multi_scale_residual = nn.Sequential(
            nn.Linear(FAST_LATENT_DIM + MID_LATENT_DIM + SLOW_LATENT_DIM, LATENT_DIM * 2),
            nn.GELU(),
            nn.Linear(LATENT_DIM * 2, LATENT_DIM),
        )

    @property
    def base_multiscale_dim(self) -> int:
        return LATENT_DIM + FAST_LATENT_DIM + MID_LATENT_DIM + SLOW_LATENT_DIM

    @staticmethod
    def _aggregate_view(x: torch.Tensor, *, lookback: int, pool: int) -> torch.Tensor:
        if lookback <= 0 or pool <= 0:
            raise ValueError("lookback and pool must be positive")

        if x.size(2) < lookback:
            raise ValueError(f"Input sequence too short for lookback={lookback}")

        view = x[:, :, -lookback:]
        if pool == 1:
            return view

        trimmed = (view.size(2) // pool) * pool
        if trimmed <= 0:
            raise ValueError(f"Pooling factor {pool} is too large for lookback={lookback}")
        view = view[:, :, -trimmed:]
        bsz, channels, length = view.shape
        view = view.reshape(bsz, channels, length // pool, pool)
        return view.mean(dim=-1)

    def encode_components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.transpose(1, 2)
        shared = self.stem(x)
        base_latent = self.base_encoder(shared.transpose(1, 2))

        fast_view = self._aggregate_view(shared, lookback=FAST_LOOKBACK, pool=1).transpose(1, 2)
        mid_view = self._aggregate_view(shared, lookback=MID_LOOKBACK, pool=MID_POOL).transpose(1, 2)
        slow_view = self._aggregate_view(shared, lookback=SLOW_LOOKBACK, pool=SLOW_POOL).transpose(1, 2)

        fast_latent = self.fast_encoder(fast_view)
        mid_latent = self.mid_encoder(mid_view)
        slow_latent = self.slow_encoder(slow_view)

        branch_context = torch.cat([fast_latent, mid_latent, slow_latent], dim=1)
        branch_context = self.multi_scale_residual(branch_context)
        fused = base_latent + 0.5 * branch_context
        fused = F.layer_norm(fused, fused.shape[1:])
        return {
            "base": base_latent,
            "fast": fast_latent,
            "mid": mid_latent,
            "slow": slow_latent,
            "fused": fused,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_components(x)["fused"]


class NeurobarPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = MultiScaleNeurobarEncoder(input_dim=input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.GELU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        pred = self.decoder(latent)
        return latent, pred


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def num_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train() -> None:
    set_seed(SEED)
    prepare_cache()
    metadata = load_metadata()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    model = NeurobarPredictor(input_dim=metadata.input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    feature_weights = load_feature_weights(device=device)

    loader = make_dataloader("train", batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    iterator = iter(loader)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    )

    total_steps = 0
    train_loss_sum = 0.0
    training_start = time.time()
    total_start = time.time()

    while True:
        elapsed = time.time() - training_start
        if total_steps > 0 and elapsed >= TIME_BUDGET:
            break

        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context:
            _, pred = model(x)
            loss_weighted = weighted_next_bar_mse(pred, y, feature_weights)
            loss_close_delta = close_delta_mse(x, pred, y, close_idx=metadata.close_idx)
            loss = loss_weighted + 0.5 * loss_close_delta

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        total_steps += 1
        train_loss_sum += float(loss.item())

    training_seconds = time.time() - training_start
    metrics = evaluate_model(model, device=device, batch_size=EVAL_BATCH_SIZE)
    total_seconds = time.time() - total_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metrics.__dict__,
            "config": {
                "encoder_type": "base_plus_multiscale_horizon_branches",
                "base_layers": BASE_LAYERS,
                "latent_dim": LATENT_DIM,
                "hidden_channels": HIDDEN_CHANNELS,
                "fast_layers": FAST_LAYERS,
                "mid_layers": MID_LAYERS,
                "slow_layers": SLOW_LAYERS,
                "fast_lookback": FAST_LOOKBACK,
                "mid_lookback": MID_LOOKBACK,
                "mid_pool": MID_POOL,
                "slow_lookback": SLOW_LOOKBACK,
                "slow_pool": SLOW_POOL,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        },
        CHECKPOINT_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics.__dict__, indent=2))

    print("---")
    print(f"val_score:             {metrics.val_score:.6f}")
    print(f"val_weighted_mse:      {metrics.val_weighted_mse:.6f}")
    print(f"val_close_delta_mse:   {metrics.val_close_delta_mse:.6f}")
    print(f"val_direction_acc:     {metrics.val_direction_acc:.6f}")
    print(f"training_seconds:      {training_seconds:.1f}")
    print(f"total_seconds:         {total_seconds:.1f}")
    print(f"peak_vram_mb:          {peak_vram_mb:.1f}")
    print(f"num_steps:             {total_steps}")
    print(f"avg_train_loss:        {train_loss_sum / max(total_steps, 1):.6f}")
    print(f"num_params_M:          {num_parameters(model) / 1e6:.3f}")
    print("encoder_type:          base_plus_multiscale_horizon_branches")
    print(f"latent_dim:            {LATENT_DIM}")
    print(f"checkpoint_path:       {CHECKPOINT_PATH}")
    print(f"metrics_path:          {METRICS_PATH}")


if __name__ == "__main__":
    train()
