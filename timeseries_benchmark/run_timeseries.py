from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    DEFAULT_FEATURE_COLUMNS,
    NormalizeMethod,
    load_market_json,
    normalize_market_frame,
)
from src.model import ModelConfig, UnitaryTransformer
from src.self_state import PositionInfo, PsychoConfig, PsychoModule


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: str = "data/BTCUSDT_2023-05-31_01-40_to_2025-04-24_12-19.json"
    seq_len: int = 128
    stride: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalize: NormalizeMethod = "log_return"
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.1
    weight_price: float = 1.0
    weight_self: float = 1.0
    self_dim: int = 6
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 512
    use_flash: bool = True
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str = "cuda"
    log_interval: int = 50
    output_dir: str = "timeseries_output"
    initial_balance: float = 1.0
    exposure_fraction: float = 0.2
    leverage: float = 3.0
    cortisol_decay: float = 0.97
    pain_scale: float = 4.0


class WindowedMarketDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        *,
        seq_len: int,
        start_row: int,
        end_row: int,
        stride: int,
    ) -> None:
        if data.ndim != 2:
            raise ValueError("data must be (rows, features).")
        if start_row < 0 or end_row > data.shape[0]:
            raise ValueError("Split rows exceed dataset bounds.")
        window = seq_len + 1
        max_start = end_row - window
        if max_start < start_row:
            raise ValueError(
                "Not enough rows for requested window: "
                f"rows={end_row - start_row} window={window}"
            )
        self.data = data
        self.seq_len = seq_len
        self.starts = list(range(start_row, max_start + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        window = self.data[start : start + self.seq_len + 1]
        return window[:-1], window[-1]


def load_market_tensor(
    path: str | Path,
    *,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    normalize: NormalizeMethod = "log_return",
) -> tuple[torch.Tensor, dict[str, object]]:
    df = load_market_json(path)
    if df.is_empty():
        raise ValueError(f"Market data is empty: {path}")
    columns = list(feature_columns)
    df = df.select(columns)
    df, stats = normalize_market_frame(
        df,
        columns,
        method=normalize,
    )
    df = df.with_columns(pl.col(columns).cast(pl.Float32))
    arr = df.select(columns).to_numpy()
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return torch.from_numpy(arr), stats


def split_rows(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int, int]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return 0, train_end, val_end, total


def build_loaders(config: ExperimentConfig) -> tuple[dict[str, DataLoader], dict[str, int], dict[str, object]]:
    data, stats = load_market_tensor(
        config.data_path,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        normalize=config.normalize,
    )
    start, train_end, val_end, end = split_rows(
        data.shape[0],
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
    )
    datasets = {
        "train": WindowedMarketDataset(
            data,
            seq_len=config.seq_len,
            start_row=start,
            end_row=train_end,
            stride=config.stride,
        ),
        "val": WindowedMarketDataset(
            data,
            seq_len=config.seq_len,
            start_row=train_end,
            end_row=val_end,
            stride=config.stride,
        ),
        "test": WindowedMarketDataset(
            data,
            seq_len=config.seq_len,
            start_row=val_end,
            end_row=end,
            stride=config.stride,
        ),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            drop_last=(name == "train"),
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )
        for name, dataset in datasets.items()
    }
    counts = {name: len(dataset) for name, dataset in datasets.items()}
    return loaders, counts, stats


def build_model(config: ExperimentConfig, market_dim: int) -> nn.Module:
    model_config = ModelConfig(
        market_dim=market_dim,
        self_dim=config.self_dim,
        action_dim=3,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        use_flash=config.use_flash,
    )
    model = UnitaryTransformer(model_config)
    if config.compile_model:
        model = torch.compile(model, mode=config.compile_mode)
    return model


def compute_self_targets(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    psycho: PsychoModule,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    position = PositionInfo(
        initial_balance=config.initial_balance,
        position_size=config.initial_balance * config.exposure_fraction,
        entry_price=1.0,
        leverage=config.leverage,
    )
    prev_state = psycho(inputs, position)
    window = torch.cat([inputs, targets.unsqueeze(1)], dim=1)
    target_state = psycho(
        window,
        position,
        expected_pnl=prev_state.pnl_unrealized,
        prev_cortisol=prev_state.cortisol,
    )
    return prev_state.as_tensor(), target_state.as_tensor()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: ExperimentConfig,
    psycho: PsychoModule | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_price = 0.0
    total_self = 0.0
    steps = 0
    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prev_self = None
        target_self = None
        if psycho is not None and config.weight_self > 0:
            prev_self, target_self = compute_self_targets(inputs, targets, psycho, config)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            _, new_self, price_pred = model(inputs, prev_self, return_price=True)
            price_loss = F.mse_loss(price_pred, targets)
            if target_self is not None:
                if new_self.ndim == 3 and target_self.ndim == 2:
                    target_self = target_self.unsqueeze(1).expand(-1, new_self.shape[1], -1)
                self_loss = F.mse_loss(new_self, target_self)
            else:
                self_loss = price_pred.new_zeros(())
            loss = config.weight_price * price_loss + config.weight_self * self_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_price += price_loss.item()
        total_self += self_loss.item()
        steps += 1

        if config.log_interval > 0 and step % config.log_interval == 0:
            print(
                "train_step",
                step,
                "loss",
                f"{loss.item():.6f}",
                "price",
                f"{price_loss.item():.6f}",
                "self",
                f"{self_loss.item():.6f}",
            )

    return {
        "loss": total_loss / max(steps, 1),
        "price_loss": total_price / max(steps, 1),
        "self_loss": total_self / max(steps, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    close_idx: int,
    psycho: PsychoModule | None,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    dir_correct = 0.0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prev_self = None
        if psycho is not None:
            position = PositionInfo(
                initial_balance=config.initial_balance,
                position_size=config.initial_balance * config.exposure_fraction,
                entry_price=1.0,
                leverage=config.leverage,
            )
            prev_self = psycho(inputs, position).as_tensor()
        _, _, price_pred = model(inputs, prev_self, return_price=True)
        pred_close = price_pred[:, close_idx]
        target_close = targets[:, close_idx]
        diff = pred_close - target_close
        mae_sum += diff.abs().sum().item()
        mse_sum += (diff ** 2).sum().item()
        dir_correct += (torch.sign(pred_close) == torch.sign(target_close)).sum().item()
        total += target_close.numel()
    mae = mae_sum / max(total, 1)
    rmse = float(np.sqrt(mse_sum / max(total, 1)))
    direction_acc = dir_correct / max(total, 1)
    return {
        "mae_close": mae,
        "rmse_close": rmse,
        "direction_acc_close": direction_acc,
        "count": float(total),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_experiment(config: ExperimentConfig, label: str) -> dict[str, dict[str, float]]:
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        torch.set_float32_matmul_precision("high")

    loaders, counts, stats = build_loaders(config)
    feature_map = {name: idx for idx, name in enumerate(DEFAULT_FEATURE_COLUMNS)}
    if "close" not in feature_map:
        raise ValueError("Required feature 'close' is missing.")
    close_idx = feature_map["close"]
    print(f"[DATA] train/val/test windows: {counts}")

    if config.weight_self > 0 and config.self_dim != 6:
        raise ValueError("self_dim must be 6 to match self-target definition.")

    model = build_model(config, market_dim=len(DEFAULT_FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    psycho = None
    if config.weight_self > 0:
        psycho = PsychoModule(
            close_idx,
            PsychoConfig(
                cortisol_decay=config.cortisol_decay,
                pain_scale=config.pain_scale,
                inputs_are_log_returns=config.normalize == "log_return",
            ),
        )

    metrics: dict[str, dict[str, float]] = {}
    for epoch in range(config.epochs):
        print(f"[TRAIN] epoch {epoch + 1}/{config.epochs} ({label})")
        train_metrics = train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            config,
            psycho,
        )
        metrics[f"train_epoch_{epoch + 1}"] = train_metrics
        val_metrics = evaluate(model, loaders["val"], device, close_idx, psycho, config)
        metrics[f"val_epoch_{epoch + 1}"] = val_metrics
        print(f"[VAL] {val_metrics}")

    test_metrics = evaluate(model, loaders["test"], device, close_idx, psycho, config)
    metrics["test"] = test_metrics
    print(f"[TEST] {test_metrics}")

    output_dir = Path(config.output_dir) / label
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "norm_stats.json", stats)
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def parse_args() -> tuple[ExperimentConfig, str]:
    parser = argparse.ArgumentParser(description="Timeseries benchmark for UnitaryTransformer.")
    parser.add_argument("--data-path", default=ExperimentConfig.data_path)
    parser.add_argument("--seq-len", type=int, default=ExperimentConfig.seq_len)
    parser.add_argument("--stride", type=int, default=ExperimentConfig.stride)
    parser.add_argument("--train-ratio", type=float, default=ExperimentConfig.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=ExperimentConfig.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=ExperimentConfig.test_ratio)
    parser.add_argument("--normalize", choices=["log_return", "zscore", "none"], default="log_return")
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=ExperimentConfig.num_workers)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument("--weight-price", type=float, default=ExperimentConfig.weight_price)
    parser.add_argument("--weight-self", type=float, default=ExperimentConfig.weight_self)
    parser.add_argument("--self-dim", type=int, default=ExperimentConfig.self_dim)
    parser.add_argument("--d-model", type=int, default=ExperimentConfig.d_model)
    parser.add_argument("--n-heads", type=int, default=ExperimentConfig.n_heads)
    parser.add_argument("--n-layers", type=int, default=ExperimentConfig.n_layers)
    parser.add_argument("--dropout", type=float, default=ExperimentConfig.dropout)
    parser.add_argument("--max-seq-len", type=int, default=ExperimentConfig.max_seq_len)
    parser.add_argument("--use-flash", action="store_true", default=True)
    parser.add_argument("--no-flash", action="store_false", dest="use_flash")
    parser.add_argument("--compile", action="store_true", dest="compile_model", default=False)
    parser.add_argument("--compile-mode", default=ExperimentConfig.compile_mode)
    parser.add_argument("--device", default=ExperimentConfig.device)
    parser.add_argument("--log-interval", type=int, default=ExperimentConfig.log_interval)
    parser.add_argument("--output-dir", default=ExperimentConfig.output_dir)
    parser.add_argument("--initial-balance", type=float, default=ExperimentConfig.initial_balance)
    parser.add_argument("--exposure-fraction", type=float, default=ExperimentConfig.exposure_fraction)
    parser.add_argument("--leverage", type=float, default=ExperimentConfig.leverage)
    parser.add_argument("--cortisol-decay", type=float, default=ExperimentConfig.cortisol_decay)
    parser.add_argument("--pain-scale", type=float, default=ExperimentConfig.pain_scale)
    parser.add_argument(
        "--model",
        choices=["baseline", "unitary", "both"],
        default="unitary",
        help="baseline=price-only, unitary=price+self, both=run both",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        data_path=args.data_path,
        seq_len=args.seq_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        normalize=args.normalize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        weight_price=args.weight_price,
        weight_self=args.weight_self,
        self_dim=args.self_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        use_flash=args.use_flash,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        device=args.device,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        initial_balance=args.initial_balance,
        exposure_fraction=args.exposure_fraction,
        leverage=args.leverage,
        cortisol_decay=args.cortisol_decay,
        pain_scale=args.pain_scale,
    )
    return config, args.model


def main() -> None:
    config, model_choice = parse_args()
    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if model_choice in ("baseline", "both"):
        baseline_config = replace(config, weight_self=0.0)
        run_experiment(baseline_config, label="baseline")

    if model_choice in ("unitary", "both"):
        run_experiment(config, label="unitary")


if __name__ == "__main__":
    main()
