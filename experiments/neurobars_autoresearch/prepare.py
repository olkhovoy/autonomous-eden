from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import glob
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = ROOT / "data" / "BTCUSDT" / "data"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mcs_neurobars_autoresearch"

DATA_DIR = Path(os.environ.get("NEUROBARS_AUTORESEARCH_DATA_DIR", str(DEFAULT_DATA_DIR)))
CACHE_DIR = Path(os.environ.get("NEUROBARS_AUTORESEARCH_CACHE_DIR", str(DEFAULT_CACHE_DIR)))

CACHE_VERSION = "v1"
SEQ_LEN = 128
TIME_BUDGET = int(os.environ.get("NEUROBARS_AUTORESEARCH_TIME_BUDGET", "300"))
TRAIN_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 1024
EVAL_SEQUENCES = 131072
VAL_START_UTC = os.environ.get("NEUROBARS_AUTORESEARCH_VAL_START_UTC", "2025-04-24 12:20:00")

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


@dataclass
class PreparedMetadata:
    cache_version: str
    seq_len: int
    input_dim: int
    train_rows: int
    val_rows: int
    train_sequences: int
    val_sequences: int
    columns: list[str]
    close_idx: int
    feature_weights: list[float]
    val_start_utc: str
    val_start_timestamp: int


class NeurobarSequenceDataset(Dataset):
    def __init__(self, features_path: Path, seq_len: int):
        self.features = np.load(features_path, mmap_mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.array(self.features[idx : idx + self.seq_len], dtype=np.float32, copy=True)
        y = np.array(self.features[idx + self.seq_len], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def parse_utc_timestamp(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def build_feature_weights(columns: Iterable[str]) -> np.ndarray:
    columns = list(columns)
    weights = np.ones(len(columns), dtype=np.float32)
    for idx, column in enumerate(columns):
        name = column.lower()
        if "close" in name:
            weights[idx] = max(weights[idx], 6.0)
        elif any(token in name for token in ("open", "high", "low")):
            weights[idx] = max(weights[idx], 4.0)
        elif "volume" in name or "taker_buy" in name:
            weights[idx] = max(weights[idx], 2.0)
    return weights


def load_market_dataframe(data_dir: Path) -> pl.DataFrame:
    files = glob.glob(str(data_dir / "**/*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = []
    for file in files:
        df = pl.read_parquet(file)
        cast_dict = {}
        for column, dtype in df.schema.items():
            if column == "timestamp":
                continue
            if dtype in NUMERIC_DTYPES:
                cast_dict[column] = pl.Float32
        if cast_dict:
            df = df.cast(cast_dict)
        dfs.append(df)

    df = pl.concat(dfs, how="diagonal_relaxed").sort("timestamp")
    numeric_cols = [column for column, dtype in df.schema.items() if column != "timestamp" and dtype in NUMERIC_DTYPES]
    if "close" not in numeric_cols:
        raise ValueError("Expected a numeric 'close' column in market data")

    df = df.select(["timestamp", *numeric_cols])
    df = df.with_columns(pl.col("timestamp").dt.epoch("s").alias("timestamp_s"))

    feature_cols = [column for column in df.columns if column not in {"timestamp", "timestamp_s"}]
    df = df.with_columns([pl.col(feature_cols).fill_null(strategy="forward")])
    df = df.with_columns([pl.col(feature_cols).fill_null(0.0)])
    return df


def split_dataframe(df: pl.DataFrame, val_start_utc: str, seq_len: int) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    val_start_timestamp = parse_utc_timestamp(val_start_utc)
    train_df = df.filter(pl.col("timestamp_s") < val_start_timestamp)
    val_df = df.filter(pl.col("timestamp_s") >= val_start_timestamp)
    if len(train_df) <= seq_len:
        raise ValueError("Training split is too small for the configured seq_len")
    if len(val_df) <= seq_len:
        raise ValueError("Validation split is too small for the configured seq_len")
    return train_df, val_df, val_start_timestamp


def _normalize(features: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    normalized = (features - means) / stds
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def prepare_cache(
    *,
    data_dir: Path = DATA_DIR,
    cache_dir: Path = CACHE_DIR,
    val_start_utc: str = VAL_START_UTC,
    seq_len: int = SEQ_LEN,
    force: bool = False,
) -> PreparedMetadata:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    train_features_path = cache_dir / "train_features.npy"
    val_features_path = cache_dir / "val_features.npy"
    means_path = cache_dir / "means.npy"
    stds_path = cache_dir / "stds.npy"
    weights_path = cache_dir / "feature_weights.npy"

    if (
        not force
        and metadata_path.exists()
        and train_features_path.exists()
        and val_features_path.exists()
        and means_path.exists()
        and stds_path.exists()
        and weights_path.exists()
    ):
        return load_metadata(cache_dir)

    df = load_market_dataframe(data_dir)
    train_df, val_df, val_start_timestamp = split_dataframe(df, val_start_utc, seq_len)

    feature_cols = [column for column in train_df.columns if column not in {"timestamp", "timestamp_s"}]
    train_features = train_df.select(feature_cols).to_numpy().astype(np.float32)
    val_features = val_df.select(feature_cols).to_numpy().astype(np.float32)

    means = np.nanmean(train_features, axis=0)
    stds = np.nanstd(train_features, axis=0)
    means = np.nan_to_num(means, nan=0.0)
    stds = np.nan_to_num(stds, nan=1.0)
    stds[stds == 0] = 1e-8

    train_norm = _normalize(train_features, means, stds)
    val_norm = _normalize(val_features, means, stds)

    weights = build_feature_weights(feature_cols)
    close_idx = feature_cols.index("close")

    np.save(train_features_path, train_norm)
    np.save(val_features_path, val_norm)
    np.save(means_path, means.astype(np.float32))
    np.save(stds_path, stds.astype(np.float32))
    np.save(weights_path, weights.astype(np.float32))

    metadata = PreparedMetadata(
        cache_version=CACHE_VERSION,
        seq_len=seq_len,
        input_dim=len(feature_cols),
        train_rows=int(len(train_norm)),
        val_rows=int(len(val_norm)),
        train_sequences=max(0, int(len(train_norm) - seq_len)),
        val_sequences=max(0, int(len(val_norm) - seq_len)),
        columns=feature_cols,
        close_idx=close_idx,
        feature_weights=weights.tolist(),
        val_start_utc=val_start_utc,
        val_start_timestamp=val_start_timestamp,
    )
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2))
    return metadata


def load_metadata(cache_dir: Path = CACHE_DIR) -> PreparedMetadata:
    data = json.loads((cache_dir / "metadata.json").read_text())
    return PreparedMetadata(**data)


def load_feature_weights(cache_dir: Path = CACHE_DIR, device: torch.device | None = None) -> torch.Tensor:
    weights = np.load(cache_dir / "feature_weights.npy").astype(np.float32)
    tensor = torch.from_numpy(weights)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def make_dataloader(
    split: str,
    *,
    batch_size: int,
    cache_dir: Path = CACHE_DIR,
    seq_len: int = SEQ_LEN,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")
    if shuffle is None:
        shuffle = split == "train"
    dataset = NeurobarSequenceDataset(cache_dir / f"{split}_features.npy", seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )


def weighted_next_bar_mse(pred: torch.Tensor, target: torch.Tensor, feature_weights: torch.Tensor) -> torch.Tensor:
    weights = feature_weights.view(1, -1)
    return (((pred - target) ** 2) * weights).mean()


def close_delta_mse(
    inputs: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    close_idx: int,
) -> torch.Tensor:
    last_close = inputs[:, -1, close_idx]
    pred_delta = pred[:, close_idx] - last_close
    target_delta = target[:, close_idx] - last_close
    return torch.mean((pred_delta - target_delta) ** 2)


def close_direction_accuracy(
    inputs: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    close_idx: int,
) -> torch.Tensor:
    last_close = inputs[:, -1, close_idx]
    pred_delta = pred[:, close_idx] - last_close
    target_delta = target[:, close_idx] - last_close
    return (torch.sign(pred_delta) == torch.sign(target_delta)).float().mean()


@dataclass
class EvalMetrics:
    val_score: float
    val_weighted_mse: float
    val_close_delta_mse: float
    val_direction_acc: float
    val_sequences_evaluated: int


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int = EVAL_BATCH_SIZE,
    cache_dir: Path = CACHE_DIR,
) -> EvalMetrics:
    metadata = load_metadata(cache_dir)
    loader = make_dataloader("val", batch_size=batch_size, cache_dir=cache_dir, shuffle=False)
    feature_weights = load_feature_weights(cache_dir, device=device)

    total_weighted_mse = 0.0
    total_close_delta_mse = 0.0
    total_direction_acc = 0.0
    total_examples = 0
    remaining = EVAL_SEQUENCES

    model.eval()
    for x, y in loader:
        if remaining <= 0:
            break
        x = x.to(device)
        y = y.to(device)
        current_batch = min(x.size(0), remaining)
        if current_batch != x.size(0):
            x = x[:current_batch]
            y = y[:current_batch]

        _, pred = model(x)
        weighted_mse = weighted_next_bar_mse(pred, y, feature_weights)
        delta_mse = close_delta_mse(x, pred, y, close_idx=metadata.close_idx)
        direction_acc = close_direction_accuracy(x, pred, y, close_idx=metadata.close_idx)

        total_weighted_mse += float(weighted_mse.item()) * current_batch
        total_close_delta_mse += float(delta_mse.item()) * current_batch
        total_direction_acc += float(direction_acc.item()) * current_batch
        total_examples += current_batch
        remaining -= current_batch

    if total_examples == 0:
        raise RuntimeError("Validation loader produced zero examples")

    val_weighted_mse = total_weighted_mse / total_examples
    val_close_delta_mse = total_close_delta_mse / total_examples
    val_direction_acc = total_direction_acc / total_examples
    val_score = val_weighted_mse + 0.5 * val_close_delta_mse + 0.05 * (1.0 - val_direction_acc)

    return EvalMetrics(
        val_score=val_score,
        val_weighted_mse=val_weighted_mse,
        val_close_delta_mse=val_close_delta_mse,
        val_direction_acc=val_direction_acc,
        val_sequences_evaluated=total_examples,
    )


def main() -> None:
    metadata = prepare_cache()
    print("Prepared neurobar autoresearch cache")
    print(f"cache_dir:            {CACHE_DIR}")
    print(f"cache_version:        {metadata.cache_version}")
    print(f"input_dim:            {metadata.input_dim}")
    print(f"train_rows:           {metadata.train_rows}")
    print(f"val_rows:             {metadata.val_rows}")
    print(f"train_sequences:      {metadata.train_sequences}")
    print(f"val_sequences:        {metadata.val_sequences}")
    print(f"val_start_utc:        {metadata.val_start_utc}")


if __name__ == "__main__":
    main()
