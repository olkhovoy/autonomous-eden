from __future__ import annotations

from pathlib import Path
import json
from typing import Literal, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

DEFAULT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)
NormalizeMethod = Literal["log_return", "zscore", "none"]


def _column_indices(columns: Sequence[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(columns)}


def load_market_json(
    path: str | Path,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> pl.DataFrame:
    path = Path(path)
    schema = {
        columns[0]: pl.Int64,
        columns[1]: pl.Float64,
        columns[2]: pl.Float64,
        columns[3]: pl.Float64,
        columns[4]: pl.Float64,
        columns[5]: pl.Float64,
    }
    try:
        df = pl.read_json(path, schema=schema)
    except Exception:
        try:
            df = pl.read_json(path)
        except Exception:
            df = None
    if df is None:
        try:
            import orjson as json_parser
            with path.open("rb") as handle:
                rows = json_parser.loads(handle.read())
        except Exception:
            import json as json_parser
            with path.open("r") as handle:
                rows = json_parser.load(handle)
        if not rows:
            return pl.DataFrame(schema=list(columns))
        df = pl.DataFrame(rows, schema=list(columns), orient="row")
    if len(df.columns) >= len(columns):
        current = list(df.columns[: len(columns)])
        if current != list(columns) and all(
            name.startswith("column_") for name in current
        ):
            df = df.rename(
                {df.columns[i]: columns[i] for i in range(len(columns))}
            )
    return df.select(list(columns))


def normalize_market_frame(
    df: pl.DataFrame,
    columns: Sequence[str],
    method: NormalizeMethod = "log_return",
    *,
    eps: float = 1e-8,
    stats: dict[str, list[float]] | None = None,
    volume_column: str = "volume",
) -> tuple[pl.DataFrame, dict[str, object]]:
    if method == "none":
        return df, {"method": "none", "columns": tuple(columns)}

    if method == "log_return":
        exprs = []
        for name in columns:
            col = pl.col(name).cast(pl.Float64)
            if name == volume_column:
                col = (col + 1.0).log()
            else:
                col = col.log()
            exprs.append(col.diff().alias(name))
        df = df.with_columns(exprs).drop_nulls()
        return df, {"method": "log_return", "columns": tuple(columns)}

    if method == "zscore":
        if stats is None:
            stats_df = df.select(
                [
                    pl.col(columns).mean().suffix("_mean"),
                    pl.col(columns).std().suffix("_std"),
                ]
            )
            stats_dict = stats_df.to_dict(as_series=False)
            means = [float(stats_dict[f"{c}_mean"][0]) for c in columns]
            stds = [float(stats_dict[f"{c}_std"][0]) for c in columns]
            stats = {"mean": means, "std": stds}
        means = stats["mean"]
        stds = stats["std"]
        exprs = []
        for name, mean, std in zip(columns, means, stds):
            denom = std if std > 0 else eps
            exprs.append(((pl.col(name) - mean) / (denom + eps)).alias(name))
        df = df.with_columns(exprs)
        return df, {
            "method": "zscore",
            "columns": tuple(columns),
            "mean": list(means),
            "std": list(stds),
            "eps": eps,
        }

    raise ValueError(f"Unknown normalization method: {method}")


def compress_ohlcv_array(
    data: np.ndarray,
    compression: int,
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    volume_column: str = "volume",
    drop_partial: bool = True,
) -> np.ndarray:
    if compression <= 0:
        raise ValueError("compression must be >= 1.")
    if compression == 1:
        return np.ascontiguousarray(data)
    if data.ndim != 2:
        raise ValueError("data must be (rows, features).")
    total = data.shape[0]
    if drop_partial:
        usable = (total // compression) * compression
    else:
        usable = total
    if usable < compression:
        raise ValueError("Not enough rows to compress.")
    trimmed = data[:usable]
    groups = trimmed.reshape(-1, compression, data.shape[1])
    out = np.empty((groups.shape[0], data.shape[1]), dtype=groups.dtype)
    indices = _column_indices(feature_columns)
    open_idx = indices.get("open")
    high_idx = indices.get("high")
    low_idx = indices.get("low")
    close_idx = indices.get("close")
    volume_idx = indices.get(volume_column)
    for name, idx in indices.items():
        if idx is None:
            continue
        if name == "open" and open_idx is not None:
            out[:, idx] = groups[:, 0, idx]
        elif name == "high" and high_idx is not None:
            out[:, idx] = groups[:, :, idx].max(axis=1)
        elif name == "low" and low_idx is not None:
            out[:, idx] = groups[:, :, idx].min(axis=1)
        elif name == "close" and close_idx is not None:
            out[:, idx] = groups[:, -1, idx]
        elif name == volume_column and volume_idx is not None:
            out[:, idx] = groups[:, :, idx].sum(axis=1)
        else:
            out[:, idx] = groups[:, -1, idx]
    return np.ascontiguousarray(out)


def normalize_ohlcv_array(
    data: np.ndarray,
    columns: Sequence[str],
    method: NormalizeMethod = "log_return",
    *,
    eps: float = 1e-8,
    stats: dict[str, list[float]] | None = None,
    volume_column: str = "volume",
) -> tuple[np.ndarray, dict[str, object]]:
    if method == "none":
        return np.ascontiguousarray(data), {"method": "none", "columns": tuple(columns)}

    if method == "log_return":
        indices = _column_indices(columns)
        log_data = np.array(data, copy=True, dtype=np.float32)
        for name, idx in indices.items():
            if name == volume_column:
                log_data[:, idx] = np.log(log_data[:, idx] + 1.0)
            else:
                log_data[:, idx] = np.log(np.maximum(log_data[:, idx], eps))
        diff = np.diff(log_data, axis=0)
        return np.ascontiguousarray(diff), {
            "method": "log_return",
            "columns": tuple(columns),
        }

    if method == "zscore":
        if stats is None:
            means = data.mean(axis=0).tolist()
            stds = data.std(axis=0).tolist()
            stats = {"mean": means, "std": stds}
        means = np.array(stats["mean"], dtype=np.float32)
        stds = np.array(stats["std"], dtype=np.float32)
        denom = np.where(stds > 0, stds, eps)
        normalized = (data - means) / (denom + eps)
        return np.ascontiguousarray(normalized), {
            "method": "zscore",
            "columns": tuple(columns),
            "mean": means.tolist(),
            "std": stds.tolist(),
            "eps": eps,
        }

    raise ValueError(f"Unknown normalization method: {method}")


def prepare_memmap(
    path: str | Path,
    *,
    out_dir: str | Path,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    normalize: NormalizeMethod = "log_return",
    dtype: np.dtype = np.float32,
    return_raw: bool = True,
    compressions: Sequence[int] | None = None,
) -> dict[str, object]:
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = path.stem
    df = load_market_json(path)
    if df.is_empty():
        raise ValueError(f"Market data is empty: {path}")
    df = df.select(list(feature_columns))
    raw_full = df.select(list(feature_columns)).to_numpy()
    raw_full = np.ascontiguousarray(raw_full, dtype=dtype)
    norm_arr, stats = normalize_ohlcv_array(
        raw_full, feature_columns, method=normalize
    )
    if normalize == "log_return":
        raw_arr = raw_full[-norm_arr.shape[0] :]
    else:
        raw_arr = raw_full
    norm_path = out_dir / f"{base_name}_norm.npy"
    np.save(norm_path, norm_arr)
    raw_path = None
    if return_raw:
        raw_path = out_dir / f"{base_name}_raw.npy"
        np.save(raw_path, raw_arr)
    meta: dict[str, object] = {
        "base": {
            "norm_path": str(norm_path),
            "raw_path": str(raw_path) if raw_path is not None else None,
            "shape": list(norm_arr.shape),
        },
        "feature_columns": list(feature_columns),
        "normalize": normalize,
        "stats": stats,
    }
    compressed_meta: dict[str, object] = {}
    if compressions:
        for comp in compressions:
            comp_raw_full = compress_ohlcv_array(
                raw_full,
                comp,
                feature_columns=feature_columns,
            )
            comp_norm, comp_stats = normalize_ohlcv_array(
                comp_raw_full, feature_columns, method=normalize
            )
            if normalize == "log_return":
                comp_raw = comp_raw_full[-comp_norm.shape[0] :]
            else:
                comp_raw = comp_raw_full
            comp_norm_path = out_dir / f"{base_name}_norm_c{comp}.npy"
            np.save(comp_norm_path, comp_norm)
            comp_raw_path = None
            if return_raw:
                comp_raw_path = out_dir / f"{base_name}_raw_c{comp}.npy"
                np.save(comp_raw_path, comp_raw)
            compressed_meta[str(comp)] = {
                "norm_path": str(comp_norm_path),
                "raw_path": str(comp_raw_path) if comp_raw_path is not None else None,
                "shape": list(comp_norm.shape),
                "stats": comp_stats,
            }
    if compressed_meta:
        meta["compressed"] = compressed_meta
    meta_path = out_dir / f"{base_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


class MarketDataset(Dataset):
    """Sliding window dataset over normalized market OHLCV data."""

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        stride: int = 1,
        normalize: NormalizeMethod = "log_return",
        feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
        dtype: np.dtype = np.float32,
        norm_stats: dict[str, list[float]] | None = None,
        return_raw: bool = False,
    ) -> None:
        df = load_market_json(path)
        df = df.select(list(feature_columns))
        raw_df = df
        df, stats = normalize_market_frame(
            df,
            feature_columns,
            method=normalize,
            stats=norm_stats,
        )
        df = df.with_columns(pl.col(feature_columns).cast(pl.Float32))
        arr = df.select(list(feature_columns)).to_numpy()
        arr = np.ascontiguousarray(arr, dtype=dtype)
        self.data = torch.from_numpy(arr)
        self.raw_data: torch.Tensor | None = None
        if return_raw:
            if raw_df.height != df.height:
                raw_df = raw_df.tail(df.height)
            raw_df = raw_df.with_columns(pl.col(feature_columns).cast(pl.Float32))
            raw_arr = raw_df.select(list(feature_columns)).to_numpy()
            raw_arr = np.ascontiguousarray(raw_arr, dtype=dtype)
            self.raw_data = torch.from_numpy(raw_arr)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.features = tuple(feature_columns)
        self.norm_stats = stats
        self.return_raw = return_raw
        total = self.data.shape[0]
        max_start = total - self.seq_len
        if max_start < 0:
            raise ValueError(
                "Sequence length is larger than available rows: "
                f"seq_len={self.seq_len} rows={total}"
            )
        self.length = max_start // self.stride + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self, idx: int
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        window = self.data.narrow(0, start, self.seq_len)
        if self.raw_data is None:
            return window
        raw_window = self.raw_data.narrow(0, start, self.seq_len)
        return window, raw_window


def load_memmap_manifest(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def open_memmap(path: str | Path) -> np.ndarray:
    return np.load(Path(path), mmap_mode="r")


class MemmapMarketDataset(Dataset):
    """Map-style dataset backed by np.memmap arrays."""

    def __init__(
        self,
        norm_path: str | Path,
        *,
        seq_len: int,
        stride: int = 1,
        raw_path: str | Path | None = None,
    ) -> None:
        self.data = open_memmap(norm_path)
        self.raw_data = open_memmap(raw_path) if raw_path is not None else None
        if self.data.ndim != 2:
            raise ValueError("Memmap data must be (rows, features).")
        if self.raw_data is not None and self.raw_data.shape != self.data.shape:
            raise ValueError("raw memmap shape must match normalized data.")
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        total = self.data.shape[0]
        max_start = total - self.seq_len
        if max_start < 0:
            raise ValueError(
                "Sequence length is larger than available rows: "
                f"seq_len={self.seq_len} rows={total}"
            )
        self.length = max_start // self.stride + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self, idx: int
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        window = torch.from_numpy(self.data[start : start + self.seq_len])
        if self.raw_data is None:
            return window
        raw_window = torch.from_numpy(self.raw_data[start : start + self.seq_len])
        return window, raw_window


class IterableMarketDataset(IterableDataset):
    """Streaming dataset backed by np.memmap arrays."""

    def __init__(
        self,
        norm_path: str | Path,
        *,
        seq_len: int,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        raw_path: str | Path | None = None,
        shuffle_buffer: int | None = None,
        seed: int = 0,
    ) -> None:
        self.data = open_memmap(norm_path)
        self.raw_data = open_memmap(raw_path) if raw_path is not None else None
        if self.data.ndim != 2:
            raise ValueError("Memmap data must be (rows, features).")
        if self.raw_data is not None and self.raw_data.shape != self.data.shape:
            raise ValueError("raw memmap shape must match normalized data.")
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.start = int(start)
        self.end = int(end) if end is not None else self.data.shape[0]
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        max_start = self.end - self.seq_len
        if max_start < self.start:
            raise ValueError("Not enough rows for requested window.")
        self.max_start = max_start

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        total_starts = self.max_start - self.start + 1
        per_worker = (total_starts + num_workers - 1) // num_workers
        worker_start = self.start + worker_id * per_worker
        worker_end = min(worker_start + per_worker, self.max_start + 1)
        if worker_start >= worker_end:
            return iter(())

        rng = np.random.default_rng(self.seed + worker_id)
        buffer: list[int] = []

        for idx in range(worker_start, worker_end, self.stride):
            if self.shuffle_buffer:
                buffer.append(idx)
                if len(buffer) >= self.shuffle_buffer:
                    pick = int(rng.integers(len(buffer)))
                    start = buffer.pop(pick)
                else:
                    continue
            else:
                start = idx
            window = torch.from_numpy(self.data[start : start + self.seq_len])
            if self.raw_data is None:
                yield window
            else:
                raw_window = torch.from_numpy(self.raw_data[start : start + self.seq_len])
                yield window, raw_window

        if buffer:
            rng.shuffle(buffer)
            for start in buffer:
                window = torch.from_numpy(self.data[start : start + self.seq_len])
                if self.raw_data is None:
                    yield window
                else:
                    raw_window = torch.from_numpy(self.raw_data[start : start + self.seq_len])
                    yield window, raw_window
