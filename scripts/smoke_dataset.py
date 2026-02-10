from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MarketDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test MarketDataset.")
    parser.add_argument(
        "--data-path",
        default="data/BTCUSDT_2023-05-31_01-40_to_2025-04-24_12-19.json",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = MarketDataset(args.data_path, seq_len=args.seq_len, normalize="log_return")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    batch = batch.to(device)
    print("batch", tuple(batch.shape), "device", batch.device)


if __name__ == "__main__":
    main()
