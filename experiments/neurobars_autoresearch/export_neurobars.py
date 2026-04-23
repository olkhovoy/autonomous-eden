#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.neurobars_autoresearch.prepare import CACHE_DIR, SEQ_LEN, load_market_dataframe, load_metadata
from experiments.neurobars_autoresearch.train import NeurobarPredictor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export neurobars from the latest autoresearch checkpoint.")
    parser.add_argument("--checkpoint-path", default="checkpoints/neurobars_autoresearch_latest.pt")
    parser.add_argument("--data-dir", default="data/BTCUSDT/data")
    parser.add_argument("--output-path", default="data/BTCUSDT_parquet_neurobars_autoresearch.npz")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--representation",
        choices=["fused", "base_multiscale"],
        default="fused",
        help="Export either the fused 32-d latent or the full base+fast+mid+slow multi-scale state.",
    )
    return parser


def _representation_dim(model: NeurobarPredictor, representation: str) -> int:
    if representation == "fused":
        return model.encoder.base_encoder.latent_proj.out_features
    if representation == "base_multiscale":
        return model.encoder.base_multiscale_dim
    raise ValueError(f"Unsupported representation: {representation}")


def _extract_representation(model: NeurobarPredictor, batch_tensor: torch.Tensor, representation: str) -> np.ndarray:
    components = model.encoder.encode_components(batch_tensor)
    if representation == "fused":
        return components["fused"].cpu().numpy()
    if representation == "base_multiscale":
        combined = torch.cat(
            [components["base"], components["fast"], components["mid"], components["slow"]],
            dim=1,
        )
        return combined.cpu().numpy()
    raise ValueError(f"Unsupported representation: {representation}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    metadata = load_metadata(cache_dir)

    means = np.load(cache_dir / "means.npy").astype(np.float32)
    stds = np.load(cache_dir / "stds.npy").astype(np.float32)
    stds[stds == 0] = 1e-8

    df = load_market_dataframe(Path(args.data_dir))
    feature_df = df.select(metadata.columns)
    features = feature_df.to_numpy().astype(np.float32)
    timestamps = df.get_column("timestamp_s").to_numpy().astype(np.int64)
    close_prices = df.get_column("close").to_numpy().astype(np.float32)

    normalized = (features - means) / stds
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NeurobarPredictor(input_dim=metadata.input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    num_neurobars = len(normalized) - SEQ_LEN + 1
    representation_dim = _representation_dim(model, args.representation)
    neurobars = np.zeros((num_neurobars, representation_dim), dtype=np.float32)

    with torch.no_grad():
        for start_idx in tqdm(range(0, num_neurobars, args.batch_size), desc="Encoding"):
            end_idx = min(start_idx + args.batch_size, num_neurobars)
            batch = np.array(
                [normalized[idx : idx + SEQ_LEN] for idx in range(start_idx, end_idx)],
                dtype=np.float32,
            )
            batch_tensor = torch.from_numpy(batch).to(device)
            neurobars[start_idx:end_idx] = _extract_representation(model, batch_tensor, args.representation)

    aligned_close_prices = close_prices[SEQ_LEN - 1 :]
    aligned_timestamps = timestamps[SEQ_LEN - 1 :]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        close_prices=aligned_close_prices,
        timestamps=aligned_timestamps,
        neurobars=neurobars,
        representation=np.array(args.representation),
    )
    print(
        f"Saved {num_neurobars} neurobars with representation={args.representation} "
        f"dim={representation_dim} to {output_path}"
    )


if __name__ == "__main__":
    main()
