import os
import numpy as np
import torch
from pathlib import Path
import time
from tqdm import tqdm
import polars as pl
import glob

import sys
sys.path.append(str(Path(__file__).parent.parent))

from umc_nn.neurobars.models import NeurobarEncoder

# --- CONFIGURATION ---
DATA_DIR = Path("data/BTCUSDT/data")
OUTPUT_DIR = Path("data")
SEQ_LEN = 128
LATENT_DIM = 32
BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
ENCODER_PATH = CHECKPOINT_DIR / "neurobar_encoder_best.pt"
NORM_STATS_PATH = CHECKPOINT_DIR / "neurobar_norm_stats.npy"

def load_and_prepare_data():
    print("Loading historical parquet data for conversion...")
    files = glob.glob(str(DATA_DIR / "**/*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")
        
    dfs = []
    for file in files:
        df = pl.read_parquet(file)
        # Cast all numeric columns to Float32
        cast_dict = {}
        for col, dtype in df.schema.items():
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32]:
                cast_dict[col] = pl.Float32
        df = df.cast(cast_dict)
        dfs.append(df)
        
    df = pl.concat(dfs, how="diagonal_relaxed").sort("timestamp")
    
    # Save the close prices and timestamps for the environment BEFORE dropping columns
    # Ensure it's ffilled
    close_prices = df.get_column("close").fill_null(strategy="forward").fill_null(0.0).to_numpy().astype(np.float32)
    # Convert timestamps to numeric (UNIX timestamp in seconds) for easy storage in npz
    timestamps = df.get_column("timestamp").dt.epoch("s").to_numpy().astype(np.int64)
    
    cols_to_drop = [
        "timestamp", 
        "cg_fear_greed_classification", 
        "ciss_quality", 
        "bias_components", 
        "dst_regime", 
        "ciss_valid"
    ]
    df = df.drop([c for c in cols_to_drop if c in df.columns])
    df = df.fill_null(strategy="forward")
    
    features = df.to_numpy().astype(np.float32)
    return features, close_prices, timestamps, len(df.columns)

def main():
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Encoder weights not found at {ENCODER_PATH}. Run train_neurobars.py first.")
    if not NORM_STATS_PATH.exists():
        raise FileNotFoundError(f"Normalization stats not found at {NORM_STATS_PATH}. Run train_neurobars.py first.")

    features, close_prices, timestamps, input_dim = load_and_prepare_data()

    print(f"Loading encoder from {ENCODER_PATH} to {DEVICE}...")
    encoder = NeurobarEncoder(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    encoder.eval()
    
    print(f"Loading normalization stats from {NORM_STATS_PATH}...")
    norm_stats = np.load(NORM_STATS_PATH, allow_pickle=True).item()
    means = norm_stats['means']
    stds = norm_stats['stds']
    
    print("Normalizing features...")
    features = (features - means) / stds
    # Replace remaining NaNs with 0.0 (which is the neutral mean in Z-score space)
    features = np.nan_to_num(features, nan=0.0)
    
    num_bars = len(features)
    num_neurobars = num_bars - SEQ_LEN + 1
    
    print(f"Generating {num_neurobars} neurobars...")
    neurobars = np.zeros((num_neurobars, LATENT_DIM), dtype=np.float32)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_neurobars, BATCH_SIZE), desc="Encoding"):
            end_idx = min(i + BATCH_SIZE, num_neurobars)
            
            # Build batch of sequences
            batch_seqs = []
            for j in range(i, end_idx):
                batch_seqs.append(features[j : j + SEQ_LEN])
                
            batch_tensor = torch.FloatTensor(np.array(batch_seqs)).to(DEVICE)
            
            # Encode
            latent = encoder(batch_tensor)
            neurobars[i:end_idx] = latent.cpu().numpy()

    # The neurobar at index `i` corresponds to the market state at `features[i + SEQ_LEN - 1]`
    aligned_close_prices = close_prices[SEQ_LEN - 1:]
    aligned_timestamps = timestamps[SEQ_LEN - 1:]
    
    out_path = OUTPUT_DIR / "BTCUSDT_parquet_neurobars.npz"
    np.savez_compressed(
        out_path,
        close_prices=aligned_close_prices,
        timestamps=aligned_timestamps,
        neurobars=neurobars
    )
    print(f"Saved {num_neurobars} neurobars to {out_path}")

if __name__ == "__main__":
    main()
