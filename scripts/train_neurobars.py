import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import polars as pl
import glob

# Add parent dir to path to import umc_nn
import sys
sys.path.append(str(Path(__file__).parent.parent))

from umc_nn.neurobars.models import NeurobarEncoder, NeurobarAutoencoder

# --- CONFIGURATION ---
DATA_DIR = Path("data/BTCUSDT/data")
SEQ_LEN = 128          # Look back 128 minutes
BATCH_SIZE = 1024
EPOCHS = 15
LEARNING_RATE = 5e-4
LATENT_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

class OHLCVSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, seq_len: int):
        self.features = torch.FloatTensor(features)
        self.seq_len = seq_len
        
    def __len__(self):
        # We need seq_len for input, and +1 for the target (next bar)
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.features[idx + self.seq_len] # Predict the next bar
        return x, y

def load_and_normalize_data() -> tuple[np.ndarray, int]:
    print("Loading historical parquet data...")
    
    files = glob.glob(str(DATA_DIR / "**/*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")
        
    print(f"Found {len(files)} parquet files. Reading and concatenating...")
    
    # Read files individually to handle schema mismatches (e.g., Int8 vs Float64)
    dfs = []
    for file in files:
        df = pl.read_parquet(file)
        # Cast all numeric columns to Float32 to avoid SchemaError
        cast_dict = {}
        for col, dtype in df.schema.items():
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32]:
                cast_dict[col] = pl.Float32
        
        df = df.cast(cast_dict)
        dfs.append(df)
        
    # Concatenate all dataframes
    df = pl.concat(dfs, how="diagonal_relaxed")
    
    # Sort by timestamp
    df = df.sort("timestamp")
    
    # Drop string and boolean columns
    cols_to_drop = [
        "timestamp", 
        "cg_fear_greed_classification", 
        "ciss_quality", 
        "bias_components", 
        "dst_regime", 
        "ciss_valid"
    ]
    df = df.drop([c for c in cols_to_drop if c in df.columns])
    
    # Forward fill NaNs
    df = df.fill_null(strategy="forward")
    
    input_dim = len(df.columns)
    print(f"Total rows loaded: {len(df)}")
    print(f"Total features: {input_dim}")
    
    # Convert to numpy float32
    combined_features = df.to_numpy().astype(np.float32)
    
    print("Applying Z-score normalization...")
    
    # Compute mean and std ignoring NaNs (on ALL data)
    with np.errstate(all='ignore'):
        means = np.nanmean(combined_features, axis=0)
        stds = np.nanstd(combined_features, axis=0)
    
    # Handle fully NaN columns
    means = np.nan_to_num(means, nan=0.0)
    stds = np.nan_to_num(stds, nan=1.0)
    stds[stds == 0] = 1e-8
    
    # Normalize all features
    normalized_features = (combined_features - means) / stds
    
    # Replace remaining NaNs with 0.0 (which is the neutral mean in Z-score space)
    normalized_features = np.nan_to_num(normalized_features, nan=0.0)
    
    # Save normalization stats for later use in the environment
    np.save(CHECKPOINT_DIR / "neurobar_norm_stats.npy", {'means': means, 'stds': stds, 'columns': df.columns})
    
    return normalized_features, input_dim

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    data, input_dim = load_and_normalize_data()
    
    # Split train/val (90/10) from the full data
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_dataset = OHLCVSequenceDataset(train_data, SEQ_LEN)
    val_dataset = OHLCVSequenceDataset(val_data, SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Initialize Model
    encoder = NeurobarEncoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    model = NeurobarAutoencoder(encoder, input_dim=input_dim).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    print("\nStarting unsupervised training of Neurobar Encoder (Next-Step Prediction)...")
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            _, y_pred = model(x)
            
            loss = criterion(y_pred, y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}", end='\r')
                
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                _, y_pred = model(x)
                val_loss += criterion(y_pred, y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{EPOCHS} completed in {epoch_time:.1f}s | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = CHECKPOINT_DIR / "neurobar_encoder_best.pt"
            torch.save(model.encoder.state_dict(), save_path)
            print(f"  [*] Best model saved to {save_path}")

    print("\nTraining complete. Encoder is ready to generate Neurobars.")

if __name__ == "__main__":
    train()
