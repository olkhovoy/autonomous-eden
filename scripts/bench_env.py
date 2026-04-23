import time
import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "umc_nn"))
from umc_cell import UMCTradingCell
from pure_env import PureTradingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_file = os.path.join(os.path.dirname(__file__), "..", "data", "BTCUSDT_parquet_neurobars.npz")
env = PureTradingEnv(max_steps=1052640, initial_balance=10000.0, data_path=data_file, use_neurobars=True)
env.start_step = 1982117

cell = UMCTradingCell(input_dim=env.input_dim, hidden_dim=64).to(device)

start = time.time()
obs = env.reset()
h_prev = torch.zeros(1, 64, device=device)

with torch.no_grad():
    done = False
    while not done:
        x_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        action_logits, h_next = cell(x_tensor, h_prev)
        action = torch.argmax(action_logits, dim=1).item()
        obs, reward, done, info = env.step(action)
        h_prev = h_next
        
        # Avoid margin call for benchmark
        if env.balance <= 0.5 * env.initial_balance:
            env.balance = env.initial_balance

print(f"Time for 1M steps: {time.time() - start:.2f} seconds")
