#!/usr/bin/env python3
import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap

# --- Scaffolds for Policy & Agent ---

import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    def forward(self, x):
        return self.fc(x)

class PatchTST_Self:
    def __init__(self, policy_state_dict=None):
        self.frozen = True
        self.policy = PolicyNet()
        if policy_state_dict is not None and "dummy" not in policy_state_dict:
            self.policy.load_state_dict(policy_state_dict)

class JanusAgent:
    def __init__(self, limbic_system, neuro_params):
        self.limbic_system = limbic_system
        self.stress_penalty = neuro_params[0]
        self.pain_scale = neuro_params[1]
        self.cortisol_decay = neuro_params[2]
        self.time_decay = neuro_params[3]
        self.entropy_coef = neuro_params[4]
        
    def act(self, obs):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits = self.limbic_system.policy(obs_tensor)
            # Deterministic: argmax
            actions = torch.argmax(logits, dim=1).numpy()
        return actions

# --- Simulating a visually interesting Environment ---
class VisualMarketEnv:
    def __init__(self, max_steps=2000):
        self.max_steps = max_steps
        self.current_step = 0
        self.price = 100.0
        self.position = 0 # 0=Flat, 1=Long, -1=Short
        
    def reset(self):
        self.current_step = 0
        self.price = 100.0
        self.position = 0
        return np.random.randn(1, 10)
        
    def step(self, action):
        self.current_step += 1
        
        # Simulate price movement (random walk with some momentum)
        change = np.random.randn() * 0.5
        self.price += change
        
        prev_position = self.position
        
        # Determine position size based on action
        # 0: Flat, 1: Long, 2: Short
        if action[0] == 0:
            self.position = 0
        elif action[0] == 1:
            self.position = 1
        elif action[0] == 2:
            self.position = -1
            
        # PnL is based on price change and current position
        step_pnl = change * prev_position
        
        # --- Friction: Transaction Costs ---
        FEE_RATE = 0.0006
        delta_position = abs(self.position - prev_position)
        transaction_cost = delta_position * self.price * FEE_RATE
        
        # Apply costs
        step_pnl -= transaction_cost
        
        obs = np.random.randn(1, 10)
        done = self.current_step >= self.max_steps
        
        info = {
            "price": self.price,
            "position": self.position,
            "step_pnl": step_pnl,
            "transaction_cost": transaction_cost
        }
        return obs, step_pnl, done, info

def colorline(x, y, z=None, cmap=plt.get_cmap('RdYlGn'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form numlines x (points per line) x 2 (x and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def main():
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    
    # 1. Find the latest checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "best_policy_gen_*.pt"))
    if not checkpoint_files:
        print("Error: No checkpoints found in checkpoints/ directory.")
        return
        
    # Extract generation numbers and find max
    latest_cp = max(checkpoint_files, key=lambda f: int(f.split("_gen_")[1].split(".")[0]))
    print(f"Loading policy from: {latest_cp}")
    
    state_dict = torch.load(latest_cp, weights_only=True)
    
    # Load neuro parameters from CSV (last row)
    csv_path = os.path.join(logs_dir, "evolution_history.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        last_row = df.iloc[-1]
        neuro_params = np.array([
            last_row["stress_penalty"],
            last_row["pain_scale"],
            last_row["cortisol_decay"],
            last_row["time_decay"],
            last_row["entropy_coef"]
        ])
        print(f"Loaded neurochemistry from gen {last_row['generation']}: {neuro_params}")
    else:
        print("Warning: CSV not found, using dummy neuro parameters")
        neuro_params = np.array([0.1, 1.0, 0.5, 0.01, 0.1])
        
    # Setup Agent and Env
    limbic = PatchTST_Self(state_dict)
    agent = JanusAgent(limbic, neuro_params)
    env = VisualMarketEnv(max_steps=2000)
    
    obs = env.reset()
    
    # Data recording
    history = {
        "step": [],
        "price": [],
        "action": [],
        "position": [],
        "pnl": [],
        "cum_pnl": []
    }
    
    cum_pnl = 0.0
    print("Running deterministic evaluation for 2000 steps...")
    
    while True:
        action = agent.act(obs)
        obs, step_pnl, done, info = env.step(action)
        
        cum_pnl += step_pnl
        history["step"].append(env.current_step)
        history["price"].append(info["price"])
        history["action"].append(action[0])
        history["position"].append(info["position"])
        history["pnl"].append(step_pnl)
        history["cum_pnl"].append(cum_pnl)
        
        if done:
            break
            
    df_hist = pd.DataFrame(history)
    
    # --- Plotting ---
    print("Generating Visualization...")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle("Janus Agent: Visual Audit (Deterministic Evaluation)", fontsize=16)
    
    # --- Panel A: Price Action & Trades ---
    ax1.plot(df_hist["step"], df_hist["price"], color="black", linewidth=1.5, label="Close Price")
    
    # Shade backgrounds based on position
    long_mask = df_hist["position"] > 0
    short_mask = df_hist["position"] < 0
    
    ax1.fill_between(df_hist["step"], ax1.get_ylim()[0], ax1.get_ylim()[1], where=long_mask, color='green', alpha=0.15, label="Long Exposure")
    ax1.fill_between(df_hist["step"], ax1.get_ylim()[0], ax1.get_ylim()[1], where=short_mask, color='red', alpha=0.15, label="Short Exposure")
    
    # Trade Markers (when action changes)
    df_hist['prev_action'] = df_hist['action'].shift(1).fillna(-1)
    trade_points = df_hist[df_hist['action'] != df_hist['prev_action']]
    
    long_entries = trade_points[trade_points['action'] == 1]
    short_entries = trade_points[trade_points['action'] == 2]
    flat_entries = trade_points[trade_points['action'] == 0]
    
    ax1.scatter(long_entries["step"], long_entries["price"], marker='^', color='green', s=100, zorder=5, label="LONG Entry")
    ax1.scatter(short_entries["step"], short_entries["price"], marker='v', color='red', s=100, zorder=5, label="SHORT Entry")
    ax1.scatter(flat_entries["step"], flat_entries["price"], marker='x', color='gray', s=100, zorder=5, label="FLAT")
    
    ax1.set_ylabel("Price")
    ax1.set_title("Panel A: The Arena (Price Action & Trades)")
    # deduplicate legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper left")
    
    # --- Panel B: The Lifeblood (Equity Curve) ---
    steps = df_hist["step"].values
    cum_pnl_vals = df_hist["cum_pnl"].values
    
    # Determine color: green if new high, else red
    running_max = np.maximum.accumulate(cum_pnl_vals)
    drawdowns = running_max - cum_pnl_vals
    
    # Map drawdowns to a color map (0 = green, >0 = red)
    # We will use a custom listed colormap: 0 -> Green, 1 -> Red
    cmap = ListedColormap(['green', 'red'])
    colors = np.where(drawdowns <= 0.0, 0, 1)
    
    ax2.set_xlim(steps.min(), steps.max())
    ax2.set_ylim(cum_pnl_vals.min() - 1, cum_pnl_vals.max() + 1)
    
    # We plot colored line segments
    colorline(steps, cum_pnl_vals, z=colors, cmap=cmap, linewidth=2, ax=ax2)
    
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Cumulative PnL")
    ax2.set_title("Panel B: The Lifeblood (Equity Curve)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    out_path = os.path.join(logs_dir, "trade_visualization.png")
    plt.savefig(out_path, dpi=300)
    print(f"Visualization saved to {out_path}")

# Patch colorline for ax support
def colorline(x, y, z=None, cmap=plt.get_cmap('RdYlGn'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, ax=None):
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)
    return lc

if __name__ == "__main__":
    main()
