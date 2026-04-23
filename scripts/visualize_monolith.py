#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

# Adjust paths to import necessary modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "umc_nn"))
from umc_cell import UMCTradingCell
from pure_env import EXECUTION_FEE_MODES, EXCHANGE_FEE_PRESETS, POSITION_SIZING_MODES, PureTradingEnv

def visualize_monolith():
    parser = argparse.ArgumentParser(description="Visualize UMC_Cell Monolith")
    parser.add_argument("--neurobars", action="store_true", help="Use pre-computed Neurobars")
    parser.add_argument("--start-step", type=int, default=0, help="Start step for evaluation")
    parser.add_argument("--max-steps", type=int, default=5000, help="Number of steps to visualize")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--exchange", choices=sorted(EXCHANGE_FEE_PRESETS.keys()), default="binance")
    parser.add_argument("--maker-fee-rate", type=float, default=None)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--execution-fee-mode", choices=sorted(EXECUTION_FEE_MODES), default="taker")
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument(
        "--position-sizing-mode",
        choices=sorted(POSITION_SIZING_MODES),
        default="fraction_of_equity",
    )
    parser.add_argument("--position-notional-fraction", type=float, default=1.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--fixed-position-qty", type=float, default=1.0)
    args = parser.parse_args()

    print("Loading the Monolith Genome...")
    
    # 1. Loading the Soul
    weights_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "monolith_best_weights_v2.npy")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        sys.exit(1)
        
    vector = np.load(weights_path)
    
    # Setup Data & Env
    # Use a different time period for out-of-sample evaluation if possible
    # We will use the middle file: "BTCUSDT_2023-05-31_01-40_to_2025-04-24_12-19.json"
    if args.neurobars:
        data_file = os.path.join(os.path.dirname(__file__), "..", "data", "BTCUSDT_parquet_neurobars.npz")
    else:
        data_file = os.path.join(os.path.dirname(__file__), "..", "data", "BTCUSDT_2025-04-24_12-20_to_2026-01-19_17-45.json")
    
    # If out of sample is missing, fallback to the training data file
    if not os.path.exists(data_file):
        if args.neurobars:
            data_file = os.path.join(os.path.dirname(__file__), "..", "data", "BTCUSDT_2025-04-24_12-20_to_2026-01-19_17-45_neurobars.npz")
        else:
            data_file = os.path.join(os.path.dirname(__file__), "..", "data", "BTCUSDT_2025-04-24_12-20_to_2026-01-19_17-45.json")
    
    # Run a longer episode for visualization (e.g. 5000 steps)
    env = PureTradingEnv(
        max_steps=args.max_steps,
        initial_balance=args.initial_balance,
        exchange=args.exchange,
        maker_fee_rate=args.maker_fee_rate,
        taker_fee_rate=args.taker_fee_rate,
        execution_fee_mode=args.execution_fee_mode,
        slippage=args.slippage,
        position_sizing_mode=args.position_sizing_mode,
        position_notional_fraction=args.position_notional_fraction,
        leverage=args.leverage,
        fixed_position_qty=args.fixed_position_qty,
        data_path=data_file,
        use_neurobars=args.neurobars,
        start_step=args.start_step,
    )
    input_dim = env.input_dim
    hidden_dim = 64
    
    device = torch.device("cpu") # Visualization is fine on CPU
    cell = UMCTradingCell(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    
    try:
        cell.set_weights_from_vector(vector)
        print("Genome successfully injected into UMC_Cell (NC4 / K-M Active).")
    except Exception as e:
        print(f"Error injecting weights: {e}")
        sys.exit(1)
        
    cell.eval()
    
    # 2. The Final Crucible (Out-of-Sample Evaluation)
    print(f"Running deterministic evaluation on data: {os.path.basename(data_file)}...")
    obs = env.reset()
    h_prev = torch.zeros(1, hidden_dim, device=device)
    
    prices = []
    balances = []
    actions = []
    positions = []
    
    prices.append(env.prices[env.current_step])
    balances.append(float(env.balance))
    positions.append(env.current_position)
    
    with torch.no_grad():
        done = False
        while not done:
            x_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            
            action_logits, h_next = cell(x_tensor, h_prev)
            
            # 0: Flat, 1: Long, 2: Short
            action = torch.argmax(action_logits, dim=1).item()
            
            # If the current action is 0, we actually want to stay in whatever state we are in
            # No wait, downward causation rule NC3 says it should act.
            # But the agent was evaluated on 2000 steps, we should evaluate it on 2000 steps here.
            
            obs, reward, done, info = env.step(action)
            h_prev = h_next
            
            prices.append(env.prices[env.current_step])
            balances.append(float(env.balance))
            actions.append(action)
            positions.append(env.current_position)
            
            if (env.current_step - args.start_step) % 1000 == 0:
                print(f"  Step {env.current_step - args.start_step}/{args.max_steps} | Balance: {env.balance:.2f}")

    print(f"Evaluation Complete. Final Balance: {env.balance:.2f}")

    # 3. The Rendering (Matplotlib)
    print("Rendering visualization...")
    
    # Use a dark, corporate aesthetic
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    fig.patch.set_facecolor('#1e1e2e')
    
    steps = np.arange(len(prices))
    prices = np.array(prices)
    balances = np.array(balances)
    positions = np.array(positions)
    
    # Top Panel: Market Price
    ax1.set_facecolor('#1e1e2e')
    ax1.plot(steps, prices, color='#a6accd', linewidth=1, alpha=0.8, label='BTC/USDT Price')
    
    # Plot Entry/Exit markers & background shading
    # 0: Flat, 1: Long, 2: Short
    for i in range(1, len(steps)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        
        # Shade background based on current position
        if curr_pos == 1:
            ax1.axvspan(i-1, i, facecolor='#40a02b', alpha=0.1) # Soft green for Long
        elif curr_pos == 2:
            ax1.axvspan(i-1, i, facecolor='#d20f39', alpha=0.1) # Soft red for Short
            
        # Draw markers when position changes
        if prev_pos != curr_pos:
            if curr_pos == 1: # Entered Long
                ax1.scatter(i, prices[i], marker='^', color='#40a02b', s=100, zorder=5)
            elif curr_pos == 2: # Entered Short
                ax1.scatter(i, prices[i], marker='v', color='#d20f39', s=100, zorder=5)
            elif curr_pos == 0: # Closed to Flat
                ax1.scatter(i, prices[i], marker='x', color='#8c8fa1', s=80, zorder=4)

    ax1.set_title("VP-UMC Architecture: Deterministic Value Preservation\nMarket Price & Agent Positions", color='#cdd6f4', fontsize=14, pad=15)
    ax1.set_ylabel("Price (USDT)", color='#cdd6f4', fontsize=12)
    ax1.tick_params(colors='#bac2de')
    for spine in ax1.spines.values():
        spine.set_color('#45475a')
    ax1.grid(True, color='#313244', linestyle='--', alpha=0.5)
    
    # Bottom Panel: Equity Curve
    ax2.set_facecolor('#1e1e2e')
    
    # Color the equity curve based on whether it's above or below initial balance
    ax2.plot(steps, balances, color='#8aadf4', linewidth=2, label='Equity Curve (USD)')
    ax2.axhline(y=args.initial_balance, color='#6c7086', linestyle='--', linewidth=1)
    
    ax2.fill_between(steps, balances, args.initial_balance, where=(balances >= args.initial_balance), facecolor='#8aadf4', alpha=0.2, interpolate=True)
    ax2.fill_between(steps, balances, args.initial_balance, where=(balances < args.initial_balance), facecolor='#d20f39', alpha=0.2, interpolate=True)

    ax2.set_title("Cumulative Balance (PnL)", color='#cdd6f4', fontsize=12, pad=10)
    ax2.set_xlabel("Time Steps (Minutes)", color='#cdd6f4', fontsize=12)
    ax2.set_ylabel("Balance (USD)", color='#cdd6f4', fontsize=12)
    ax2.tick_params(colors='#bac2de')
    for spine in ax2.spines.values():
        spine.set_color('#45475a')
    ax2.grid(True, color='#313244', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "monolith_evaluation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Chart successfully saved to: {output_path}")

if __name__ == "__main__":
    visualize_monolith()
