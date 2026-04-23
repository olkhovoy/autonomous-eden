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
    parser = argparse.ArgumentParser(description="Visualize UMC_Cell Monolith OOS")
    parser.add_argument("--neurobars", action="store_true", help="Use pre-computed Neurobars")
    parser.add_argument("--start_step", type=int, default=2000000, help="Start step for OOS evaluation")
    parser.add_argument("--max_steps", type=int, default=20000, help="Number of steps for OOS evaluation")
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

    print("Loading the Monolith Genome for OOS validation...")
    
    # 1. Loading the Soul
    weights_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "monolith_best_weights_v2.npy")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        sys.exit(1)
        
    vector = np.load(weights_path)
    
    # Setup Data & Env
    if args.neurobars:
        data_file = os.path.join(os.path.dirname(__file__), "..", "data", "BTCUSDT_parquet_neurobars.npz")
    else:
        print("Please use --neurobars")
        sys.exit(1)
    
    # Run a longer episode for visualization
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
        print(f"Genome successfully injected into UMC_Cell. Running OOS starting from step {args.start_step} for {args.max_steps} steps.")
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
    
    wins = 0
    trades = 0
    max_balance = args.initial_balance
    max_dd = 0.0
    
    with torch.no_grad():
        done = False
        while not done:
            x_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            
            action_logits, h_next = cell(x_tensor, h_prev)
            
            # 0: Flat, 1: Long, 2: Short
            action = torch.argmax(action_logits, dim=1).item()
            
            obs, reward, done, info = env.step(action)
            h_prev = h_next
            
            prices.append(env.prices[env.current_step])
            balances.append(float(env.balance))
            actions.append(action)
            positions.append(env.current_position)
            
            if env.balance > max_balance:
                max_balance = env.balance
            
            dd = (max_balance - env.balance) / max_balance
            if dd > max_dd:
                max_dd = dd
                
            if (env.current_step - args.start_step) % 2000 == 0:
                print(f"  Step {env.current_step - args.start_step}/{args.max_steps} | Balance: {env.balance:.2f} | DD: {max_dd*100:.2f}%")
                
            if env.current_step - args.start_step >= args.max_steps:
                break

    # Calculate actual trades and win rate correctly
    position = 0
    entry_price = 0
    for i in range(1, len(positions)):
        if positions[i] != position:
            # We changed position
            if position != 0:
                # We closed a trade
                trades += 1
                exit_price = prices[i]
                pnl = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
                if pnl > 0:
                    wins += 1
            if positions[i] != 0:
                # We opened a trade
                entry_price = prices[i]
            position = positions[i]

    win_rate = (wins / trades * 100) if trades > 0 else 0.0

    print(f"\n====================================")
    print(f"OOS EVALUATION COMPLETE")
    print(f"====================================")
    print(f"Final Balance:  {env.balance:.2f} USDT")
    print(f"Net PnL:        {env.balance - args.initial_balance:.2f} USDT")
    print(f"Max Drawdown:   {max_dd*100:.2f}%")
    print(f"Total Trades:   {trades}")
    print(f"Win Rate:       {win_rate:.2f}%")
    print(f"====================================\n")

    if env.balance > args.initial_balance:
        print("THE CELL SURVIVED. IT IS NOT A STATISTICAL ANOMALY.")
    else:
        print("The Cell was consumed by the Generative Interface's entropy.")

    # 3. The Rendering (Matplotlib)
    print("Rendering visualization...")
    
    # Use a dark, corporate aesthetic
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    fig.patch.set_facecolor('#1e1e2e')
    
    steps = np.arange(len(prices))
    prices = np.array(prices)
    balances = np.array(balances)
    positions = np.array(positions)
    
    # Top Panel: Market Price
    ax1.set_facecolor('#1e1e2e')
    ax1.plot(steps, prices, color='#a6accd', linewidth=1, alpha=0.8, label='BTC/USDT Price')
    
    # Plot Entry/Exit markers & background shading
    # Due to length, we might only plot shading and not dots, or dots will be too dense
    for i in range(1, len(steps)):
        curr_pos = positions[i]
        if curr_pos == 1:
            ax1.axvspan(i-1, i, facecolor='#40a02b', alpha=0.1) # Soft green for Long
        elif curr_pos == 2:
            ax1.axvspan(i-1, i, facecolor='#d20f39', alpha=0.1) # Soft red for Short
            
    ax1.set_title("OOS Validation (20,000 steps)\nMarket Price & Agent Positions", color='#cdd6f4', fontsize=14, pad=15)
    ax1.set_ylabel("Price (USDT)", color='#cdd6f4', fontsize=12)
    ax1.tick_params(colors='#bac2de')
    for spine in ax1.spines.values():
        spine.set_color('#45475a')
    ax1.grid(True, color='#313244', linestyle='--', alpha=0.3)
    
    # Bottom Panel: Equity Curve
    ax2.set_facecolor('#1e1e2e')
    
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
    ax2.grid(True, color='#313244', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "monolith_OOS_evaluation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"OOS Chart successfully saved to: {output_path}")

if __name__ == "__main__":
    visualize_monolith()
