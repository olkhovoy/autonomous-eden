#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

# Fix potential hanging issue with OpenMP/MKL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

print("Importing PyTorch...")
import torch
print("Imported torch successfully.")
print("Importing Rust Module...")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "gggp_bundle"))
import semiotic_hypercube as umc_core
print("Imported rust module successfully.")

import csv
import time
from scipy.stats import entropy

ROOT = os.path.join(os.path.dirname(__file__), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from umc_nn.candidate_engines import ACTION_HEAD_MODES, ENGINE_FAMILIES, EngineConfig, engine_num_parameters
from umc_nn.pure_env import EXECUTION_FEE_MODES, EXCHANGE_FEE_PRESETS, POSITION_SIZING_MODES, PureTradingEnv
from umc_nn.trading_eval import evaluate_engine_vector

# Global state for logging
GLOBAL_STATE = {
    "evals": 0,
    "pop_size": 30,
    "current_gen": 0,
    "best_fitness_this_gen": -1e9,
    "gen_fitness_sum": 0.0,
    "gen_profit_sum": 0.0,
    "gen_drawdown_sum": 0.0,
    "gen_trades_sum": 0.0,
    "gen_win_rate_sum": 0.0,
    "gen_start_time": time.time(),
    "valid_individuals_count": 0,
    "filtered_out_count": 0,
    "is_filtering_phase": True,
    "use_neurobars": False,
    "cached_env": None,
    "env_kwargs": {},
    "data_file": None,
    "start_step": 1982117,
    "max_steps": 1052640,
    "skip_filtering_phase": False,
    "engine_config": EngineConfig(),
    "fitness_profile": "hunter",
    "min_trades": 5,
    "activity_target_trades": 12,
    "trade_band_low": None,
    "trade_band_high": None,
    "trade_band_floor": 0.25,
}


def _trade_band_multiplier(trades_count: int) -> float:
    low = GLOBAL_STATE["trade_band_low"]
    high = GLOBAL_STATE["trade_band_high"]
    floor = float(GLOBAL_STATE["trade_band_floor"])
    if low is None and high is None:
        return 1.0
    if low is not None and trades_count < int(low):
        ratio = float(trades_count) / max(float(low), 1.0)
        return max(floor, ratio)
    if high is not None and trades_count > int(high):
        ratio = float(high) / max(float(trades_count), 1.0)
        return max(floor, ratio)
    return 1.0

def evaluate_cell(vector: np.ndarray) -> float:
    """
    Fitness function for the UMC_Cell genome.
    Evaluates the injected parameters in a deterministic pure environment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cache environment to avoid reloading 25MB JSON/NPZ on every evaluation
    if GLOBAL_STATE["cached_env"] is None:
        GLOBAL_STATE["cached_env"] = PureTradingEnv(
            max_steps=GLOBAL_STATE["max_steps"],
            data_path=GLOBAL_STATE["data_file"],
            use_neurobars=GLOBAL_STATE["use_neurobars"],
            start_step=GLOBAL_STATE["start_step"],
            **GLOBAL_STATE["env_kwargs"],
        )
    
    env = GLOBAL_STATE["cached_env"]
    try:
        trace = evaluate_engine_vector(
            env,
            vector,
            engine_config=GLOBAL_STATE["engine_config"],
            device=device,
        )
    except Exception as e:
        print(f"Error injecting weights: {e}")
        return -1000.0
    
    metrics = trace.metrics
    balance_history = trace.balance_history
    action_history = trace.action_history
    trades_count = metrics.trades
    win_rate = metrics.win_rate_pct
    profit = metrics.pnl
    max_drawdown = metrics.max_drawdown_pct
            
    # --- 1. COHERENCE (Smoothness of Survival) ---
    balance_history_arr = np.array(balance_history)
    returns = np.divide(
        np.diff(balance_history_arr),
        np.maximum(balance_history_arr[:-1], 1e-12),
        out=np.zeros(max(len(balance_history_arr) - 1, 0), dtype=np.float64),
        where=np.maximum(balance_history_arr[:-1], 1e-12) > 0,
    )
    downside_returns = returns[returns < 0]

    if len(downside_returns) > 0:
        downside_volatility = np.std(downside_returns)
        # Add small epsilon to prevent division by zero
        coherence = np.mean(returns) / (downside_volatility + 1e-8) 
    else:
        coherence = np.mean(returns) # Perfect coherence (no losses)

    # --- 2. STIMULATION (Richness of Choice) ---
    # Calculate the probability distribution of chosen actions
    if action_history:
        value_counts = np.unique(action_history, return_counts=True)[1]
        action_probs = value_counts / len(action_history)
        stimulation = entropy(action_probs) # High if actions are varied, 0 if always the same action
    else:
        stimulation = 0.0

    # --- 3. THE HUNTER FITNESS SCORE (Capitalist Apex Predator) ---
    # Fitness = Total PnL * (WinRate/100) * (1 / (1 + MaxDrawdown))
    if trades_count == 0:
        UMC_FITNESS = -1000.0
    else:
        dd_ratio = max_drawdown / 100.0
        fitness_profile = GLOBAL_STATE["fitness_profile"]
        if fitness_profile == "profit_dd":
            UMC_FITNESS = profit * (1.0 / (1.0 + dd_ratio))
        elif fitness_profile == "profit_active":
            activity_target = max(int(GLOBAL_STATE["activity_target_trades"]), 1)
            activity = min(trades_count / float(activity_target), 1.0)
            UMC_FITNESS = profit * activity * (0.5 + win_rate / 200.0) * (1.0 / (1.0 + dd_ratio))
        elif profit > 0:
            UMC_FITNESS = profit * (win_rate / 100.0) * (1.0 / (1.0 + dd_ratio))
        else:
            UMC_FITNESS = profit # Plain negative profit is penalizing enough

    trade_band_multiplier = _trade_band_multiplier(trades_count)
    if trade_band_multiplier < 1.0:
        if UMC_FITNESS >= 0.0:
            UMC_FITNESS *= trade_band_multiplier
        else:
            UMC_FITNESS /= max(trade_band_multiplier, 1e-8)
    
    # Selection Pressure for Generation 0
    # "не менее 5 сделок" (убрали profit >= 0, так как на 1 млн баров случайно найти профит нереально)
    is_valid = trades_count >= int(GLOBAL_STATE["min_trades"])
    
    if GLOBAL_STATE["skip_filtering_phase"]:
        GLOBAL_STATE["is_filtering_phase"] = False

    if GLOBAL_STATE["is_filtering_phase"]:
        if not is_valid:
            UMC_FITNESS = -1000.0 # Instant death
            GLOBAL_STATE["filtered_out_count"] += 1
            if GLOBAL_STATE["filtered_out_count"] % 100 == 0:
                print(f"  [Filtering Phase] Still searching... Filtered out {GLOBAL_STATE['filtered_out_count']} candidates so far.", end='\r')
        else:
            GLOBAL_STATE["valid_individuals_count"] += 1
            print(f"\n  [Filtering Phase] SURVIVOR FOUND ({GLOBAL_STATE['valid_individuals_count']}/{GLOBAL_STATE['pop_size']}) after {GLOBAL_STATE['filtered_out_count']} deaths: trades={trades_count:4d}, PnL={profit:6.1f}, win={win_rate:5.1f}%, Fit={UMC_FITNESS:.4f}")
            GLOBAL_STATE["filtered_out_count"] = 0 # Reset death counter for next survivor
            if GLOBAL_STATE["valid_individuals_count"] >= GLOBAL_STATE["pop_size"]:
                print("\n=== FILTERING PHASE COMPLETE. INITIAL POPULATION READY. ===\n")
                GLOBAL_STATE["is_filtering_phase"] = False
                GLOBAL_STATE["evals"] = 0 # Reset evaluation counter for the real generations
                GLOBAL_STATE["gen_start_time"] = time.time()
                
    elif not is_valid:
        UMC_FITNESS = -1000.0
        
    fitness = float(UMC_FITNESS)
    
    # Evaluate individual
    if not GLOBAL_STATE["is_filtering_phase"]:
        print(f"  Eval {GLOBAL_STATE['evals']+1}/{GLOBAL_STATE['pop_size']}                    ", end='\r')
        
        # Track statistics
        GLOBAL_STATE["evals"] += 1
        GLOBAL_STATE["gen_fitness_sum"] += fitness
        GLOBAL_STATE["gen_profit_sum"] += profit
        GLOBAL_STATE["gen_drawdown_sum"] += max_drawdown
        GLOBAL_STATE["gen_trades_sum"] += trades_count
        GLOBAL_STATE["gen_win_rate_sum"] += win_rate
        
        if fitness > GLOBAL_STATE["best_fitness_this_gen"]:
            GLOBAL_STATE["best_fitness_this_gen"] = fitness
            
        # Check if a generation is complete
        if GLOBAL_STATE["evals"] % GLOBAL_STATE["pop_size"] == 0:
            gen = GLOBAL_STATE["current_gen"]
            pop_size = GLOBAL_STATE["pop_size"]
            
            mean_fitness = GLOBAL_STATE["gen_fitness_sum"] / pop_size
            mean_profit = GLOBAL_STATE["gen_profit_sum"] / pop_size
            mean_dd = GLOBAL_STATE["gen_drawdown_sum"] / pop_size
            mean_trades = GLOBAL_STATE["gen_trades_sum"] / pop_size
            mean_win_rate = GLOBAL_STATE["gen_win_rate_sum"] / pop_size
            
            best_fitness = GLOBAL_STATE["best_fitness_this_gen"]
            elapsed = time.time() - GLOBAL_STATE["gen_start_time"]
            
            # Clear the evaluation counter line and print the generation summary
            print(f"\rGen {gen:03d} | Fit: {mean_fitness:6.2f} (Best: {best_fitness:6.2f}) | PnL: {mean_profit:6.1f} | DD: {mean_dd:4.1f}% | Trades: {mean_trades:4.0f} | Win: {mean_win_rate:4.1f}% | Time: {elapsed:4.1f}s")
            
            # Log to CSV
            log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "monolith_evolution.csv")
            file_exists = os.path.exists(log_path)
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["generation", "mean_fitness", "best_fitness", "mean_profit", "mean_drawdown_pct", "mean_trades", "mean_win_rate", "time_seconds"])
                writer.writerow([gen, mean_fitness, best_fitness, mean_profit, mean_dd, mean_trades, mean_win_rate, elapsed])
                
            # Reset for next generation
            GLOBAL_STATE["current_gen"] += 1
            GLOBAL_STATE["best_fitness_this_gen"] = -1e9
            GLOBAL_STATE["gen_fitness_sum"] = 0.0
            GLOBAL_STATE["gen_profit_sum"] = 0.0
            GLOBAL_STATE["gen_drawdown_sum"] = 0.0
            GLOBAL_STATE["gen_trades_sum"] = 0.0
            GLOBAL_STATE["gen_win_rate_sum"] = 0.0
            GLOBAL_STATE["gen_start_time"] = time.time()
            
            # We no longer pick a new dynamic epoch
            # START_IDX is fixed for the 2024-2025 period
            pass

    return fitness

def main():
    print("Script started.")
    parser = argparse.ArgumentParser(description="Evolve UMC_Cell Monolith")
    parser.add_argument("--neurobars", action="store_true", help="Use pre-computed Neurobars instead of raw OHLCV")
    parser.add_argument("--data-path", default=None, help="Optional custom JSON/NPZ data file")
    parser.add_argument("--start-step", type=int, default=1982117)
    parser.add_argument("--max-steps", type=int, default=1052640)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--skip-filtering-phase", action="store_true")
    parser.add_argument("--weights-output", default="checkpoints/monolith_best_weights_v2.npy")
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
    parser.add_argument("--engine-family", choices=sorted(ENGINE_FAMILIES), default="umc")
    parser.add_argument("--engine-hidden-dim", type=int, default=64)
    parser.add_argument("--engine-alpha", type=float, default=0.5)
    parser.add_argument("--action-head-mode", choices=sorted(ACTION_HEAD_MODES), default="argmax")
    parser.add_argument("--action-threshold", type=float, default=0.55)
    parser.add_argument("--fitness-profile", choices=["hunter", "profit_dd", "profit_active"], default="hunter")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--activity-target-trades", type=int, default=12)
    parser.add_argument("--trade-band-low", type=int, default=None)
    parser.add_argument("--trade-band-high", type=int, default=None)
    parser.add_argument("--trade-band-floor", type=float, default=0.25)
    args = parser.parse_args()
    
    if args.data_path is None:
        if args.neurobars:
            data_file = os.path.join(os.path.dirname(__file__), "..", "data", "BTCUSDT_parquet_neurobars.npz")
        else:
            data_file = os.path.join(os.path.dirname(__file__), "..", "data", "BTCUSDT_2025-04-24_12-20_to_2026-01-19_17-45.json")
    else:
        data_file = args.data_path

    inferred_neurobars = str(data_file).endswith(".npz")
    GLOBAL_STATE["use_neurobars"] = args.neurobars or inferred_neurobars
    GLOBAL_STATE["data_file"] = data_file
    GLOBAL_STATE["start_step"] = args.start_step
    GLOBAL_STATE["max_steps"] = args.max_steps
    GLOBAL_STATE["skip_filtering_phase"] = args.skip_filtering_phase
    GLOBAL_STATE["engine_config"] = EngineConfig(
        family=args.engine_family,
        hidden_dim=args.engine_hidden_dim,
        alpha=args.engine_alpha,
        action_head_mode=args.action_head_mode,
        action_threshold=args.action_threshold,
    )
    GLOBAL_STATE["engine_config"].validate()
    GLOBAL_STATE["fitness_profile"] = args.fitness_profile
    GLOBAL_STATE["min_trades"] = max(int(args.min_trades), 1)
    GLOBAL_STATE["activity_target_trades"] = max(int(args.activity_target_trades), 1)
    if args.trade_band_low is not None and args.trade_band_low <= 0:
        raise ValueError("--trade-band-low must be > 0 when provided")
    if args.trade_band_high is not None and args.trade_band_high <= 0:
        raise ValueError("--trade-band-high must be > 0 when provided")
    if (
        args.trade_band_low is not None
        and args.trade_band_high is not None
        and args.trade_band_high < args.trade_band_low
    ):
        raise ValueError("--trade-band-high must be >= --trade-band-low")
    if not (0.0 < args.trade_band_floor <= 1.0):
        raise ValueError("--trade-band-floor must be in (0, 1]")
    GLOBAL_STATE["trade_band_low"] = args.trade_band_low
    GLOBAL_STATE["trade_band_high"] = args.trade_band_high
    GLOBAL_STATE["trade_band_floor"] = float(args.trade_band_floor)
    GLOBAL_STATE["env_kwargs"] = {
        "initial_balance": args.initial_balance,
        "exchange": args.exchange,
        "maker_fee_rate": args.maker_fee_rate,
        "taker_fee_rate": args.taker_fee_rate,
        "execution_fee_mode": args.execution_fee_mode,
        "slippage": args.slippage,
        "position_sizing_mode": args.position_sizing_mode,
        "position_notional_fraction": args.position_notional_fraction,
        "leverage": args.leverage,
        "fixed_position_qty": args.fixed_position_qty,
    }
    
    print("Initializing SemanticHypercube with neuro_grammar.cfg...")
    grammar_path = os.path.join(os.path.dirname(__file__), "..", "gggp_bundle", "neuro_grammar.cfg")
    
    if not os.path.exists(grammar_path):
        print(f"Error: Grammar file not found at {grammar_path}")
        sys.exit(1)
        
    print("Loading SemanticHypercube...")
    hypercube = umc_core.SemanticHypercube(grammar_path)
    print("SemanticHypercube loaded.")
    
    # Calculate the exact dimension required by the UMCTradingCell
    dummy_env = PureTradingEnv(
        data_path=data_file,
        use_neurobars=GLOBAL_STATE["use_neurobars"],
        **GLOBAL_STATE["env_kwargs"],
    )
    dim = engine_num_parameters(dummy_env.input_dim, GLOBAL_STATE["engine_config"])
    
    generations = args.generations
    population_size = args.population_size
    
    GLOBAL_STATE["pop_size"] = population_size
    GLOBAL_STATE["gen_start_time"] = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n================================================")
    print("      THE MONOLITHIC RUN INITIATED              ")
    print("================================================")
    print(f"Device:          {device}")
    print(f"Architecture:    {GLOBAL_STATE['engine_config'].family}")
    input_mode = (
        f"Neurobars (Feature Dim {dummy_env.input_dim})"
        if GLOBAL_STATE["use_neurobars"]
        else f"Raw OHLCV (Dim {dummy_env.input_dim})"
    )
    print(f"Input Mode:      {input_mode}")
    print(f"Data File:       {data_file}")
    print(f"Dimensions:      {dim} (Weights & Biases)")
    print(f"Engine Config:   {GLOBAL_STATE['engine_config'].to_dict()}")
    print(f"Generations:     {generations}")
    print(f"Population Size: {population_size}")
    print(f"Evaluation:      {args.max_steps}-step deterministic loop from start_step={args.start_step}")
    print(
        f"Econ Model:      {args.initial_balance:.0f} USD | {args.exchange} | "
        f"{args.execution_fee_mode} | {args.position_sizing_mode}"
    )
    print(
        f"Positioning:     notional_fraction={args.position_notional_fraction:.2f} | "
        f"leverage={args.leverage:.2f} | fixed_qty={args.fixed_position_qty:.4f}"
    )
    print(f"Slippage:        {args.slippage:.5f}")
    print(f"Fitness:         {args.fitness_profile}")
    print(
        f"Trade Floor:     min_trades={GLOBAL_STATE['min_trades']} | "
        f"activity_target={GLOBAL_STATE['activity_target_trades']}"
    )
    print(
        f"Trade Band:      low={GLOBAL_STATE['trade_band_low']} | "
        f"high={GLOBAL_STATE['trade_band_high']} | "
        f"floor={GLOBAL_STATE['trade_band_floor']:.2f}"
    )
    print("Selection:       min_trades enforced across all generations")
    print("================================================\n")
    
    # Custom initial population loop to ensure we don't start with completely random "dead" cells.
    print("Filtering Generation 0 for minimum viability...")
    
    # Execute the Ask-Tell loop with the Rust optimizer
    print("Starting hypercube.evolve_with_fitness...")
    optimized_weights, out_array, final_fitness = hypercube.evolve_with_fitness(
        dim, 
        generations, 
        population_size,
        evaluate_cell
    )
    print("Evolution loop finished.")
    
    print("\n--- Evolution Complete ---")
    print(f"Final Fitness (Balance): {final_fitness:.4f}")
    
    # Save the optimized weights
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "checkpoints"), exist_ok=True)
    best_weights_path = os.path.join(os.path.dirname(__file__), "..", args.weights_output)
    os.makedirs(os.path.dirname(best_weights_path), exist_ok=True)
    np.save(best_weights_path, out_array)
    print(f"Saved optimized genome to {best_weights_path}")

if __name__ == "__main__":
    main()
