#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.pure_env import EXECUTION_FEE_MODES, EXCHANGE_FEE_PRESETS, POSITION_SIZING_MODES, PureTradingEnv
from umc_nn.trading_eval import DEFAULT_DATE_WINDOWS, build_policy, resolve_date_window


@dataclass
class PolicyTrace:
    actions: np.ndarray
    positions: np.ndarray
    balances: np.ndarray
    trades: int
    final_balance: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1 neurobar overview visualization.")
    parser.add_argument("--data-path", default="data/BTCUSDT_parquet_neurobars.npz")
    parser.add_argument("--output-path", default="checkpoints/neurobars_phase1_overview.png")
    parser.add_argument("--start-step", type=int, help="Explicit start index")
    parser.add_argument("--max-steps", type=int, default=2048, help="Number of steps to render")
    parser.add_argument(
        "--date-window",
        choices=sorted(DEFAULT_DATE_WINDOWS.keys()),
        default="2025_2026",
        help="Named UTC date window if --start-step is not provided",
    )
    parser.add_argument(
        "--policy",
        choices=["none", "flat", "long", "short", "monolith"],
        default="none",
        help="Optional policy overlay for actions/equity",
    )
    parser.add_argument("--weights-path", default="checkpoints/monolith_best_weights_v2.npy")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default=None)
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
    parser.add_argument(
        "--latent-order",
        choices=["none", "variance", "abs_change"],
        default="abs_change",
        help="Reorder latent dimensions within the rendered window",
    )
    parser.add_argument("--max-latent-z", type=float, default=3.0)
    return parser


def _resolve_window(args: argparse.Namespace) -> tuple[int, int]:
    if args.start_step is not None:
        return args.start_step, args.max_steps

    start_utc, end_utc = DEFAULT_DATE_WINDOWS[args.date_window]
    start_step, window_steps = resolve_date_window(args.data_path, start_utc, end_utc)
    return start_step, min(args.max_steps, window_steps)


def _window_zscore(latents: np.ndarray) -> np.ndarray:
    mean = latents.mean(axis=0, keepdims=True)
    std = latents.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (latents - mean) / std


def _latent_order(latents: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.arange(latents.shape[1])
    if mode == "variance":
        scores = latents.std(axis=0)
        return np.argsort(scores)[::-1]
    diffs = np.abs(np.diff(latents, axis=0)).mean(axis=0)
    return np.argsort(diffs)[::-1]


def _timestamps_to_mpl_dates(timestamps: np.ndarray) -> np.ndarray:
    datetimes = [datetime.fromtimestamp(int(ts), tz=timezone.utc) for ts in timestamps]
    return mdates.date2num(datetimes)


def _simulate_policy_trace(args: argparse.Namespace, start_step: int, max_steps: int) -> PolicyTrace:
    env = PureTradingEnv(
        max_steps=max_steps,
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
        data_path=str(args.data_path),
        use_neurobars=True,
        start_step=start_step,
    )
    policy = build_policy(
        policy_name=args.policy,
        input_dim=env.input_dim,
        hidden_dim=args.hidden_dim,
        weights_path=args.weights_path if args.policy == "monolith" else None,
        device=args.device,
    )

    obs = env.reset()
    policy.reset()

    actions = []
    positions = [env.current_position]
    balances = [float(env.balance)]
    trades = 0
    prev_action = env.current_position

    done = False
    while not done:
        action = int(policy.act(obs))
        obs, _, done, _ = env.step(action)
        actions.append(action)
        positions.append(env.current_position)
        balances.append(float(env.balance))
        if action != prev_action and prev_action != 0:
            trades += 1
        prev_action = action

    return PolicyTrace(
        actions=np.asarray(actions, dtype=np.int8),
        positions=np.asarray(positions, dtype=np.int8),
        balances=np.asarray(balances, dtype=np.float64),
        trades=trades,
        final_balance=float(balances[-1]),
    )


def _add_position_markers(ax: plt.Axes, x: np.ndarray, prices: np.ndarray, positions: np.ndarray) -> None:
    for idx in range(1, len(positions)):
        prev_pos = positions[idx - 1]
        curr_pos = positions[idx]
        if curr_pos == prev_pos:
            continue
        if curr_pos == 1:
            ax.scatter(x[idx], prices[idx], marker="^", color="#2ecc71", s=18, zorder=6)
        elif curr_pos == 2:
            ax.scatter(x[idx], prices[idx], marker="v", color="#ff5c5c", s=18, zorder=6)
        else:
            ax.scatter(x[idx], prices[idx], marker="x", color="#d0d4e4", s=16, zorder=6)


def _shade_positions(ax: plt.Axes, x: np.ndarray, positions: np.ndarray) -> None:
    for idx in range(1, len(positions)):
        curr_pos = positions[idx]
        if curr_pos == 1:
            ax.axvspan(x[idx - 1], x[idx], facecolor="#2ecc71", alpha=0.07)
        elif curr_pos == 2:
            ax.axvspan(x[idx - 1], x[idx], facecolor="#ff5c5c", alpha=0.07)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    args.data_path = str(Path(args.data_path))
    args.output_path = Path(args.output_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_path)
    timestamps = data["timestamps"]
    prices = data["close_prices"]
    neurobars = data["neurobars"]

    start_step, max_steps = _resolve_window(args)
    end_step = min(start_step + max_steps, len(prices))
    if end_step - start_step < 32:
        raise ValueError("Window too small for a useful visualization")

    window_timestamps = timestamps[start_step:end_step]
    window_prices = prices[start_step:end_step]
    window_latents = neurobars[start_step:end_step]

    ordered_idx = _latent_order(window_latents, args.latent_order)
    ordered_latents = window_latents[:, ordered_idx]
    z_latents = _window_zscore(ordered_latents)
    z_latents = np.clip(z_latents, -args.max_latent_z, args.max_latent_z)

    x_dates = _timestamps_to_mpl_dates(window_timestamps)
    latent_norm = np.linalg.norm(window_latents, axis=1)
    latent_delta = np.concatenate([[0.0], np.linalg.norm(np.diff(window_latents, axis=0), axis=1)])

    policy_trace = None
    if args.policy != "none":
        policy_trace = _simulate_policy_trace(args, start_step=start_step, max_steps=end_step - start_step)
        if len(policy_trace.positions) != len(window_prices):
            min_len = min(len(policy_trace.positions), len(window_prices))
            window_prices = window_prices[:min_len]
            window_timestamps = window_timestamps[:min_len]
            x_dates = x_dates[:min_len]
            window_latents = window_latents[:min_len]
            z_latents = z_latents[:min_len]
            latent_norm = latent_norm[:min_len]
            latent_delta = latent_delta[:min_len]
            policy_trace.positions = policy_trace.positions[:min_len]
            policy_trace.balances = policy_trace.balances[:min_len]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    fig.patch.set_facecolor("#0b1020")
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.8, 1.6])

    ax_price = fig.add_subplot(gs[0])
    ax_ribbon = fig.add_subplot(gs[1], sharex=ax_price)
    ax_bottom = fig.add_subplot(gs[2], sharex=ax_price)

    ax_price.set_facecolor("#0f172a")
    ax_ribbon.set_facecolor("#0f172a")
    ax_bottom.set_facecolor("#0f172a")

    ax_price.plot(x_dates, window_prices, "-", color="#dbe4ff", linewidth=1.1, alpha=0.95)
    price_change_pct = ((window_prices[-1] / window_prices[0]) - 1.0) * 100.0
    ax_price.set_ylabel("Price")
    ax_price.set_title(
        f"Neurobar Phase 1 Overview\n"
        f"window={start_step}:{end_step}  steps={len(window_prices)}  price_change={price_change_pct:+.2f}%",
        fontsize=14,
        pad=10,
    )
    ax_price.grid(True, color="#22304d", linestyle="--", alpha=0.35)

    if policy_trace is not None:
        _shade_positions(ax_price, x_dates, policy_trace.positions)
        _add_position_markers(ax_price, x_dates, window_prices, policy_trace.positions)

    img = ax_ribbon.imshow(
        z_latents.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        interpolation="nearest",
        extent=[x_dates[0], x_dates[-1], -0.5, z_latents.shape[1] - 0.5],
        vmin=-args.max_latent_z,
        vmax=args.max_latent_z,
    )
    ax_ribbon.set_ylabel("Latent dim")
    ax_ribbon.set_title(f"Latent Ribbon ({args.latent_order} order, z-clipped to +/-{args.max_latent_z:.1f})", fontsize=12)
    cbar = fig.colorbar(img, ax=ax_ribbon, pad=0.01, aspect=30)
    cbar.set_label("Latent activation (window z-score)")

    if policy_trace is None:
        ax_bottom.plot(x_dates, latent_norm, "-", color="#5bc0eb", linewidth=1.2, label="latent_norm")
        ax_bottom.plot(x_dates, latent_delta, "-", color="#f6ae2d", linewidth=1.0, alpha=0.8, label="latent_delta")
        ax_bottom.set_ylabel("Latent activity")
        ax_bottom.legend(loc="upper left", frameon=False)
        bottom_summary = f"latent_norm mean={latent_norm.mean():.3f} | latent_delta mean={latent_delta.mean():.3f}"
    else:
        _shade_positions(ax_bottom, x_dates, policy_trace.positions)
        ax_bottom.step(x_dates, policy_trace.positions, where="post", color="#f6ae2d", linewidth=1.2, label="position")
        ax_bottom.set_ylabel("Position")
        ax_bottom.set_yticks([0, 1, 2])
        ax_bottom.set_yticklabels(["flat", "long", "short"])
        ax_balance = ax_bottom.twinx()
        ax_balance.plot(x_dates, policy_trace.balances, "-", color="#5bc0eb", linewidth=1.4, label="equity")
        ax_balance.axhline(args.initial_balance, color="#7f8ea3", linewidth=0.9, linestyle="--", alpha=0.8)
        ax_balance.set_ylabel("Equity")
        pnl = policy_trace.final_balance - args.initial_balance
        bottom_summary = (
            f"policy={args.policy} | final_balance={policy_trace.final_balance:.2f} | "
            f"pnl={pnl:+.2f} | trades={policy_trace.trades}"
        )

    ax_bottom.set_title(bottom_summary, fontsize=11)
    ax_bottom.grid(True, color="#22304d", linestyle="--", alpha=0.35)
    ax_bottom.set_xlabel("UTC time")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_bottom.xaxis.set_major_locator(locator)
    ax_bottom.xaxis.set_major_formatter(formatter)

    for ax in (ax_price, ax_ribbon, ax_bottom):
        ax.tick_params(colors="#dbe4ff")
        for spine in ax.spines.values():
            spine.set_color("#31415f")

    fig.savefig(args.output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved neurobar overview to {args.output_path}")
    print(f"window_steps:          {len(window_prices)}")
    print(f"start_step:            {start_step}")
    print(f"end_step:              {end_step}")
    print(f"latent_order:          {args.latent_order}")
    if policy_trace is not None:
        print(f"policy:                {args.policy}")
        print(f"final_balance:         {policy_trace.final_balance:.2f}")


if __name__ == "__main__":
    main()
