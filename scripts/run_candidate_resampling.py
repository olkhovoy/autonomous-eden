#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, closed_trace_for_period, closed_trades_for_period, scan_fraction_grid


def _candidate_ids(registry: CandidateRegistry, args: argparse.Namespace) -> list[str]:
    if args.candidate_id:
        return list(args.candidate_id)
    return [candidate.candidate_id for candidate in registry.list_candidates(status=args.status, tags=args.tag)]


def _parse_fractions(args: argparse.Namespace) -> list[float]:
    if args.fraction:
        return [float(item) for item in args.fraction]
    return [1.0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run replay-aware resampling for registry candidates.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--status")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--period", default="train", choices=["train", "valid", "oos", "oos_adjacent", "total"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", action="append", help="Repeat to scan several fractions.")
    parser.add_argument("--name-prefix", default="bootstrap")
    parser.add_argument(
        "--replay-mode",
        default="trace_block_bootstrap",
        choices=["trace_block_bootstrap", "identity_replay", "trade_bootstrap_legacy"],
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    registry = CandidateRegistry(ROOT / args.registry_root)
    fractions = _parse_fractions(args)

    for candidate_id in _candidate_ids(registry, args):
        candidate = registry.load_candidate(candidate_id)
        if candidate.trade_records_path is None:
            raise ValueError(f"Candidate {candidate_id} has no trade artifact. Run export_candidate_trades.py first.")

        trades = registry.load_trade_records(candidate_id)
        period_trades = closed_trades_for_period(trades, args.period)
        if not period_trades and args.period == "oos":
            period_trades = closed_trades_for_period(trades, "oos_adjacent")
        if not period_trades:
            print(candidate_id, "no trades for period", args.period)
            continue
        traces = registry.load_trace_records(candidate_id) if candidate.trace_records_path else []
        period_trace = closed_trace_for_period(traces, args.period)
        if period_trace is None and args.period == "oos":
            period_trace = closed_trace_for_period(traces, "oos_adjacent")

        initial_balance = float(candidate.manifest.econ_config["initial_balance"]) if candidate.manifest else 10000.0
        results = scan_fraction_grid(
            period_trades,
            args.period,
            fractions=fractions,
            iterations=args.iterations,
            sample_size=args.sample_size,
            block_size=args.block_size,
            seed=args.seed,
            initial_balance=initial_balance,
            name_prefix=args.name_prefix,
            trace=period_trace,
            replay_mode=args.replay_mode,
        )
        registry.attach_resampling_results(
            candidate_id,
            {result.name: result for result in results},
            overwrite=True,
        )
        for result in results:
            print(
                candidate_id,
                result.name,
                result.replay_mode,
                f"orig_np={result.original_net_profit:+.2f}",
                f"p05_np={result.p05_net_profit:+.2f}",
                f"p95_dd={result.p95_max_drawdown_pct:.2f}",
                f"profit_rate={result.profitable_rate:.3f}",
            )


if __name__ == "__main__":
    main()
