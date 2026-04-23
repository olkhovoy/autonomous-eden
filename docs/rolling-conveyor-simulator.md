# Rolling Conveyor Simulator

Last updated: 2026-03-10

## Purpose

The rolling conveyor simulator is the first historical end-to-end replay of the
whole research stack.

It repeats:

1. continuous search cycle
2. tradeforward plan
3. tradeforward evaluation
4. ledger compounding across sequential forward windows

This is the first layer where the conveyor can be judged as a machine instead
of as isolated artifacts.

## Current Implementation

Core module:

1. [rolling.py](/home/user/mcs/umc_nn/candidates/rolling.py)

CLI:

1. [run_rolling_conveyor_simulator.py](/home/user/mcs/scripts/run_rolling_conveyor_simulator.py)

Registry artifact directory:

1. [candidate_registry/rolling](/home/user/mcs/candidate_registry/rolling)

## Current Real Smoke Run

Current report:

1. [reuse_rolling_20250501.json](/home/user/mcs/candidate_registry/rolling/reuse_rolling_20250501.json)

Observed result:

1. `2` completed cycles
2. final balance: `10430.79`
3. total PnL: `+430.79`
4. total return: `+4.31%`
5. max DD: `2.80%`

Cycle summary:

1. cycle 1 `2025-05-08 -> 2025-05-15`: `+225.96`, DD `3.26%`, `2` selected systems
2. cycle 2 `2025-05-15 -> 2025-05-22`: `+204.83`, DD `1.66%`, `1` selected system

## Usage

```bash
.venv/bin/python scripts/run_rolling_conveyor_simulator.py \
  --registry-root candidate_registry \
  --report-name reuse_rolling_20250501 \
  --mode reuse \
  --selection-start-utc '2025-05-01 00:00:00' \
  --selection-days 7 \
  --forward-days 7 \
  --num-cycles 2 \
  --status research \
  --tag walkforward \
  --tag probe \
  --tag fused32
```

Additional unknown args are forwarded to
[run_continuous_search_cycle.py](/home/user/mcs/scripts/run_continuous_search_cycle.py),
so the same candidate-selection, resampling, shortlist, and allocator settings
can be reused across every cycle.

## What It Emits

The rolling report currently stores:

1. cycle-level selection and forward windows
2. chosen candidate sets per cycle
3. per-cycle portfolio PnL and DD
4. compounded ledger balance across cycles
5. aggregate return and max drawdown for the whole run

## Next Step

The immediate next development target is:

1. portfolio ledger and rebalance accounting on top of the lifecycle layer

The current lifecycle layer already consumes this rolling output:

1. [reuse_rolling_20250501_lifecycle.json](/home/user/mcs/candidate_registry/lifecycle/reuse_rolling_20250501_lifecycle.json)

The current portfolio ledger layer then consumes both rolling and lifecycle:

1. [reuse_rolling_20250501_portfolio.json](/home/user/mcs/candidate_registry/portfolio_ledger/reuse_rolling_20250501_portfolio.json)
