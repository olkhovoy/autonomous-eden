# Tradeforward Evaluator

Last updated: 2026-03-10

## Purpose

The tradeforward evaluator is the first forward-looking feedback layer of the
conveyor.

It takes a selected tradeforward plan and measures what actually happened on the
next unseen window.

That closes the first loop between:

1. research selection
2. expected portfolio behavior
3. actual forward behavior

## Current Implementation

Core module:

1. [tradeforward_eval.py](/home/user/mcs/umc_nn/candidates/tradeforward_eval.py)

CLI:

1. [evaluate_tradeforward_plan.py](/home/user/mcs/scripts/evaluate_tradeforward_plan.py)

Registry artifact directory:

1. [candidate_registry/tradeforward_eval](/home/user/mcs/candidate_registry/tradeforward_eval)

## Current Real Evaluation

Current report:

1. [reuse_cycle_20250508_tradeforward_eval.json](/home/user/mcs/candidate_registry/tradeforward_eval/reuse_cycle_20250508_tradeforward_eval.json)

Observed result:

1. actual portfolio PnL: `+200.31`
2. actual portfolio max DD: `1.66%`
3. expected original scenario PnL: `+138.24`
4. delta vs expected original: `+62.07`
5. delta vs expected pessimistic p05: `+367.78`

## What It Emits

The current report includes:

1. source plan and source scenario
2. expected scenario statistics from allocator or combination search
3. actual forward portfolio outcome
4. per-candidate forward metrics
5. forward portfolio equity curve
6. deltas between actual and expected behavior

## Why It Matters

This closes the first forward-feedback loop.

Without it, `tradeforward` is only a selection handoff.
With it, the system accumulates evidence about whether selected systems
continue to behave acceptably after the review window.

This evaluator is now consumed by the rolling conveyor simulator:

1. [reuse_rolling_20250501.json](/home/user/mcs/candidate_registry/rolling/reuse_rolling_20250501.json)
2. and then by the lifecycle supervisor:
   [reuse_rolling_20250501_lifecycle.json](/home/user/mcs/candidate_registry/lifecycle/reuse_rolling_20250501_lifecycle.json)

## Next Step

The immediate next development target is:

1. portfolio ledger and rebalance accounting on top of lifecycle decisions
