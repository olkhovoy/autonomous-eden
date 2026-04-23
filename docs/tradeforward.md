# Tradeforward

Last updated: 2026-03-10

## Purpose

`tradeforward` is the handoff from research selection to the next unseen window.

It is not live trading yet.
It is the explicit artifact that says:

1. which subset was chosen
2. with which weights
3. from which reports
4. for which next forward window

That artifact should exist on disk, not only in operator memory.

## Current Implementation

Core module:

1. [tradeforward.py](/home/user/mcs/umc_nn/candidates/tradeforward.py)

CLI:

1. [build_tradeforward_plan.py](/home/user/mcs/scripts/build_tradeforward_plan.py)

Registry artifact directory:

1. [candidate_registry/tradeforward](/home/user/mcs/candidate_registry/tradeforward)

Evaluator module:

1. [tradeforward_eval.py](/home/user/mcs/umc_nn/candidates/tradeforward_eval.py)

Evaluator CLI:

1. [evaluate_tradeforward_plan.py](/home/user/mcs/scripts/evaluate_tradeforward_plan.py)

Evaluation artifact directory:

1. [candidate_registry/tradeforward_eval](/home/user/mcs/candidate_registry/tradeforward_eval)

## Current Real Plan

Current real handoff:

1. [reuse_cycle_20250508_tradeforward.json](/home/user/mcs/candidate_registry/tradeforward/reuse_cycle_20250508_tradeforward.json)

Observed result on the current `reuse` smoke run:

1. source mode: `combination`
2. chosen scenario: `subset_1_cand_eb598e21812d_risk_0.50`
3. selected candidate: `wf_01_20240101_run01`
4. allocated risk fraction: `0.35`
5. reserve fraction: `0.65`
6. forward window: `2025-05-15 00:00:00 -> 2025-05-22 00:00:00 UTC`

## Current Real Evaluation

Current evaluated result:

1. [reuse_cycle_20250508_tradeforward_eval.json](/home/user/mcs/candidate_registry/tradeforward_eval/reuse_cycle_20250508_tradeforward_eval.json)

Observed result:

1. actual forward PnL: `+200.31`
2. expected original scenario PnL: `+138.24`
3. delta vs expected original: `+62.07`
4. actual forward max DD: `1.66%`
5. source candidate traded `6` times on the forward span

## Usage

```bash
.venv/bin/python scripts/build_tradeforward_plan.py \
  --registry-root candidate_registry \
  --plan-name tf_next_window \
  --combination-report reuse_cycle_20250508_combinations \
  --forward-start-utc '2025-05-15 00:00:00' \
  --forward-end-utc '2025-05-22 00:00:00'
```

```bash
.venv/bin/python scripts/evaluate_tradeforward_plan.py \
  --registry-root candidate_registry \
  --plan-name reuse_cycle_20250508_tradeforward \
  --report-name reuse_cycle_20250508_tradeforward_eval
```

## Limits

Still pending:

1. lifecycle transition from `research` to `paper`/`active`
2. paper/shadow execution bridge
3. policy for keeping, draining, or retiring systems after repeated forward windows

This layer is now chained historically by the rolling conveyor simulator:

1. [reuse_rolling_20250501.json](/home/user/mcs/candidate_registry/rolling/reuse_rolling_20250501.json)

Follow-up roadmap:

1. [next-development-roadmap.md](./next-development-roadmap.md)
2. [multi-market-architecture.md](./multi-market-architecture.md)
