# Trade Resampling

Last updated: 2026-03-10

## Purpose

This is the first implementation step toward the legacy-style robustness layer:

1. export closed trades from the canonical evaluator
2. bootstrap those trades
3. attach summary statistics back to the candidate registry
4. use those summaries in saved filters and future allocation logic

## Current Implementation

Core modules:

1. [trading_eval.py](/home/user/mcs/umc_nn/trading_eval.py)
2. [resampling.py](/home/user/mcs/umc_nn/candidates/resampling.py)
3. [registry.py](/home/user/mcs/umc_nn/candidates/registry.py)

CLI:

1. [export_candidate_trades.py](/home/user/mcs/scripts/export_candidate_trades.py)
2. [run_candidate_resampling.py](/home/user/mcs/scripts/run_candidate_resampling.py)

## Trade Export Contract

Closed trades are exported from the same evaluation path used for metrics.

Each exported trade now carries:

1. period name
2. direction
3. entry and exit step
4. entry and exit timestamp when available
5. entry and exit price
6. quantity
7. gross PnL
8. net PnL
9. fees paid
10. duration
11. entry and exit balance
12. return on equity

This is enough for:

1. bootstrap resampling
2. future similarity metrics
3. future sizing experiments

## Current Resampling Model

Implemented mode:

1. `fractional_returns`

Meaning:

1. each trade is represented by `return_on_equity`
2. bootstrap resamples trades with replacement
3. the resampled equity path is replayed as:
   `equity *= (1 + fraction * trade_return)`

This is intentionally simple.
It is the right first step for:

1. pessimistic return estimates
2. drawdown distribution estimates
3. future fraction scans

It is not yet the full overlap-aware legacy engine for multi-system portfolios.

## Current CLI Usage

Export trades:

```bash
.venv/bin/python scripts/export_candidate_trades.py --registry-root candidate_registry
```

Run bootstrap:

```bash
.venv/bin/python scripts/run_candidate_resampling.py \
  --registry-root candidate_registry \
  --period train \
  --iterations 500 \
  --fraction 0.5 \
  --fraction 1.0 \
  --fraction 1.5 \
  --name-prefix train_bootstrap
```

Filter by resampling:

```bash
.venv/bin/python scripts/apply_candidate_rules.py \
  --registry-root candidate_registry \
  --rule resampling_results.train_bootstrap_f1.00.profitable_rate:gte:0.9 \
  --rule resampling_results.train_bootstrap_f1.00.p05_net_profit:gt:-10
```

## Current Observations On The Walk-Forward Probe Candidates

After exporting trades and running bootstrap on the current six walk-forward
candidates:

1. two candidates look relatively strong on train bootstrap under a shallow
   pessimistic-loss rule
2. neither of those two is the one with the only positive adjacent OOS
3. the one adjacent-OOS-positive candidate still has weak train-bootstrap tail
   behavior

Practical interpretation:

1. raw train PnL and adjacent OOS survival are not enough
2. train bootstrap and adjacent OOS are already disagreeing in useful ways
3. the registry is now able to expose that disagreement explicitly

## What This Enables Next

This layer is the setup for:

1. more realistic resampling objectives
2. fraction scans and future `optimal f` style experiments
3. candidate similarity metrics from trade streams
4. operator filters combining:
   OOS behavior,
   bootstrap robustness,
   and later diversification
5. portfolio-level allocator workbench

## Current Limitation

The single-candidate bootstrap engine is still single-stream.

Portfolio overlap handling now exists in a separate layer:

1. [allocator.py](/home/user/mcs/umc_nn/candidates/allocator.py)

Current status:

1. candidate-level bootstrap remains trade-based and single-stream
2. portfolio-level workbench uses synchronized common-window block bootstrap
3. exact legacy trade-overlap replay is still a future refinement
