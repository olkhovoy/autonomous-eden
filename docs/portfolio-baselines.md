# Portfolio Baselines

Last updated: 2026-03-10

## Purpose

The portfolio baseline layer is the first explicit judge of the whole conveyor.

It compares the current conveyor portfolio against simpler alternatives on the
same rolling forward windows.

Current baseline set:

1. `flat`
2. `long`
3. `equal_weight_selected_subset`
4. `single_best_candidate`
5. `naive_top_oos_rotation`

## Current Implementation

Core module:

1. [portfolio_baselines.py](/home/user/mcs/umc_nn/candidates/portfolio_baselines.py)

CLI:

1. [run_portfolio_baselines.py](/home/user/mcs/scripts/run_portfolio_baselines.py)

Registry artifact directory:

1. [candidate_registry/portfolio_baselines](/home/user/mcs/candidate_registry/portfolio_baselines)

## Current Real Smoke Run

Current report:

1. [reuse_rolling_20250501_baselines.json](/home/user/mcs/candidate_registry/portfolio_baselines/reuse_rolling_20250501_baselines.json)

Observed result:

1. conveyor PnL: `+430.79`
2. gate verdict: `False`
3. beaten baselines: `flat`, `single_best_candidate`, `naive_top_oos_rotation`
4. missed baselines: `long`, `equal_weight_selected_subset`

The closest miss is important:

1. vs `equal_weight_selected_subset`: `-6.42` PnL delta

That means the current conveyor is not yet clearly earning its complexity over
a simpler equal-weight policy on the same chosen subset.

## Default Gates

The first gate layer is intentionally simple:

1. total PnL must be non-negative
2. total max drawdown must stay under the configured limit
3. required baselines must be beaten by total PnL
4. at least a minimum number of baselines must be beaten overall

Current default required baselines:

1. `flat`
2. `equal_weight_selected_subset`

## Usage

```bash
.venv/bin/python scripts/run_portfolio_baselines.py \
  --registry-root candidate_registry \
  --portfolio-ledger-report reuse_rolling_20250501_portfolio \
  --report-name reuse_rolling_20250501_baselines
```

## Next Step

The immediate next development target is:

1. candidate-farm expansion on larger rolling runs, judged by these gates
