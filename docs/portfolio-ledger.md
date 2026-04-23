# Portfolio Ledger

Last updated: 2026-03-10

## Purpose

The portfolio ledger is the first service-level history of the conveyor.

It turns:

1. rolling cycle outcomes
2. lifecycle state transitions
3. tradeforward allocations

into one explicit record of:

1. portfolio equity over time
2. reserve fraction over time
3. rebalance turnover
4. add/remove/increase/decrease transitions between cycles
5. cluster concentration and service-state diagnostics

## Current Implementation

Core module:

1. [portfolio_ledger.py](/home/user/mcs/umc_nn/candidates/portfolio_ledger.py)

CLI:

1. [run_portfolio_ledger.py](/home/user/mcs/scripts/run_portfolio_ledger.py)

Registry artifact directory:

1. [candidate_registry/portfolio_ledger](/home/user/mcs/candidate_registry/portfolio_ledger)

## Current Real Smoke Run

Current report:

1. [reuse_rolling_20250501_portfolio.json](/home/user/mcs/candidate_registry/portfolio_ledger/reuse_rolling_20250501_portfolio.json)

Observed result:

1. final balance: `10430.79`
2. total PnL: `+430.79`
3. total return: `+4.31%`
4. max DD: `2.80%`
5. total gross turnover: `1.05`
6. average reserve fraction: `0.475`
7. non-tradable selected candidates: `2`

Cycle highlights:

1. cycle 1 added `wf_03_20250501_run02` and `wf_01_20240101_run01`
2. cycle 2 removed `wf_03_20250501_run02`
3. cycle 1 selected only `research`-status candidates, which is now visible as a service-state mismatch

## Rebalance Cost Model

The first pass keeps rebalance cost optional.

Current default:

1. `turnover_cost_rate = 0.0`

That means the ledger reproduces the rolling conveyor equity exactly while
still tracking turnover and churn.

If a non-zero `turnover_cost_rate` is provided, the ledger applies an
estimated pre-cycle rebalance cost proportional to gross turnover and starting
equity for that cycle.

## Usage

```bash
.venv/bin/python scripts/run_portfolio_ledger.py \
  --registry-root candidate_registry \
  --rolling-report reuse_rolling_20250501 \
  --lifecycle-report reuse_rolling_20250501_lifecycle \
  --report-name reuse_rolling_20250501_portfolio
```

Optional turnover cost example:

```bash
.venv/bin/python scripts/run_portfolio_ledger.py \
  --registry-root candidate_registry \
  --rolling-report reuse_rolling_20250501 \
  --lifecycle-report reuse_rolling_20250501_lifecycle \
  --report-name reuse_rolling_20250501_portfolio_costed \
  --turnover-cost-rate 0.001
```

## Next Step

The immediate next development target is:

1. candidate-farm expansion guided by the new baseline/gate layer

The current baseline layer already consumes this ledger output:

1. [reuse_rolling_20250501_baselines.json](/home/user/mcs/candidate_registry/portfolio_baselines/reuse_rolling_20250501_baselines.json)
