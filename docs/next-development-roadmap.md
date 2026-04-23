# Next Development Roadmap

Last updated: 2026-03-10

## Current State

Compact conveyor sequence:

`realistic trading env` -> `neurobars autoresearch` -> `multi-scale neurobars` -> `walk-forward candidate generation` -> `candidate registry` -> `trade export` -> `candidate resampling` -> `diversification` -> `clusters / concentration controls` -> `shortlist` -> `allocator workbench` -> `combination search` -> `operator overrides / audit trail` -> `dashboard feed` -> `operator UI` -> `continuous search cycle` -> `tradeforward` -> `tradeforward evaluator` -> `rolling conveyor simulator` -> `lifecycle state machine` -> `portfolio ledger` -> `rebalance logic` -> `portfolio-level baselines` -> `candidate-farm expansion` -> `UI expansion`

The project now has:

1. candidate generation and registry import
2. trade export and candidate resampling
3. diversification, clusters, shortlist, combinations, allocator
4. snapshot dashboard UI
5. continuous-cycle launcher
6. tradeforward handoff plan
7. tradeforward evaluator
8. rolling conveyor simulator
9. lifecycle state machine
10. portfolio ledger and rebalance logic
11. portfolio-level baselines and gates
12. manifest-driven candidate-farm expansion

This means the conveyor can now be judged as a portfolio machine against simple
alternatives.
What is still missing is scale: broader historical runs and stronger candidate
farms that can clear these gates consistently.

## Agreed Next Sequence

### 1. Tradeforward Evaluator

Build a runner that takes one tradeforward plan and evaluates it on the next
forward window.

Outputs should include:

1. forward PnL and drawdown
2. forward trade list
3. forward equity curve
4. comparison against allocator/combination expectations

Current state:

1. implemented
2. current real artifact:
   [reuse_cycle_20250508_tradeforward_eval.json](/home/user/mcs/candidate_registry/tradeforward_eval/reuse_cycle_20250508_tradeforward_eval.json)

This layer is complete and now feeds the rolling conveyor simulator.

### 2. Rolling Conveyor Simulator

Chain many cycles through history:

1. search on one training span
2. select and size on one review span
3. tradeforward on the next unseen span
4. roll forward and repeat

This is the first place where the whole system can be judged as a machine.

Current state:

1. implemented
2. current real artifact:
   [reuse_rolling_20250501.json](/home/user/mcs/candidate_registry/rolling/reuse_rolling_20250501.json)
3. current smoke-run result:
   `2` cycles, `+430.79` PnL, `+4.31%` return, `2.80%` max DD

This means the immediate next implementation target is now the lifecycle state
machine.

### 3. Lifecycle State Machine

Add explicit simulated states:

1. `research`
2. `approved`
3. `paper_sim`
4. `active_sim`
5. `draining`
6. `retired`

The forward evaluator should be able to move systems between these states.

Current state:

1. implemented
2. current real artifact:
   [reuse_rolling_20250501_lifecycle.json](/home/user/mcs/candidate_registry/lifecycle/reuse_rolling_20250501_lifecycle.json)
3. current smoke-run result:
   `wf_01_20240101_run01` moved `research -> paper`
   and `wf_03_20250501_run02` moved `research -> approved`

This means the immediate next implementation target is now the portfolio
ledger.

### 4. Portfolio Ledger

Track the whole conveyor, not only per-window winners:

1. total portfolio equity
2. active set over time
3. reserve fraction
4. turnover and rebalance cost
5. system churn
6. cluster concentration over time

Current state:

1. implemented together with rebalance logic
2. current real artifact:
   [reuse_rolling_20250501_portfolio.json](/home/user/mcs/candidate_registry/portfolio_ledger/reuse_rolling_20250501_portfolio.json)
3. current smoke-run result:
   `+430.79` PnL, `+4.31%` return, `2.80%` max DD,
   `1.05` gross turnover, `0.475` average reserve

This means the immediate next implementation target is now portfolio-level
baselines and gates.

### 5. Rebalance Logic

Allocator output is still a workbench.
It needed a real rebalance layer that models:

1. what stayed active
2. what was added
3. what was reduced
4. what was removed

Current state:

1. implemented in the portfolio ledger layer
2. current report tracks add/remove/increase/decrease transitions per cycle
3. turnover cost is optional and defaults to zero until a stronger cost model is agreed

### 6. Portfolio-Level Baselines And Gates

Judge the full conveyor against:

1. `flat`
2. `long`
3. equal-weight selected subset
4. single best candidate
5. naive top-OOS rotation

The historical conveyor should beat these often enough before any real-time
execution work begins.

Current state:

1. implemented
2. current real artifact:
   [reuse_rolling_20250501_baselines.json](/home/user/mcs/candidate_registry/portfolio_baselines/reuse_rolling_20250501_baselines.json)
3. current smoke-run verdict:
   gate `False`
4. current conveyor beats `flat`, `single_best_candidate`, and `naive_top_oos_rotation`
5. current conveyor lags `long` and slightly lags `equal_weight_selected_subset`

This means the immediate next development target is now candidate-farm
expansion under the new gates.

### 7. Candidate-Farm Expansion

Only after the historical conveyor is credible should we:

1. increase continuous generation throughput
2. add more search degrees of freedom
3. let `GGGP` search broader candidate families and later portfolio structures

Current state:

1. first manifest-driven farm runner implemented
2. each farm scenario now runs:
   `rolling -> lifecycle -> portfolio ledger -> portfolio baselines`
3. farm-level summary and registry artifacts implemented
4. reference doc:
   [candidate-farm-expansion.md](./candidate-farm-expansion.md)
5. current smoke artifact:
   [reuse_farm_smoke_light_202505.json](/home/user/mcs/candidate_registry/farms/reuse_farm_smoke_light_202505.json)
6. current smoke result:
   `2` scenarios completed, `1` gate pass, best scenario `+232.00` PnL
7. current farm layer now also supports:
   selection-anchored `generate`, template `cases`, cartesian `matrix` sweeps,
   and `--resume-completed`

This means the next major product-facing target is now UI expansion on top of
farm outputs and longer-running search batches.

### 8. UI Expansion

Once the forward simulator exists, extend the operator UI with:

1. cycle history
2. active/draining/retired system views
3. forward-vs-expected drift panels
4. stagnation and diversity-collapse warnings

## What Is Explicitly Deferred

Until the historical conveyor shows good profitability and robustness, do not
prioritize:

1. exchange adapters
2. real-time paper trading
3. live shadow mode
4. live capital routing

## Parallel Track: Multi-Stream Phase A

The project now also has a parallel architecture track for expanding the
information universe beyond one BTC price stream.

Reference docs:

1. [multi-stream-intelligence-architecture.md](/home/user/mcs/docs/multi-stream-intelligence-architecture.md)
2. [multi-stream-phase-a-source-matrix.md](/home/user/mcs/docs/multi-stream-phase-a-source-matrix.md)
3. [mcs2026q1_reviewed.md](/home/user/mcs/mcs2026q1_reviewed.md)

The immediate purpose of this track is not live news trading.
It is:

1. source selection
2. `available_at` discipline
3. access planning
4. context and regime features for the existing conveyor

Recommended first access wave:

Already available:

1. `FRED`
2. `Bybit`
3. `CoinGlass`

Next access wave:

1. `Binance`
2. `Deribit`
3. `Polygon`
4. one of `Freightos FBX` or `Truflation`
5. one of `Glassnode` or `CryptoQuant` only if `CoinGlass` coverage is insufficient
