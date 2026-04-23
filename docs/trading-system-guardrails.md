# Trading System Guardrails

Last updated: 2026-03-09

## Purpose

This document is the external memory for the trading project.
It exists to prevent the system from drifting back into toy-simulator mode
after context compression, model changes, or long coding sessions.

No new strategy claims are valid unless the checks in this document pass.

## Known Current Failures

These are not hypothetical risks. They are current blockers confirmed from
code and saved experiment artifacts.

1. `PureTradingEnv` now has an explicit notional model instead of the old
   implicit fixed-`1 BTC` behavior, but all strategy reports must state whether
   they used equity-based sizing or the legacy fixed-quantity mode.
2. The simulator now defaults to explicit equity-based sizing on a `10000 USD`
   account, but still omits funding, liquidation, maintenance margin, and lot
   rounding. Treat it as a simplified futures model, not exchange parity.
3. The current evolutionary objective favors "lose less while barely trading"
   over robust profitability.
4. The current neurobar pipeline normalizes and trains on full-history data,
   which contaminates later out-of-sample periods.
5. Existing experiment logs mix incompatible CSV schemas and cannot be treated
   as a clean time series of comparable runs.

Until these are fixed, any positive-looking equity curve is only a hint, not
evidence of a tradeable system.

## Simulator Contract

The simulator must satisfy these invariants.

1. `flat` must preserve capital exactly when there are no explicit holding costs.
2. A position change from `flat -> long` or `flat -> short` must charge one
   friction event.
3. A reversal `long -> short` or `short -> long` must charge two friction events.
4. PnL math, commission math, slippage math, and trade counting must be defined
   once and reused everywhere.
5. Reported trade statistics must use the same close/open semantics in
   evolution, visualization, and OOS evaluation.
6. Time windows must be explicit and reproducible. "train", "validation", and
   "OOS" may never be implicit.
7. The simulator must document its notional model:
   fixed size, balance-proportional size, leverage, lot rounding, and margin
   rules must all be explicit.

## Default Economics

Current default simulator assumptions:

1. `initial_balance = 10000 USD`
2. `position_sizing_mode = fraction_of_equity`
3. `position_notional_fraction = 1.0`
4. `leverage = 1.0`
5. `execution_fee_mode = taker`
6. Exchange presets:
   `binance = 0.020% maker / 0.050% taker`,
   `bybit = 0.020% maker / 0.055% taker`,
   `okx = 0.020% maker / 0.050% taker`

If any experiment deviates from these defaults, the override must be logged
with the run config and metric output.

## Evaluation Protocol

Use fixed UTC windows and compare every candidate against the same baselines.

Default windows for the consolidated neurobar history:

1. `2021-07-05 15:00:00` to `2023-05-31 01:40:00`
2. `2023-05-31 01:40:00` to `2025-04-24 12:20:00`
3. `2025-04-24 12:20:00` to `2026-01-19 17:45:00`
4. `2024-01-01 00:00:00` to `2026-01-01 00:00:00` as the current fixed
   train-like window used by monolith evolution

Mandatory baselines on every evaluated slice:

1. `flat`
2. `long`
3. `short`
4. `monolith` or any candidate policy under test

Minimum reported metrics:

1. Final balance
2. Net PnL
3. Max drawdown percent
4. Steps actually run
5. Closed trades
6. Win rate on closed trades
7. Action counts

## Strategy Claim Rules

A strategy is not allowed to be described as "working", "profitable", or
"robust" unless all of the following are true.

1. It beats `flat` net of costs on the exact same slice.
2. It remains solvent on the full requested window instead of dying early and
   being judged only on the prefix.
3. It is evaluated on at least one window that was not used for fitting or
   representation learning.
4. The encoder, normalization statistics, and fitness all respect temporal
   ordering.
5. The result can be reproduced from a saved config, code revision, and
   checkpoint path.

## Experiment Hygiene

Every experiment run should save:

1. Git commit or dirty-worktree marker
2. Python executable path
3. Data path and exact window
4. Policy checkpoint path
5. Full metric output for candidate and baselines
6. A schema version for CSV or JSON logs

Never append incompatible schemas to the same CSV file.

## Immediate Next Engineering Tasks

1. Keep all trading metrics in one reusable module.
2. Add simulator invariant tests and keep them green.
3. Run baseline comparisons before trying any new model change.
4. Repair the environment economics before claiming model failure or success.
5. Remove temporal leakage from encoder training and feature normalization.
