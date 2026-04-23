# CoinGlass Slow-Context v1 A/B Results

Date: 2026-03-17

## Setup

- Baseline branch: `baseline_market_only_v1`
- CoinGlass branch: `coinglass_slow_v1`
- Ladder executed:
  `train -> export -> probe`
- Probe config:
  - `runs_per_window = 2`
  - `generations = 8`
  - `population_size = 8`
  - `exchange = binance`
  - `train_days = 7`
  - `oos_days = 7`

## Representation Metrics

- Baseline: `val_score = 0.379888`
- CoinGlass: `val_score = 0.114793`

Interpretation:
- `coinglass_slow_v1` materially improved representation-side validation metrics.

## Walk-Forward Probe Summary

Source artifacts:
- `/home/user/mcs/checkpoints/monolith_walkforward_probe/baseline_market_only_v1/summary.json`
- `/home/user/mcs/checkpoints/monolith_walkforward_probe/coinglass_slow_v1/summary.json`

### Baseline

- `runs = 6`
- `train_positive_runs = 1`
- `oos_positive_runs = 1`
- `oos_profitable_and_active_runs = 1`
- `median_train_pnl = -141.36`
- `median_oos_pnl = -654.40`
- `median_oos_trades = 58.0`

### CoinGlass

- `runs = 6`
- `train_positive_runs = 3`
- `oos_positive_runs = 2`
- `oos_profitable_and_active_runs = 0`
- `median_train_pnl = +129.38`
- `median_oos_pnl = -241.34`
- `median_oos_trades = 0.0`

## Verdict

Status: `stop at probe gate`

Why:
- The agreed secondary diagnostic gate required improvement in `oos_profitable_and_active_runs` before spending more compute on `rolling`.
- Baseline achieved `1`.
- `coinglass_slow_v1` achieved `0`.

What improved:
- Better representation metrics.
- Better median train PnL.
- Better median OOS PnL.
- More OOS-positive runs.

What failed:
- The improvement collapsed into inactive or effectively flat OOS behavior.
- The branch did not produce a single OOS run that was both profitable and active.
- Therefore it did not earn promotion to `rolling` under the current stop/go policy.

## Next Step

Do not add more sources yet.

First iterate on `CoinGlass` feature design:
- freshness thresholds
- availability lags
- ETF visibility rules
- feature shortlist and normalization
- activity-aware objectives downstream

Only rerun `rolling` after `probe` improves the active OOS criterion.
