# CoinGlass Slow-Context v1

Last updated: 2026-03-17

This document records the first practical CoinGlass integration branch for the
current BTC-first research stack.

## Purpose

The goal of `coinglass_slow_v1` is not to replace venue-native market data.
It is to test whether a small, explicit, and timing-disciplined CoinGlass
regime layer can improve the existing conveyor enough to pass the current
rolling portfolio gate.

## Important Clarification

The historical anchor parquet in [data/BTCUSDT/data](/home/user/mcs/data/BTCUSDT/data)
already contains an older opaque block of `cg_*`, `macro_*`, `cal_*`, and
other engineered columns.

For honest attribution, this implementation does **not** use that full frame as
the baseline branch.

Instead it creates two clean branches:

1. `baseline_market_only_v1`
   - only market-native columns:
     `timestamp`, `open`, `high`, `low`, `close`, `volume`, `turnover`
2. `coinglass_slow_v1`
   - the same market-native anchor
   - plus an explicit, compact CoinGlass slow-context layer

This keeps the A/B honest and prevents accidental attribution to unrelated
legacy engineered features.

## Implemented Paths

Generated parquet branches:

1. [baseline_market_only_v1](/home/user/mcs/data/generated/BTCUSDT/baseline_market_only_v1/data)
2. [coinglass_slow_v1](/home/user/mcs/data/generated/BTCUSDT/coinglass_slow_v1/data)

Export targets:

1. [BTCUSDT_parquet_neurobars_baseline_market_only_v1.npz](/home/user/mcs/data/BTCUSDT_parquet_neurobars_baseline_market_only_v1.npz)
2. [BTCUSDT_parquet_neurobars_coinglass_slow_v1.npz](/home/user/mcs/data/BTCUSDT_parquet_neurobars_coinglass_slow_v1.npz)

Branch-specific autoresearch caches:

1. `/home/user/mcs/cache/neurobars_autoresearch/baseline_market_only_v1`
2. `/home/user/mcs/cache/neurobars_autoresearch/coinglass_slow_v1`

## Source Modes

### 1. `legacy`

Default mode for reproducible historical A/B.

This mode builds the new compact slow-context frame from legacy `cg_*` columns
already embedded in the anchor parquet, then re-materializes them into the new
explicit contract:

1. derivatives family
2. sentiment family
3. ETF family
4. `age_minutes`
5. `is_stale`

This lets the branch be tested historically right now without depending on live
API fetching or plan-retention limits.

### 2. `live`

The new API-backed path.

This mode uses `COINGLASS_API_KEY` and the new client in
[umc_nn/multistream/coinglass_client.py](/home/user/mcs/umc_nn/multistream/coinglass_client.py)
to fetch raw CoinGlass payloads into disk cache and normalize them into the
same slow-context contract.

`live` mode was implemented but not exercised in this commit because the key is
kept out of repo state and the first acceptance target was the historical A/B
path.

## Implemented Feature Families

`coinglass_slow_v1` currently carries:

1. derivatives
   - `cg_derivatives_oi_level`
   - `cg_derivatives_oi_delta`
   - `cg_derivatives_funding_last`
   - `cg_derivatives_funding_rolling_mean`
   - `cg_derivatives_long_liquidation_usd`
   - `cg_derivatives_short_liquidation_usd`
   - `cg_derivatives_long_short_ratio`
   - `cg_derivatives_liquidation_imbalance`
2. sentiment
   - `cg_sentiment_fear_greed_level`
   - `cg_sentiment_fear_greed_delta`
   - `cg_sentiment_stablecoin_market_cap_level`
   - `cg_sentiment_stablecoin_market_cap_delta`
3. ETF
   - `cg_etf_btc_net_flow`
   - `cg_etf_btc_net_flow_rolling_5d`
4. freshness
   - `cg_derivatives_age_minutes`
   - `cg_derivatives_is_stale`
   - `cg_sentiment_age_minutes`
   - `cg_sentiment_is_stale`
   - `cg_etf_age_minutes`
   - `cg_etf_is_stale`

## Timing Rules

The implemented v1 defaults are conservative and all-time friendly:

1. slow-context interval defaults to `1d`
2. joins are `latest available_at <= anchor timestamp`
3. no interpolation
4. stale threshold defaults to `36h`
5. missing family values become unavailable/stale rather than leaking future rows

This is intentionally slower and cleaner than a broad intraday context layer.

## Main Entrypoints

Dataset builder:

```bash
.venv/bin/python scripts/build_coinglass_slow_context_v1.py --source-mode legacy
```

Full A/B ladder:

```bash
.venv/bin/python scripts/run_coinglass_slow_v1_ab.py --stage all --branch both
```

## Acceptance Already Confirmed

Confirmed in this implementation:

1. new multistream package compiles
2. unit tests for client, feature builders, join logic, and dataset builder pass
3. real build of both parquet branches from the full historical tree succeeds
4. `prepare.py` accepts both generated branches and creates separate caches

Not yet confirmed in this implementation:

1. full train/export on both branches
2. walk-forward probe comparison
3. rolling portfolio gate comparison
4. live API-backed historical build with the current CoinGlass plan
