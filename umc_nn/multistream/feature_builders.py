from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import polars as pl


SLOW_CONTEXT_DAILY_STALE_MINUTES = 36 * 60
DERIVATIVE_COLUMNS = {
    "cg_derivatives_oi_level": pl.Float64,
    "cg_derivatives_oi_delta": pl.Float64,
    "cg_derivatives_funding_last": pl.Float64,
    "cg_derivatives_funding_rolling_mean": pl.Float64,
    "cg_derivatives_long_liquidation_usd": pl.Float64,
    "cg_derivatives_short_liquidation_usd": pl.Float64,
    "cg_derivatives_long_short_ratio": pl.Float64,
    "cg_derivatives_liquidation_imbalance": pl.Float64,
}
SENTIMENT_COLUMNS = {
    "cg_sentiment_fear_greed_level": pl.Float64,
    "cg_sentiment_fear_greed_delta": pl.Float64,
    "cg_sentiment_stablecoin_market_cap_level": pl.Float64,
    "cg_sentiment_stablecoin_market_cap_delta": pl.Float64,
}
ETF_COLUMNS = {
    "cg_etf_btc_net_flow": pl.Float64,
    "cg_etf_btc_net_flow_rolling_5d": pl.Float64,
}


@dataclass(frozen=True)
class CoinGlassSlowContextConfig:
    symbol: str = "BTC"
    pair_symbol: str = "BTCUSDT"
    exchange: str = "Bybit"
    liquidation_exchange_list: str = "Bybit"
    interval: str = "1d"
    derivatives_available_lag_minutes: int = 0
    sentiment_available_lag_minutes: int = 0
    etf_available_lag_minutes: int = 60
    stale_after_minutes: int = SLOW_CONTEXT_DAILY_STALE_MINUTES
    revision: str = "coinglass_slow_v1"


def _unwrap_payload(payload: dict[str, Any]) -> Any:
    return payload.get("payload", payload)


def _payload_data(payload: dict[str, Any]) -> Any:
    inner = _unwrap_payload(payload)
    if isinstance(inner, dict):
        for key in ("data", "result", "response", "payload"):
            if key in inner:
                return inner[key]
    return inner


def _as_epoch_seconds(value: Any) -> int:
    if value is None:
        raise ValueError("timestamp value is missing")
    if isinstance(value, (int, float)):
        value = int(value)
        if value > 10_000_000_000:
            return value // 1000
        return value
    text = str(value).strip()
    if text.isdigit():
        parsed = int(text)
        return parsed // 1000 if parsed > 10_000_000_000 else parsed
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return int(datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _extract_series_rows(payload: dict[str, Any], *, value_keys: Iterable[str]) -> list[dict[str, Any]]:
    data = _payload_data(payload)
    if isinstance(data, list):
        rows: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                rows.append(item)
        return rows

    if isinstance(data, dict):
        if "data_list" in data and "time_list" in data:
            time_list = data["time_list"]
            data_list = data["data_list"]
            if len(time_list) != len(data_list):
                raise ValueError("CoinGlass payload has mismatched time_list/data_list lengths")
            return [{"time": ts, value_keys.__iter__().__next__(): value} for ts, value in zip(time_list, data_list)]

        for key in ("list", "rows", "items", "history"):
            if key in data and isinstance(data[key], list):
                return [item for item in data[key] if isinstance(item, dict)]

    raise ValueError("Unsupported CoinGlass payload shape")


def _float_or_none(item: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in item and item[key] is not None:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                continue
    return None


def _float_or_sum_mapping(item: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in item or item[key] is None:
            continue
        value = item[key]
        if isinstance(value, dict):
            total = 0.0
            found = False
            for nested in value.values():
                try:
                    total += float(nested)
                    found = True
                except (TypeError, ValueError):
                    continue
            if found:
                return total
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _time_or_none(item: dict[str, Any]) -> int | None:
    for key in ("time", "timestamp", "date", "t", "timeStamp"):
        if key in item and item[key] is not None:
            return _as_epoch_seconds(item[key])
    return None


def _normalize_frame(rows: list[dict[str, Any]], *, available_lag_minutes: int, revision: str) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {
                "observed_at": pl.Series([], dtype=pl.Int64),
                "published_at": pl.Series([], dtype=pl.Int64),
                "available_at": pl.Series([], dtype=pl.Int64),
                "revision": pl.Series([], dtype=pl.Utf8),
            }
        )
    frame = pl.DataFrame(rows, infer_schema_length=max(len(rows), 1))
    lag_seconds = available_lag_minutes * 60
    frame = frame.sort("observed_at").with_columns(
        [
            pl.col("observed_at").cast(pl.Int64),
            pl.col("observed_at").cast(pl.Int64).alias("published_at"),
            (pl.col("observed_at").cast(pl.Int64) + lag_seconds).alias("available_at"),
            pl.lit(revision).alias("revision"),
        ]
    )
    return frame


def _ensure_columns(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    missing = [
        pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in schema.items()
        if name not in frame.columns
    ]
    if missing:
        frame = frame.with_columns(missing)
    ordered = ["observed_at", "published_at", "available_at", "revision", *schema.keys()]
    present = [column for column in ordered if column in frame.columns]
    return frame.select(present)


def build_daily_derivatives_context(
    open_interest_payload: dict[str, Any] | None,
    funding_payload: dict[str, Any] | None,
    liquidation_payload: dict[str, Any] | None,
    long_short_ratio_payload: dict[str, Any] | None,
    *,
    config: CoinGlassSlowContextConfig,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []

    if open_interest_payload is not None:
        rows = []
        for item in _extract_series_rows(open_interest_payload, value_keys=("value",)):
            observed_at = _time_or_none(item)
            if observed_at is None:
                continue
            oi_level = _float_or_none(item, "close", "value", "openInterest", "sumOpenInterest")
            if oi_level is None:
                continue
            rows.append(
                {
                    "observed_at": observed_at,
                    "cg_derivatives_oi_level": oi_level,
                }
            )
        frame = _normalize_frame(rows, available_lag_minutes=config.derivatives_available_lag_minutes, revision=config.revision)
        if not frame.is_empty():
            frame = frame.with_columns(
                pl.col("cg_derivatives_oi_level").diff().fill_null(0.0).alias("cg_derivatives_oi_delta")
            )
        frames.append(frame)

    if funding_payload is not None:
        rows = []
        for item in _extract_series_rows(funding_payload, value_keys=("value",)):
            observed_at = _time_or_none(item)
            if observed_at is None:
                continue
            funding_last = _float_or_none(item, "close", "value", "fundingRate", "funding_rate")
            if funding_last is None:
                continue
            rows.append({"observed_at": observed_at, "cg_derivatives_funding_last": funding_last})
        frame = _normalize_frame(rows, available_lag_minutes=config.derivatives_available_lag_minutes, revision=config.revision)
        if not frame.is_empty():
            frame = frame.with_columns(
                pl.col("cg_derivatives_funding_last")
                .rolling_mean(window_size=5, min_samples=1)
                .alias("cg_derivatives_funding_rolling_mean")
            )
        frames.append(frame)

    if liquidation_payload is not None:
        rows = []
        for item in _extract_series_rows(liquidation_payload, value_keys=("value",)):
            observed_at = _time_or_none(item)
            if observed_at is None:
                continue
            long_liq = _float_or_none(
                item,
                "longLiquidationUsd",
                "long",
                "longs",
                "longVolUsd",
                "longsUsd",
                "aggregated_long_liquidation_usd",
            )
            short_liq = _float_or_none(
                item,
                "shortLiquidationUsd",
                "short",
                "shorts",
                "shortVolUsd",
                "shortsUsd",
                "aggregated_short_liquidation_usd",
            )
            if long_liq is None and short_liq is None:
                continue
            long_liq = 0.0 if long_liq is None else long_liq
            short_liq = 0.0 if short_liq is None else short_liq
            denom = max(long_liq + short_liq, 1e-8)
            rows.append(
                {
                    "observed_at": observed_at,
                    "cg_derivatives_long_liquidation_usd": long_liq,
                    "cg_derivatives_short_liquidation_usd": short_liq,
                    "cg_derivatives_liquidation_imbalance": (long_liq - short_liq) / denom,
                }
            )
        frames.append(
            _normalize_frame(rows, available_lag_minutes=config.derivatives_available_lag_minutes, revision=config.revision)
        )

    if long_short_ratio_payload is not None:
        rows = []
        for item in _extract_series_rows(long_short_ratio_payload, value_keys=("value",)):
            observed_at = _time_or_none(item)
            if observed_at is None:
                continue
            ratio = _float_or_none(
                item,
                "longShortRadio",
                "longShortRatio",
                "global_account_long_short_ratio",
                "value",
                "close",
            )
            if ratio is None:
                continue
            rows.append({"observed_at": observed_at, "cg_derivatives_long_short_ratio": ratio})
        frames.append(
            _normalize_frame(rows, available_lag_minutes=config.derivatives_available_lag_minutes, revision=config.revision)
        )

    if not frames:
        return _ensure_columns(
            _normalize_frame([], available_lag_minutes=config.derivatives_available_lag_minutes, revision=config.revision),
            DERIVATIVE_COLUMNS,
        )

    merged = frames[0]
    for frame in frames[1:]:
        if frame.is_empty():
            continue
        merged = merged.join(frame, on=["observed_at", "published_at", "available_at", "revision"], how="full", coalesce=True)
    return _ensure_columns(merged.sort("observed_at"), DERIVATIVE_COLUMNS)


def build_daily_sentiment_context(
    fear_greed_payload: dict[str, Any] | None,
    stablecoin_market_cap_payload: dict[str, Any] | None,
    *,
    config: CoinGlassSlowContextConfig,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []

    if fear_greed_payload is not None:
        rows = []
        for item in _extract_series_rows(fear_greed_payload, value_keys=("value",)):
            observed_at = _time_or_none(item)
            if observed_at is None:
                continue
            value = _float_or_none(item, "value", "close")
            if value is None:
                continue
            rows.append({"observed_at": observed_at, "cg_sentiment_fear_greed_level": value})
        frame = _normalize_frame(rows, available_lag_minutes=config.sentiment_available_lag_minutes, revision=config.revision)
        if not frame.is_empty():
            frame = frame.with_columns(
                pl.col("cg_sentiment_fear_greed_level").diff().fill_null(0.0).alias("cg_sentiment_fear_greed_delta")
            )
        frames.append(frame)

    if stablecoin_market_cap_payload is not None:
        rows = []
        for item in _extract_series_rows(stablecoin_market_cap_payload, value_keys=("value",)):
            observed_at = _time_or_none(item)
            if observed_at is None:
                continue
            value = _float_or_sum_mapping(item, "marketCap", "marketcap", "value", "close")
            if value is None:
                continue
            rows.append({"observed_at": observed_at, "cg_sentiment_stablecoin_market_cap_level": value})
        frame = _normalize_frame(rows, available_lag_minutes=config.sentiment_available_lag_minutes, revision=config.revision)
        if not frame.is_empty():
            frame = frame.with_columns(
                pl.col("cg_sentiment_stablecoin_market_cap_level")
                .diff()
                .fill_null(0.0)
                .alias("cg_sentiment_stablecoin_market_cap_delta")
            )
        frames.append(frame)

    if not frames:
        return _ensure_columns(
            _normalize_frame([], available_lag_minutes=config.sentiment_available_lag_minutes, revision=config.revision),
            SENTIMENT_COLUMNS,
        )

    merged = frames[0]
    for frame in frames[1:]:
        if frame.is_empty():
            continue
        merged = merged.join(frame, on=["observed_at", "published_at", "available_at", "revision"], how="full", coalesce=True)
    return _ensure_columns(merged.sort("observed_at"), SENTIMENT_COLUMNS)


def build_daily_etf_context(etf_payload: dict[str, Any] | None, *, config: CoinGlassSlowContextConfig) -> pl.DataFrame:
    if etf_payload is None:
        return _ensure_columns(
            _normalize_frame([], available_lag_minutes=config.etf_available_lag_minutes, revision=config.revision),
            ETF_COLUMNS,
        )

    rows = []
    for item in _extract_series_rows(etf_payload, value_keys=("value",)):
        observed_at = _time_or_none(item)
        if observed_at is None:
            continue
        net_flow = _float_or_none(item, "changeUsd", "flow_usd", "netFlow", "net_flow", "value")
        if net_flow is None:
            continue
        rows.append({"observed_at": observed_at, "cg_etf_btc_net_flow": net_flow})

    frame = _normalize_frame(rows, available_lag_minutes=config.etf_available_lag_minutes, revision=config.revision)
    if not frame.is_empty():
        frame = frame.with_columns(
            pl.col("cg_etf_btc_net_flow")
            .rolling_sum(window_size=5, min_samples=1)
            .alias("cg_etf_btc_net_flow_rolling_5d")
        )
    return _ensure_columns(frame, ETF_COLUMNS)


def build_coinglass_slow_context_frame(
    *,
    open_interest_payload: dict[str, Any] | None,
    funding_payload: dict[str, Any] | None,
    liquidation_payload: dict[str, Any] | None,
    long_short_ratio_payload: dict[str, Any] | None,
    fear_greed_payload: dict[str, Any] | None,
    stablecoin_market_cap_payload: dict[str, Any] | None,
    etf_payload: dict[str, Any] | None,
    config: CoinGlassSlowContextConfig,
) -> dict[str, pl.DataFrame]:
    return {
        "derivatives": build_daily_derivatives_context(
            open_interest_payload,
            funding_payload,
            liquidation_payload,
            long_short_ratio_payload,
            config=config,
        ),
        "sentiment": build_daily_sentiment_context(
            fear_greed_payload,
            stablecoin_market_cap_payload,
            config=config,
        ),
        "etf": build_daily_etf_context(etf_payload, config=config),
    }


def build_legacy_coinglass_slow_context_frame(anchor_df: pl.DataFrame, *, config: CoinGlassSlowContextConfig) -> dict[str, pl.DataFrame]:
    if "timestamp" not in anchor_df.columns:
        raise ValueError("anchor_df must contain a timestamp column")
    if anchor_df.schema["timestamp"] != pl.Datetime(time_unit="us", time_zone="UTC"):
        anchor_df = anchor_df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

    base = anchor_df.sort("timestamp").with_columns(
        [
            pl.col("timestamp").dt.truncate("1d").alias("day"),
            pl.col("timestamp").dt.epoch("s").alias("timestamp_s"),
        ]
    )

    frames: dict[str, pl.DataFrame] = {}

    legacy_derivative_cols = [
        "cg_oi_open_interest",
        "cg_oi_open_interest_change",
        "cg_funding_rate",
        "cg_funding_rate_ma",
        "cg_liquidations_long_usd",
        "cg_liquidations_short_usd",
        "cg_ls_ratio_value",
    ]
    available_derivative_cols = [column for column in legacy_derivative_cols if column in base.columns]
    derivative = (
        base.group_by("day")
        .agg([pl.col(column).last() for column in available_derivative_cols])
        .rename(
            {
                "day": "observed_at_day",
                "cg_oi_open_interest": "cg_derivatives_oi_level",
                "cg_oi_open_interest_change": "cg_derivatives_oi_delta",
                "cg_funding_rate": "cg_derivatives_funding_last",
                "cg_funding_rate_ma": "cg_derivatives_funding_rolling_mean",
                "cg_liquidations_long_usd": "cg_derivatives_long_liquidation_usd",
                "cg_liquidations_short_usd": "cg_derivatives_short_liquidation_usd",
                "cg_ls_ratio_value": "cg_derivatives_long_short_ratio",
            }
        )
        .with_columns(
            [
                pl.col("observed_at_day").dt.epoch("s").cast(pl.Int64).alias("observed_at"),
            ]
        )
        .drop("observed_at_day")
    )
    if not derivative.is_empty():
        if "cg_derivatives_oi_delta" not in derivative.columns and "cg_derivatives_oi_level" in derivative.columns:
            derivative = derivative.with_columns(
                pl.col("cg_derivatives_oi_level").diff().fill_null(0.0).alias("cg_derivatives_oi_delta")
            )
        if "cg_derivatives_funding_rolling_mean" not in derivative.columns and "cg_derivatives_funding_last" in derivative.columns:
            derivative = derivative.with_columns(
                pl.col("cg_derivatives_funding_last")
                .rolling_mean(window_size=5, min_samples=1)
                .alias("cg_derivatives_funding_rolling_mean")
            )
        if "cg_derivatives_long_liquidation_usd" in derivative.columns and "cg_derivatives_short_liquidation_usd" in derivative.columns:
            derivative = derivative.with_columns(
                (
                    (pl.col("cg_derivatives_long_liquidation_usd") - pl.col("cg_derivatives_short_liquidation_usd"))
                    / (pl.col("cg_derivatives_long_liquidation_usd") + pl.col("cg_derivatives_short_liquidation_usd") + 1e-8)
                ).alias("cg_derivatives_liquidation_imbalance")
            )
        derivative = _normalize_frame(
            derivative.to_dicts(),
            available_lag_minutes=config.derivatives_available_lag_minutes,
            revision=config.revision,
        )
    frames["derivatives"] = _ensure_columns(derivative, DERIVATIVE_COLUMNS)

    legacy_sentiment_cols = ["cg_fear_greed_value"]
    available_sentiment_cols = [column for column in legacy_sentiment_cols if column in base.columns]
    sentiment = (
        base.group_by("day")
        .agg([pl.col(column).last() for column in available_sentiment_cols])
        .rename(
            {
                "day": "observed_at_day",
                "cg_fear_greed_value": "cg_sentiment_fear_greed_level",
            }
        )
        .with_columns(pl.col("observed_at_day").dt.epoch("s").cast(pl.Int64).alias("observed_at"))
        .drop("observed_at_day")
    )
    if not sentiment.is_empty():
        sentiment = sentiment.with_columns(
            pl.col("cg_sentiment_fear_greed_level").diff().fill_null(0.0).alias("cg_sentiment_fear_greed_delta"),
            pl.lit(0.0).alias("cg_sentiment_stablecoin_market_cap_level"),
            pl.lit(0.0).alias("cg_sentiment_stablecoin_market_cap_delta"),
        )
        sentiment = _normalize_frame(
            sentiment.to_dicts(),
            available_lag_minutes=config.sentiment_available_lag_minutes,
            revision=config.revision,
        )
    frames["sentiment"] = _ensure_columns(sentiment, SENTIMENT_COLUMNS)

    legacy_etf_cols = ["cg_etf_flows_net_flow"]
    available_etf_cols = [column for column in legacy_etf_cols if column in base.columns]
    etf = (
        base.group_by("day")
        .agg([pl.col(column).last() for column in available_etf_cols])
        .rename(
            {
                "day": "observed_at_day",
                "cg_etf_flows_net_flow": "cg_etf_btc_net_flow",
            }
        )
        .with_columns(pl.col("observed_at_day").dt.epoch("s").cast(pl.Int64).alias("observed_at"))
        .drop("observed_at_day")
    )
    if not etf.is_empty():
        etf = etf.with_columns(
            pl.col("cg_etf_btc_net_flow")
            .rolling_sum(window_size=5, min_samples=1)
            .alias("cg_etf_btc_net_flow_rolling_5d")
        )
        etf = _normalize_frame(
            etf.to_dicts(),
            available_lag_minutes=config.etf_available_lag_minutes,
            revision=config.revision,
        )
    frames["etf"] = _ensure_columns(etf, ETF_COLUMNS)
    return frames
