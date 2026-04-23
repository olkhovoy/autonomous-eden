from __future__ import annotations

from typing import Iterable

import polars as pl


ANCHOR_MARKET_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
)


def extract_market_only_frame(frame: pl.DataFrame, *, keep_columns: Iterable[str] = ANCHOR_MARKET_COLUMNS) -> pl.DataFrame:
    available = [column for column in keep_columns if column in frame.columns]
    if "timestamp" not in available:
        raise ValueError("Anchor frame must contain timestamp")
    return frame.select(available).sort("timestamp")


def _join_one_family(
    anchor: pl.DataFrame,
    feature_frame: pl.DataFrame,
    *,
    family_name: str,
    stale_after_minutes: int,
) -> pl.DataFrame:
    age_column = f"cg_{family_name}_age_minutes"
    stale_column = f"cg_{family_name}_is_stale"

    if feature_frame.is_empty():
        feature_columns = [
            column
            for column in feature_frame.columns
            if column not in {"observed_at", "published_at", "available_at", "revision", "source_id", "market_scope"}
        ]
        missing_exprs = [pl.lit(None).alias(column) for column in feature_columns]
        return anchor.with_columns(
            [
                *missing_exprs,
                pl.lit(-1.0).alias(age_column),
                pl.lit(1.0).alias(stale_column),
            ]
        )

    features = feature_frame.sort("available_at").rename({"available_at": f"{family_name}_available_at"})
    joined = anchor.join_asof(
        features,
        left_on="timestamp_s",
        right_on=f"{family_name}_available_at",
        strategy="backward",
    )
    age_expr = (
        (pl.col("timestamp_s") - pl.col(f"{family_name}_available_at")) / 60.0
    ).fill_null(-1.0)
    joined = joined.with_columns(
        [
            age_expr.alias(age_column),
            (
                pl.when(pl.col(f"{family_name}_available_at").is_null())
                .then(1.0)
                .when(age_expr > float(stale_after_minutes))
                .then(1.0)
                .otherwise(0.0)
            ).alias(stale_column),
        ]
    )
    drop_cols = [column for column in ("source_id", "market_scope", "published_at", "revision", f"{family_name}_available_at") if column in joined.columns]
    if "observed_at" in joined.columns:
        drop_cols.append("observed_at")
    return joined.drop(drop_cols)


def build_merged_anchor_frame(
    anchor_frame: pl.DataFrame,
    feature_frames: dict[str, pl.DataFrame],
    *,
    stale_after_minutes: int,
) -> pl.DataFrame:
    if "timestamp" not in anchor_frame.columns:
        raise ValueError("anchor_frame must contain timestamp")
    frame = anchor_frame.sort("timestamp")
    if "timestamp_s" not in frame.columns:
        frame = frame.with_columns(pl.col("timestamp").dt.epoch("s").cast(pl.Int64).alias("timestamp_s"))

    for family_name, feature_frame in feature_frames.items():
        frame = _join_one_family(
            frame,
            feature_frame,
            family_name=family_name,
            stale_after_minutes=stale_after_minutes,
        )
    return frame.sort("timestamp").drop("timestamp_s")
