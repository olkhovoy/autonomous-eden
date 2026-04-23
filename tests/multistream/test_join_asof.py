from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from umc_nn.multistream.join_asof import build_merged_anchor_frame


def _ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def test_join_asof_never_leaks_future_values_and_marks_missing_as_stale():
    anchor = pl.DataFrame(
        {
            "timestamp": [
                _ts("2025-05-01 00:00:00"),
                _ts("2025-05-01 12:00:00"),
                _ts("2025-05-02 12:00:00"),
            ],
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    derivatives = pl.DataFrame(
        {
            "observed_at": [int(_ts("2025-05-01 06:00:00").timestamp())],
            "published_at": [int(_ts("2025-05-01 06:00:00").timestamp())],
            "available_at": [int(_ts("2025-05-01 06:00:00").timestamp())],
            "revision": ["v1"],
            "cg_derivatives_oi_level": [100.0],
        }
    )
    merged = build_merged_anchor_frame(anchor, {"derivatives": derivatives}, stale_after_minutes=36 * 60)
    rows = merged.select(
        "cg_derivatives_oi_level",
        "cg_derivatives_age_minutes",
        "cg_derivatives_is_stale",
    ).to_dicts()

    assert rows[0]["cg_derivatives_oi_level"] is None
    assert rows[0]["cg_derivatives_age_minutes"] == -1.0
    assert rows[0]["cg_derivatives_is_stale"] == 1.0
    assert rows[1]["cg_derivatives_oi_level"] == 100.0
    assert rows[1]["cg_derivatives_is_stale"] == 0.0


def test_join_asof_weekend_daily_context_freezes_and_turns_stale():
    anchor = pl.DataFrame(
        {
            "timestamp": [
                _ts("2025-05-03 12:00:00"),
                _ts("2025-05-05 12:00:00"),
                _ts("2025-05-05 22:00:00"),
            ],
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    etf = pl.DataFrame(
        {
            "observed_at": [int(_ts("2025-05-02 00:00:00").timestamp()), int(_ts("2025-05-05 00:00:00").timestamp())],
            "published_at": [int(_ts("2025-05-02 00:00:00").timestamp()), int(_ts("2025-05-05 00:00:00").timestamp())],
            "available_at": [int(_ts("2025-05-02 21:00:00").timestamp()), int(_ts("2025-05-05 21:00:00").timestamp())],
            "revision": ["v1", "v1"],
            "cg_etf_btc_net_flow": [10.0, 20.0],
        }
    )
    merged = build_merged_anchor_frame(anchor, {"etf": etf}, stale_after_minutes=36 * 60)
    rows = merged.select("cg_etf_btc_net_flow", "cg_etf_age_minutes", "cg_etf_is_stale").to_dicts()

    assert rows[0]["cg_etf_btc_net_flow"] == 10.0
    assert rows[0]["cg_etf_is_stale"] == 0.0
    assert rows[1]["cg_etf_btc_net_flow"] == 10.0
    assert rows[1]["cg_etf_is_stale"] == 1.0
    assert rows[2]["cg_etf_btc_net_flow"] == 20.0
    assert rows[2]["cg_etf_is_stale"] == 0.0
