from __future__ import annotations

from argparse import Namespace

from scripts.run_continuous_search_cycle import _generator_window_plan, _selection_span_days


def test_selection_span_days_requires_positive_whole_days() -> None:
    assert _selection_span_days("2025-05-08 00:00:00", "2025-05-15 00:00:00") == 7


def test_generator_window_plan_selection_anchored_derives_adjacent_train_window() -> None:
    args = Namespace(
        window_start=None,
        generator_window_mode="selection_anchored",
        generator_window_offset_days=None,
        selection_start_utc="2025-05-08 00:00:00",
        selection_end_utc="2025-05-15 00:00:00",
        train_days=7,
        oos_days=99,
    )

    starts, oos_days = _generator_window_plan(args)

    assert starts == ["2025-05-01 00:00:00"]
    assert oos_days == 7


def test_generator_window_plan_supports_multiple_negative_offsets() -> None:
    args = Namespace(
        window_start=None,
        generator_window_mode="selection_anchored",
        generator_window_offset_days=[0, -7, -14],
        selection_start_utc="2025-05-08 00:00:00",
        selection_end_utc="2025-05-15 00:00:00",
        train_days=7,
        oos_days=7,
    )

    starts, oos_days = _generator_window_plan(args)

    assert starts == [
        "2025-04-17 00:00:00",
        "2025-04-24 00:00:00",
        "2025-05-01 00:00:00",
    ]
    assert oos_days == 7


def test_generator_window_plan_explicit_mode_respects_explicit_starts() -> None:
    args = Namespace(
        window_start=["2024-01-01 00:00:00", "2024-06-01 00:00:00"],
        generator_window_mode="explicit",
        generator_window_offset_days=None,
        selection_start_utc="2025-05-08 00:00:00",
        selection_end_utc="2025-05-15 00:00:00",
        train_days=7,
        oos_days=5,
    )

    starts, oos_days = _generator_window_plan(args)

    assert starts == ["2024-01-01 00:00:00", "2024-06-01 00:00:00"]
    assert oos_days == 5
