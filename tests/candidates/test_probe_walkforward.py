from __future__ import annotations

import pytest

from scripts.probe_monolith_walkforward import _window_quads


def test_window_quads_split_train_valid_oos() -> None:
    windows = _window_quads(["2025-05-01 00:00:00"], train_days=7, valid_days=2, oos_days=7)
    assert windows == [
        (
            "wf_01_20250501",
            "2025-05-01 00:00:00",
            "2025-05-06 00:00:00",
            "2025-05-08 00:00:00",
            "2025-05-15 00:00:00",
        )
    ]


def test_window_quads_reject_invalid_valid_days() -> None:
    with pytest.raises(ValueError):
        _window_quads(["2025-05-01 00:00:00"], train_days=7, valid_days=7, oos_days=7)
