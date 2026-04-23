from __future__ import annotations

from datetime import datetime, timezone
from statistics import median
from typing import Any

from .registry import CandidateRegistry
from .schema import CandidateFarmReport, FarmDashboardFeed, FarmProgressEvent, FarmProgressLog, utc_now_text


def _parse_utc(text: str | None) -> datetime | None:
    if not text:
        return None
    for parser in (
        lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")),
        lambda value: datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc),
    ):
        try:
            return parser(text)
        except ValueError:
            continue
    return None


def _normalized_curve(history: list[float]) -> list[float]:
    if not history:
        return []
    base = float(history[0]) if history[0] else 1.0
    if base == 0.0:
        base = 1.0
    return [float(value) / base for value in history]


def _curve_payload(registry: CandidateRegistry, scenario_row: dict[str, Any]) -> dict[str, Any] | None:
    ledger_name = scenario_row.get("portfolio_ledger_report_name")
    if ledger_name and registry.portfolio_ledger_path(ledger_name).exists():
        ledger = registry.load_portfolio_ledger(ledger_name)
        return {
            "sample_indices": list(ledger.ledger_cycle_indices),
            "normalized_balance_history": _normalized_curve(list(ledger.ledger_balance_history)),
            "final_balance": float(ledger.final_balance),
            "total_pnl": float(ledger.total_pnl),
            "max_drawdown_pct": float(ledger.max_drawdown_pct),
        }
    rolling_name = scenario_row.get("rolling_report_name")
    if rolling_name and registry.rolling_path(rolling_name).exists():
        rolling = registry.load_rolling_conveyor(rolling_name)
        return {
            "sample_indices": list(rolling.ledger_cycle_indices),
            "normalized_balance_history": _normalized_curve(list(rolling.ledger_balance_history)),
            "final_balance": float(rolling.final_balance),
            "total_pnl": float(rolling.total_pnl),
            "max_drawdown_pct": float(rolling.max_drawdown_pct),
        }
    return None


def _selection_window(config: dict[str, Any]) -> tuple[str | None, int | None, int | None]:
    rolling_args = dict(config.get("rolling_args", {}))
    return (
        rolling_args.get("selection_start_utc"),
        None if rolling_args.get("selection_days") is None else int(rolling_args["selection_days"]),
        None if rolling_args.get("num_cycles") is None else int(rolling_args["num_cycles"]),
    )


def _heartbeat_state(seconds_since_last_event: float | None, median_gap_seconds: float | None) -> str:
    if seconds_since_last_event is None:
        return "unknown"
    if median_gap_seconds is None or median_gap_seconds <= 0.0:
        if seconds_since_last_event <= 120.0:
            return "fresh"
        if seconds_since_last_event <= 900.0:
            return "watch"
        return "stale"
    ratio = seconds_since_last_event / median_gap_seconds
    if ratio <= 1.5:
        return "fresh"
    if ratio <= 4.0:
        return "watch"
    return "stale"


def _stagnation_state(
    *,
    seconds_since_last_completion: float | None,
    seconds_since_last_gate_pass: float | None,
    median_completion_gap_seconds: float | None,
    completion_count: int,
    gate_pass_completion_count: int,
) -> str:
    if completion_count == 0:
        return "pre-first-completion"
    if gate_pass_completion_count == 0:
        if seconds_since_last_completion is not None and seconds_since_last_completion <= 1800.0:
            return "searching"
        return "stagnating"
    threshold = 3600.0
    if median_completion_gap_seconds is not None and median_completion_gap_seconds > 0.0:
        threshold = max(threshold, median_completion_gap_seconds * 4.0)
    if seconds_since_last_gate_pass is not None and seconds_since_last_gate_pass > threshold:
        return "stagnating"
    return "healthy"


def _progress_metrics(progress_log: FarmProgressLog | None) -> dict[str, Any]:
    if progress_log is None or not progress_log.events:
        return {
            "event_count": 0,
            "completion_event_count": 0,
            "gate_pass_completion_count": 0,
            "events_per_hour": None,
            "completion_events_per_hour": None,
            "gate_pass_events_per_hour": None,
            "median_event_gap_seconds": None,
            "median_completion_gap_seconds": None,
            "seconds_since_last_event": None,
            "seconds_since_last_completion": None,
            "seconds_since_last_gate_pass": None,
            "events_last_15m": 0,
            "completion_events_last_15m": 0,
            "gate_pass_events_last_15m": 0,
            "heartbeat_state": "unknown",
            "stagnation_state": "pre-first-completion",
            "recent_events": [],
        }

    ordered_events = sorted(progress_log.events, key=lambda item: item.sequence)
    parsed_times = [_parse_utc(item.created_at_utc) for item in ordered_events]
    valid_times = [item for item in parsed_times if item is not None]
    first_event_at = valid_times[0] if valid_times else None
    latest_event_at = valid_times[-1] if valid_times else None
    elapsed_seconds = (
        max(0.0, (latest_event_at - first_event_at).total_seconds())
        if first_event_at is not None and latest_event_at is not None
        else None
    )

    completion_events = [
        item for item in ordered_events if item.status in {"completed", "reused"} and item.event_kind in {"completed", "reused"}
    ]
    gate_pass_events = [item for item in completion_events if item.gate_pass is True]

    def _deltas(values: list[datetime | None]) -> list[float]:
        pairs = [
            (left, right)
            for left, right in zip(values, values[1:])
            if left is not None and right is not None
        ]
        return [(right - left).total_seconds() for left, right in pairs if (right - left).total_seconds() >= 0.0]

    event_gaps = _deltas(parsed_times)
    completion_gaps = _deltas([_parse_utc(item.created_at_utc) for item in completion_events])
    median_event_gap_seconds = None if not event_gaps else float(median(event_gaps))
    median_completion_gap_seconds = None if not completion_gaps else float(median(completion_gaps))

    now = datetime.now(timezone.utc)
    seconds_since_last_event = (
        None if latest_event_at is None else max(0.0, (now - latest_event_at).total_seconds())
    )
    last_completion_at = _parse_utc(completion_events[-1].created_at_utc) if completion_events else None
    last_gate_pass_at = _parse_utc(gate_pass_events[-1].created_at_utc) if gate_pass_events else None
    seconds_since_last_completion = (
        None if last_completion_at is None else max(0.0, (now - last_completion_at).total_seconds())
    )
    seconds_since_last_gate_pass = (
        None if last_gate_pass_at is None else max(0.0, (now - last_gate_pass_at).total_seconds())
    )

    def _rate(count: int) -> float | None:
        if elapsed_seconds is None or elapsed_seconds <= 0.0:
            return None
        return float(count) / (elapsed_seconds / 3600.0)

    def _recent_count(items: list[FarmProgressEvent], seconds: float) -> int:
        if latest_event_at is None:
            return 0
        total = 0
        for item in items:
            item_at = _parse_utc(item.created_at_utc)
            if item_at is None:
                continue
            if (latest_event_at - item_at).total_seconds() <= seconds:
                total += 1
        return total

    recent_events = [
        {
            "sequence": item.sequence,
            "created_at_utc": item.created_at_utc,
            "scenario_name": item.scenario_name,
            "status": item.status,
            "progress_stage": item.progress_stage,
            "event_kind": item.event_kind,
            "gate_pass": item.gate_pass,
            "total_pnl": item.total_pnl,
            "note": item.note,
        }
        for item in ordered_events[-12:]
    ]

    return {
        "event_count": len(ordered_events),
        "completion_event_count": len(completion_events),
        "gate_pass_completion_count": len(gate_pass_events),
        "last_event_at": None if latest_event_at is None else latest_event_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_completion_at": None if last_completion_at is None else last_completion_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_gate_pass_at": None if last_gate_pass_at is None else last_gate_pass_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "events_per_hour": _rate(len(ordered_events)),
        "completion_events_per_hour": _rate(len(completion_events)),
        "gate_pass_events_per_hour": _rate(len(gate_pass_events)),
        "median_event_gap_seconds": median_event_gap_seconds,
        "median_completion_gap_seconds": median_completion_gap_seconds,
        "seconds_since_last_event": seconds_since_last_event,
        "seconds_since_last_completion": seconds_since_last_completion,
        "seconds_since_last_gate_pass": seconds_since_last_gate_pass,
        "events_last_15m": _recent_count(ordered_events, 900.0),
        "completion_events_last_15m": _recent_count(completion_events, 900.0),
        "gate_pass_events_last_15m": _recent_count(gate_pass_events, 900.0),
        "heartbeat_state": _heartbeat_state(seconds_since_last_event, median_event_gap_seconds),
        "stagnation_state": _stagnation_state(
            seconds_since_last_completion=seconds_since_last_completion,
            seconds_since_last_gate_pass=seconds_since_last_gate_pass,
            median_completion_gap_seconds=median_completion_gap_seconds,
            completion_count=len(completion_events),
            gate_pass_completion_count=len(gate_pass_events),
        ),
        "recent_events": recent_events,
    }


def build_farm_dashboard_feed(
    registry: CandidateRegistry,
    name: str,
    *,
    farm_report: CandidateFarmReport,
    max_scenarios: int | None = 500,
    max_broom_lines: int | None = 240,
    notes: str | None = None,
) -> FarmDashboardFeed:
    progress_log = None
    if registry.farm_progress_path(farm_report.name).exists():
        progress_log = registry.load_farm_progress_log(farm_report.name)

    scenario_rows: list[dict[str, Any]] = []
    for scenario in farm_report.scenarios:
        selection_start_utc, selection_days, num_cycles = _selection_window(scenario.config)
        scenario_rows.append(
            {
                "scenario_name": scenario.scenario_name,
                "status": scenario.status,
                "progress_stage": scenario.progress_stage,
                "updated_at_utc": scenario.updated_at_utc,
                "mode": scenario.mode,
                "selection_start_utc": selection_start_utc,
                "selection_days": selection_days,
                "num_cycles": num_cycles,
                "rolling_report_name": scenario.rolling_report_name,
                "lifecycle_report_name": scenario.lifecycle_report_name,
                "portfolio_ledger_report_name": scenario.portfolio_ledger_report_name,
                "portfolio_baselines_report_name": scenario.portfolio_baselines_report_name,
                "output_dir": scenario.output_dir,
                "candidate_pool_count": len(scenario.candidate_pool_ids),
                "selected_candidate_count": len(scenario.selected_candidate_ids),
                "final_status_counts": dict(scenario.final_status_counts),
                "gate_pass": scenario.gate_pass,
                "beaten_baselines": list(scenario.beaten_baselines),
                "failed_required_baselines": list(scenario.failed_required_baselines),
                "total_pnl": scenario.total_pnl,
                "total_return_pct": scenario.total_return_pct,
                "max_drawdown_pct": scenario.max_drawdown_pct,
                "evaluated_cycle_count": scenario.evaluated_cycle_count,
                "positive_cycle_count": scenario.positive_cycle_count,
                "error_message": scenario.error_message,
                "log_paths": dict(scenario.log_paths),
            }
        )

    def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        updated = _parse_utc(row.get("updated_at_utc"))
        status_rank = {
            "running": 0,
            "completed": 1,
            "reused": 2,
            "planned": 3,
            "failed": 4,
        }.get(str(row.get("status")), 5)
        return (
            status_rank,
            0 if row.get("gate_pass") else 1,
            -(float(row["total_pnl"]) if row.get("total_pnl") is not None else float("-inf")),
            0 if updated is None else -updated.timestamp(),
        )

    scenario_rows.sort(key=_sort_key)
    all_scenario_rows = list(scenario_rows)
    if max_scenarios is not None:
        scenario_rows = scenario_rows[:max_scenarios]

    completed_rows = [row for row in all_scenario_rows if row["status"] in {"completed", "reused"}]
    running_rows = [row for row in all_scenario_rows if row["status"] == "running"]
    failed_rows = [row for row in all_scenario_rows if row["status"] == "failed"]
    planned_rows = [row for row in all_scenario_rows if row["status"] == "planned"]
    gate_rows = [row for row in completed_rows if row.get("gate_pass") is True]

    broom_lines: list[dict[str, Any]] = []
    source_reports: list[str] = []
    ranked_for_broom = sorted(
        completed_rows,
        key=lambda row: (
            row.get("gate_pass") is True,
            float(row["total_pnl"]) if row.get("total_pnl") is not None else float("-inf"),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked_for_broom[: max_broom_lines or len(ranked_for_broom)]):
        curve = _curve_payload(registry, row)
        if curve is None:
            continue
        source_reports.append(
            row.get("portfolio_ledger_report_name")
            or row.get("rolling_report_name")
            or row["scenario_name"]
        )
        brightness = max(0.16, 1.0 - (rank * 0.12))
        broom_lines.append(
            {
                "scenario_name": row["scenario_name"],
                "status": row["status"],
                "mode": row["mode"],
                "gate_pass": row.get("gate_pass"),
                "progress_stage": row.get("progress_stage"),
                "brightness_hint": brightness,
                "sample_indices": curve["sample_indices"],
                "normalized_balance_history": curve["normalized_balance_history"],
                "final_balance": curve["final_balance"],
                "total_pnl": curve["total_pnl"],
                "max_drawdown_pct": curve["max_drawdown_pct"],
            }
        )

    status_counts: dict[str, int] = {}
    progress_stage_counts: dict[str, int] = {}
    for row in all_scenario_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        stage = row.get("progress_stage") or "none"
        progress_stage_counts[stage] = progress_stage_counts.get(stage, 0) + 1

    monitoring = {
        "latest_updated_at": max(
            (row.get("updated_at_utc") for row in all_scenario_rows if row.get("updated_at_utc")),
            default=None,
        ),
        "running_scenarios": [row["scenario_name"] for row in running_rows],
        "recent_gate_pass_scenarios": [row["scenario_name"] for row in gate_rows[:8]],
        "recent_failed_scenarios": [row["scenario_name"] for row in failed_rows[:8]],
        "planned_scenarios": [row["scenario_name"] for row in planned_rows[:8]],
        "progress_stage_counts": progress_stage_counts,
        **_progress_metrics(progress_log),
    }

    summary = {
        **dict(farm_report.summary),
        "status_counts": status_counts,
        "running_scenarios": len(running_rows),
        "completed_or_reused_scenarios": len(completed_rows),
        "failed_scenarios": len(failed_rows),
        "planned_scenarios": len(planned_rows),
    }

    broom = None
    if broom_lines:
        broom = {
            "source_report": farm_report.name,
            "source_reports": source_reports,
            "line_count": len(broom_lines),
            "total_line_count": len(completed_rows),
            "lines": broom_lines,
        }

    return FarmDashboardFeed(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_farm_report=farm_report.name,
        summary=summary,
        scenarios=scenario_rows,
        monitoring=monitoring,
        broom=broom,
        notes=notes,
    )
