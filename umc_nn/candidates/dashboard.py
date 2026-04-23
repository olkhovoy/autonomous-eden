from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .clustering import cluster_id_by_candidate
from .overrides import candidate_override_map, cluster_override_map
from .registry import CandidateRegistry
from .schema import (
    AllocatorScenario,
    AllocatorWorkbenchReport,
    CandidateRecord,
    ClusterReport,
    CombinationScenario,
    CombinationSearchReport,
    DashboardFeed,
    DiversificationPairStats,
    DiversificationReport,
    OverrideSet,
    ShortlistCandidateScore,
    ShortlistReport,
    utc_now_text,
)


def _parse_utc(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _candidate_period_payload(candidate: CandidateRecord) -> dict[str, dict[str, Any]]:
    periods: dict[str, dict[str, Any]] = {}
    for name, stats in candidate.periods.items():
        periods[name] = {
            "pnl": stats.pnl,
            "final_balance": stats.final_balance,
            "max_drawdown_pct": stats.max_drawdown_pct,
            "trades": stats.trades,
            "win_rate_pct": stats.win_rate_pct,
            "full_window": stats.full_window,
            "beats_flat": stats.beats_flat,
            "beats_best_baseline": stats.beats_best_baseline,
            "baseline_winner": stats.baseline_winner,
            "start_utc": stats.start_utc,
            "end_utc": stats.end_utc,
        }
    return periods


def _resampling_payload(candidate: CandidateRecord, preferred_name: str | None) -> dict[str, Any] | None:
    if not candidate.resampling_results:
        return None
    stats = None
    name = preferred_name
    if name and name in candidate.resampling_results:
        stats = candidate.resampling_results[name]
    elif preferred_name is None:
        name = sorted(candidate.resampling_results)[0]
        stats = candidate.resampling_results[name]
    else:
        name = sorted(candidate.resampling_results)[0]
        stats = candidate.resampling_results[name]
    if stats is None:
        return None
    return {
        "name": name,
        "fraction": stats.fraction,
        "pessimistic_net_profit": stats.pessimistic_net_profit,
        "pessimistic_max_drawdown_pct": stats.pessimistic_max_drawdown_pct,
        "p05_net_profit": stats.p05_net_profit,
        "p95_max_drawdown_pct": stats.p95_max_drawdown_pct,
        "loss_rate": stats.loss_rate,
        "profitable_rate": stats.profitable_rate,
        "ruin_rate": stats.ruin_rate,
        "original_net_profit": stats.original_net_profit,
        "original_max_drawdown_pct": stats.original_max_drawdown_pct,
        "iterations": stats.iterations,
    }


def _scenario_weights_payload(weights: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": weight.candidate_id,
            "display_name": weight.display_name,
            "cluster_id": weight.cluster_id,
            "normalized_share": weight.normalized_share,
            "capital_fraction": weight.capital_fraction,
            "capped": weight.capped,
            "cap_reason": weight.cap_reason,
        }
        for weight in weights
    ]


def _allocator_scenario_payload(scenario: AllocatorScenario) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "objective_score": scenario.objective_score,
        "requested_risk_fraction": scenario.requested_risk_fraction,
        "allocated_risk_fraction": scenario.allocated_risk_fraction,
        "reserve_fraction": scenario.reserve_fraction,
        "score_mode": scenario.score_mode,
        "weights": _scenario_weights_payload(scenario.weights),
        "curve_sample_indices": list(scenario.curve_sample_indices),
        "normalized_balance_history": list(scenario.normalized_balance_history),
        "resampling": {
            "p05_net_profit": scenario.resampling.p05_net_profit,
            "p25_net_profit": scenario.resampling.p25_net_profit,
            "median_net_profit": scenario.resampling.median_net_profit,
            "p95_max_drawdown_pct": scenario.resampling.p95_max_drawdown_pct,
            "profitable_rate": scenario.resampling.profitable_rate,
            "loss_rate": scenario.resampling.loss_rate,
            "ruin_rate": scenario.resampling.ruin_rate,
            "original_net_profit": scenario.resampling.original_net_profit,
            "original_max_drawdown_pct": scenario.resampling.original_max_drawdown_pct,
        },
    }


def _combination_scenario_payload(scenario: CombinationScenario) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "subset_candidate_ids": list(scenario.subset_candidate_ids),
        "subset_display_names": list(scenario.subset_display_names),
        "subset_size": scenario.subset_size,
        "objective_score": scenario.objective_score,
        "requested_risk_fraction": scenario.requested_risk_fraction,
        "allocated_risk_fraction": scenario.allocated_risk_fraction,
        "reserve_fraction": scenario.reserve_fraction,
        "score_mode": scenario.score_mode,
        "weights": _scenario_weights_payload(scenario.weights),
        "curve_sample_indices": list(scenario.curve_sample_indices),
        "normalized_balance_history": list(scenario.normalized_balance_history),
        "resampling": {
            "p05_net_profit": scenario.resampling.p05_net_profit,
            "p25_net_profit": scenario.resampling.p25_net_profit,
            "median_net_profit": scenario.resampling.median_net_profit,
            "p95_max_drawdown_pct": scenario.resampling.p95_max_drawdown_pct,
            "profitable_rate": scenario.resampling.profitable_rate,
            "loss_rate": scenario.resampling.loss_rate,
            "ruin_rate": scenario.resampling.ruin_rate,
            "original_net_profit": scenario.resampling.original_net_profit,
            "original_max_drawdown_pct": scenario.resampling.original_max_drawdown_pct,
        },
    }


def _pair_lookup(pair_stats: list[DiversificationPairStats]) -> dict[tuple[str, str], DiversificationPairStats]:
    lookup: dict[tuple[str, str], DiversificationPairStats] = {}
    for pair in pair_stats:
        key = tuple(sorted((pair.left_candidate_id, pair.right_candidate_id)))
        lookup[key] = pair
    return lookup


def _candidate_score_map(shortlist_report: ShortlistReport | None) -> dict[str, ShortlistCandidateScore]:
    if shortlist_report is None:
        return {}
    return {item.candidate_id: item for item in shortlist_report.candidate_scores}


def build_dashboard_feed(
    registry: CandidateRegistry,
    name: str,
    *,
    shortlist_report: ShortlistReport | None = None,
    diversification_report: DiversificationReport | None = None,
    cluster_report: ClusterReport | None = None,
    override_set: OverrideSet | None = None,
    allocator_report: AllocatorWorkbenchReport | None = None,
    combination_report: CombinationSearchReport | None = None,
    resampling_name: str | None = None,
    max_candidate_rows: int | None = 500,
    max_broom_lines: int | None = 300,
    max_allocator_scenarios: int = 6,
    max_combination_scenarios: int = 12,
    max_audit_entries: int = 50,
    notes: str | None = None,
) -> DashboardFeed:
    candidates = list(registry.iter_candidates())
    cluster_by_candidate = {} if cluster_report is None else cluster_id_by_candidate(cluster_report)
    cluster_summary_by_id = {} if cluster_report is None else {item.cluster_id: item for item in cluster_report.clusters}
    score_by_candidate = _candidate_score_map(shortlist_report)
    curve_by_candidate = {}
    pair_by_candidate_pair = {}
    if diversification_report is not None:
        curve_by_candidate = {item.candidate_id: item for item in diversification_report.candidate_curves}
        pair_by_candidate_pair = _pair_lookup(diversification_report.pair_stats)
    candidate_overrides = {} if override_set is None else candidate_override_map(override_set)
    cluster_overrides = {} if override_set is None else cluster_override_map(override_set)
    selected_ids = set() if shortlist_report is None else set(shortlist_report.selected_candidate_ids)
    preferred_resampling_name = resampling_name
    if preferred_resampling_name is None and shortlist_report is not None:
        preferred_resampling_name = shortlist_report.resampling_name

    row_payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        score = score_by_candidate.get(candidate.candidate_id)
        cluster_id = cluster_by_candidate.get(candidate.candidate_id)
        cluster_summary = None if cluster_id is None else cluster_summary_by_id.get(cluster_id)
        override = candidate_overrides.get(candidate.candidate_id)
        row_payloads.append(
            {
                "candidate_id": candidate.candidate_id,
                "display_name": candidate.display_name,
                "created_at_utc": candidate.created_at_utc,
                "status": candidate.status,
                "tags": list(candidate.tags),
                "engine_name": candidate.engine_name,
                "engine_role": candidate.engine_role,
                "representation_name": candidate.manifest.representation_name if candidate.manifest else None,
                "cluster_id": cluster_id,
                "cluster_size": None if cluster_summary is None else cluster_summary.cluster_size,
                "selection_flags": dict(candidate.selection_flags),
                "periods": _candidate_period_payload(candidate),
                "resampling": _resampling_payload(candidate, preferred_resampling_name),
                "shortlist": {
                    "selected": bool(score and score.selected),
                    "selected_rank": None if score is None else score.selected_rank,
                    "base_score": None if score is None else score.base_score,
                    "marginal_score": None if score is None else score.marginal_score,
                    "brightness_hint": None if score is None else score.brightness_hint,
                    "exception_flags": [] if score is None else list(score.exception_flags),
                    "score_components": {} if score is None else dict(score.score_components),
                },
                "overrides": {
                    "pin": False if override is None else override.pin,
                    "force_include": False if override is None else override.force_include,
                    "exclude": False if override is None else override.exclude,
                    "max_cap_fraction": None if override is None else override.max_cap_fraction,
                    "cluster_max_cap_fraction": None
                    if cluster_id is None or cluster_id not in cluster_overrides
                    else cluster_overrides[cluster_id].max_cap_fraction,
                },
                "notes": candidate.notes,
            }
        )

    def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        created = _parse_utc(row["created_at_utc"])
        oos_pnl = None
        oos_payload = row["periods"].get("oos_adjacent") or row["periods"].get("oos")
        if oos_payload is not None:
            oos_pnl = oos_payload.get("pnl")
        shortlist = row["shortlist"]
        overrides = row["overrides"]
        return (
            0 if shortlist.get("selected") else 1,
            0 if overrides.get("pin") else 1,
            -(shortlist.get("base_score") or float("-inf")),
            -(oos_pnl if oos_pnl is not None else float("-inf")),
            0 if created is None else -created.timestamp(),
        )

    row_payloads.sort(key=_row_sort_key)
    all_row_payloads = list(row_payloads)
    total_candidates = len(all_row_payloads)
    if max_candidate_rows is not None:
        row_payloads = row_payloads[:max_candidate_rows]

    broom = None
    if diversification_report is not None:
        broom_lines: list[dict[str, Any]] = []
        for row in row_payloads:
            curve = curve_by_candidate.get(row["candidate_id"])
            if curve is None:
                continue
            broom_lines.append(
                {
                    "candidate_id": row["candidate_id"],
                    "display_name": row["display_name"],
                    "status": row["status"],
                    "cluster_id": row["cluster_id"],
                    "selected": row["shortlist"]["selected"],
                    "brightness_hint": row["shortlist"]["brightness_hint"],
                    "exception_flags": row["shortlist"]["exception_flags"],
                    "normalized_balance_history": list(curve.normalized_balance_history),
                    "sample_indices": list(curve.sample_indices),
                    "final_balance": curve.final_balance,
                    "max_drawdown_pct": curve.max_drawdown_pct,
                }
            )
        broom_lines.sort(
            key=lambda item: (
                0 if item["selected"] else 1,
                -(item["brightness_hint"] or 0.0),
                item["display_name"],
            )
        )
        total_broom_lines = len(broom_lines)
        if max_broom_lines is not None:
            broom_lines = broom_lines[:max_broom_lines]
        broom = {
            "source_report": diversification_report.name,
            "start_utc": diversification_report.start_utc,
            "end_utc": diversification_report.end_utc,
            "start_step": diversification_report.start_step,
            "max_steps": diversification_report.max_steps,
            "line_count": len(broom_lines),
            "total_line_count": total_broom_lines,
            "lines": broom_lines,
        }

    status_counts: dict[str, int] = {}
    engine_counts: dict[str, int] = {}
    representation_counts: dict[str, int] = {}
    latest_candidate = None
    latest_oos_positive = None
    latest_selected = None
    candidate_timeline: list[dict[str, Any]] = []
    for row in all_row_payloads:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        engine_counts[row["engine_name"]] = engine_counts.get(row["engine_name"], 0) + 1
        representation = row["representation_name"] or "unknown"
        representation_counts[representation] = representation_counts.get(representation, 0) + 1
        created = _parse_utc(row["created_at_utc"])
        oos_positive = bool(row["selection_flags"].get("oos_positive"))
        if latest_candidate is None or (created is not None and created > latest_candidate):
            latest_candidate = created
        if oos_positive and (latest_oos_positive is None or (created is not None and created > latest_oos_positive)):
            latest_oos_positive = created
        if row["shortlist"]["selected"] and (latest_selected is None or (created is not None and created > latest_selected)):
            latest_selected = created
        candidate_timeline.append(
            {
                "candidate_id": row["candidate_id"],
                "display_name": row["display_name"],
                "created_at_utc": row["created_at_utc"],
                "status": row["status"],
                "oos_positive": oos_positive,
                "selected": row["shortlist"]["selected"],
                "pinned": row["overrides"]["pin"],
            }
        )
    candidate_timeline.sort(key=lambda item: (_parse_utc(item["created_at_utc"]) or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)

    cluster_payloads: list[dict[str, Any]] = []
    if cluster_report is not None:
        for cluster in cluster_report.clusters:
            candidate_ids = list(cluster.candidate_ids)
            selected_count = sum(1 for candidate_id in candidate_ids if candidate_id in selected_ids)
            pinned_count = sum(
                1
                for candidate_id in candidate_ids
                if candidate_overrides.get(candidate_id) is not None and candidate_overrides[candidate_id].pin
            )
            cluster_payloads.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "candidate_ids": list(candidate_ids),
                    "display_names": list(cluster.display_names),
                    "cluster_size": cluster.cluster_size,
                    "mean_return_corr": cluster.mean_return_corr,
                    "mean_downside_corr": cluster.mean_downside_corr,
                    "mean_simultaneous_loss_rate": cluster.mean_simultaneous_loss_rate,
                    "mean_similarity_score": cluster.mean_similarity_score,
                    "selected_count": selected_count,
                    "pinned_count": pinned_count,
                    "max_cap_fraction": None
                    if cluster.cluster_id not in cluster_overrides
                    else cluster_overrides[cluster.cluster_id].max_cap_fraction,
                }
            )
        cluster_payloads.sort(key=lambda item: (-item["cluster_size"], item["cluster_id"]))

    overrides_payload = None
    if override_set is not None:
        overrides_payload = {
            "name": override_set.name,
            "updated_at_utc": override_set.updated_at_utc,
            "candidate_override_count": len(override_set.candidate_overrides),
            "cluster_override_count": len(override_set.cluster_overrides),
            "candidate_overrides": [item.to_dict() for item in override_set.candidate_overrides],
            "cluster_overrides": [item.to_dict() for item in override_set.cluster_overrides],
            "recent_audit_entries": [item.to_dict() for item in override_set.audit_entries[-max_audit_entries:]][::-1],
        }

    allocator_payload = None
    if allocator_report is not None:
        scenarios = sorted(allocator_report.scenarios, key=lambda item: item.objective_score, reverse=True)
        allocator_payload = {
            "name": allocator_report.name,
            "chosen_scenario_name": allocator_report.chosen_scenario_name,
            "requested_risk_fractions": list(allocator_report.requested_risk_fractions),
            "selected_candidate_ids": list(allocator_report.selected_candidate_ids),
            "scenarios": [_allocator_scenario_payload(item) for item in scenarios[:max_allocator_scenarios]],
        }

    combinations_payload = None
    if combination_report is not None:
        scenarios = sorted(combination_report.scenarios, key=lambda item: item.objective_score, reverse=True)
        combinations_payload = {
            "name": combination_report.name,
            "best_scenario_name": combination_report.best_scenario_name,
            "pool_candidate_ids": list(combination_report.pool_candidate_ids),
            "searched_subset_sizes": list(combination_report.searched_subset_sizes),
            "evaluated_combination_count": combination_report.evaluated_combination_count,
            "evaluated_scenario_count": combination_report.evaluated_scenario_count,
            "scenarios": [_combination_scenario_payload(item) for item in scenarios[:max_combination_scenarios]],
        }

    summary = {
        "total_candidates": total_candidates,
        "visible_candidate_rows": len(row_payloads),
        "selected_candidate_count": sum(1 for row in all_row_payloads if row["shortlist"]["selected"]),
        "exception_candidate_count": sum(1 for row in all_row_payloads if row["shortlist"]["exception_flags"]),
        "oos_positive_count": sum(1 for row in all_row_payloads if row["selection_flags"].get("oos_positive")),
        "pinned_candidate_count": sum(1 for row in all_row_payloads if row["overrides"]["pin"]),
        "excluded_candidate_count": sum(1 for row in all_row_payloads if row["overrides"]["exclude"]),
        "forced_candidate_count": sum(1 for row in all_row_payloads if row["overrides"]["force_include"]),
        "cluster_count": 0 if cluster_report is None else len(cluster_report.clusters),
        "status_counts": status_counts,
        "engine_counts": engine_counts,
        "representation_counts": representation_counts,
    }
    monitoring = {
        "latest_candidate_created_at": None if latest_candidate is None else latest_candidate.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latest_oos_positive_created_at": None if latest_oos_positive is None else latest_oos_positive.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latest_selected_created_at": None if latest_selected is None else latest_selected.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recent_candidates": candidate_timeline[: min(50, len(candidate_timeline))],
        "interesting_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "display_name": row["display_name"],
                "exception_flags": row["shortlist"]["exception_flags"],
                "pinned": row["overrides"]["pin"],
                "selected": row["shortlist"]["selected"],
                "oos_positive": bool(row["selection_flags"].get("oos_positive")),
            }
            for row in all_row_payloads
            if row["shortlist"]["exception_flags"] or row["overrides"]["pin"]
        ],
    }

    return DashboardFeed(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_shortlist_report=None if shortlist_report is None else shortlist_report.name,
        source_diversification_report=None if diversification_report is None else diversification_report.name,
        source_cluster_report=None if cluster_report is None else cluster_report.name,
        source_override_set=None if override_set is None else override_set.name,
        source_allocator_report=None if allocator_report is None else allocator_report.name,
        source_combination_report=None if combination_report is None else combination_report.name,
        summary=summary,
        candidates=row_payloads,
        monitoring=monitoring,
        broom=broom,
        clusters=cluster_payloads,
        overrides=overrides_payload,
        allocator=allocator_payload,
        combinations=combinations_payload,
        notes=notes,
    )
