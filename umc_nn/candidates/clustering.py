from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from .schema import (
    ClusterAssignment,
    ClusterReport,
    ClusterSummary,
    DiversificationPairStats,
    DiversificationReport,
)


def pair_similarity_score(pair: DiversificationPairStats) -> float:
    return (
        0.45 * max(pair.downside_corr, 0.0)
        + 0.30 * pair.simultaneous_loss_rate
        + 0.15 * max(pair.return_corr, 0.0)
        + 0.10 * pair.action_agreement_rate
    )


def build_cluster_report(
    diversification_report: DiversificationReport,
    *,
    name: str,
    created_at_utc: str,
    similarity_threshold: float = 0.40,
    notes: str | None = None,
) -> ClusterReport:
    if similarity_threshold < 0.0:
        raise ValueError("similarity_threshold must be >= 0")

    candidate_ids = list(diversification_report.candidate_ids)
    display_name_by_id = {
        curve.candidate_id: curve.display_name
        for curve in diversification_report.candidate_curves
    }
    for candidate_id in candidate_ids:
        display_name_by_id.setdefault(candidate_id, candidate_id)

    adjacency: dict[str, set[str]] = {candidate_id: set() for candidate_id in candidate_ids}
    pair_map: dict[frozenset[str], DiversificationPairStats] = {}
    for pair in diversification_report.pair_stats:
        key = frozenset((pair.left_candidate_id, pair.right_candidate_id))
        pair_map[key] = pair
        similarity = pair_similarity_score(pair)
        if similarity >= similarity_threshold:
            adjacency[pair.left_candidate_id].add(pair.right_candidate_id)
            adjacency[pair.right_candidate_id].add(pair.left_candidate_id)

    assignments: list[ClusterAssignment] = []
    clusters: list[ClusterSummary] = []
    visited: set[str] = set()
    cluster_index = 0
    for candidate_id in candidate_ids:
        if candidate_id in visited:
            continue
        component = _component(candidate_id, adjacency, visited)
        cluster_id = f"cluster_{cluster_index:03d}"
        cluster_index += 1
        for member_id in component:
            assignments.append(
                ClusterAssignment(
                    candidate_id=member_id,
                    display_name=display_name_by_id[member_id],
                    cluster_id=cluster_id,
                )
            )
        clusters.append(
            _cluster_summary(
                cluster_id=cluster_id,
                candidate_ids=component,
                display_name_by_id=display_name_by_id,
                pair_map=pair_map,
            )
        )

    assignments.sort(key=lambda item: (item.cluster_id, item.display_name))
    clusters.sort(key=lambda item: (item.cluster_size, item.mean_similarity_score, item.cluster_id), reverse=True)
    return ClusterReport(
        schema_version="1",
        name=name,
        created_at_utc=created_at_utc,
        source_diversification_report=diversification_report.name,
        data_path=diversification_report.data_path,
        start_utc=diversification_report.start_utc,
        end_utc=diversification_report.end_utc,
        start_step=diversification_report.start_step,
        max_steps=diversification_report.max_steps,
        similarity_threshold=similarity_threshold,
        candidate_ids=candidate_ids,
        assignments=assignments,
        clusters=clusters,
        notes=notes,
    )


def cluster_id_by_candidate(cluster_report: ClusterReport) -> dict[str, str]:
    return {item.candidate_id: item.cluster_id for item in cluster_report.assignments}


def _component(start: str, adjacency: dict[str, set[str]], visited: set[str]) -> list[str]:
    stack = [start]
    members: list[str] = []
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        members.append(current)
        stack.extend(sorted(adjacency[current] - visited))
    members.sort()
    return members


def _cluster_summary(
    *,
    cluster_id: str,
    candidate_ids: list[str],
    display_name_by_id: dict[str, str],
    pair_map: dict[frozenset[str], DiversificationPairStats],
) -> ClusterSummary:
    pair_items = [
        pair_map[frozenset((left_id, right_id))]
        for left_id, right_id in combinations(candidate_ids, 2)
        if frozenset((left_id, right_id)) in pair_map
    ]
    if not pair_items:
        return ClusterSummary(
            cluster_id=cluster_id,
            candidate_ids=list(candidate_ids),
            display_names=[display_name_by_id[item] for item in candidate_ids],
            cluster_size=len(candidate_ids),
            mean_return_corr=0.0,
            mean_downside_corr=0.0,
            mean_simultaneous_loss_rate=0.0,
            mean_similarity_score=0.0,
        )
    return ClusterSummary(
        cluster_id=cluster_id,
        candidate_ids=list(candidate_ids),
        display_names=[display_name_by_id[item] for item in candidate_ids],
        cluster_size=len(candidate_ids),
        mean_return_corr=sum(pair.return_corr for pair in pair_items) / len(pair_items),
        mean_downside_corr=sum(pair.downside_corr for pair in pair_items) / len(pair_items),
        mean_simultaneous_loss_rate=sum(pair.simultaneous_loss_rate for pair in pair_items) / len(pair_items),
        mean_similarity_score=sum(pair_similarity_score(pair) for pair in pair_items) / len(pair_items),
    )
