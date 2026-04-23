from __future__ import annotations

from dataclasses import replace

from .schema import CandidateOverride, ClusterOverride, OverrideAuditEntry, OverrideSet, utc_now_text


def empty_override_set(
    *,
    name: str,
    description: str | None = None,
    source_cluster_report: str | None = None,
) -> OverrideSet:
    now = utc_now_text()
    return OverrideSet(
        schema_version="1",
        name=name,
        created_at_utc=now,
        updated_at_utc=now,
        source_cluster_report=source_cluster_report,
        description=description,
        candidate_overrides=[],
        cluster_overrides=[],
        audit_entries=[],
    )


def candidate_override_map(override_set: OverrideSet | None) -> dict[str, CandidateOverride]:
    if override_set is None:
        return {}
    return {item.candidate_id: item for item in override_set.candidate_overrides}


def cluster_override_map(override_set: OverrideSet | None) -> dict[str, ClusterOverride]:
    if override_set is None:
        return {}
    return {item.cluster_id: item for item in override_set.cluster_overrides}


def pinned_candidate_ids(override_set: OverrideSet | None) -> list[str]:
    mapping = candidate_override_map(override_set)
    return [candidate_id for candidate_id, item in mapping.items() if item.pin]


def forced_candidate_ids(override_set: OverrideSet | None) -> list[str]:
    mapping = candidate_override_map(override_set)
    return [candidate_id for candidate_id, item in mapping.items() if item.force_include or item.pin]


def excluded_candidate_ids(override_set: OverrideSet | None) -> list[str]:
    mapping = candidate_override_map(override_set)
    return [candidate_id for candidate_id, item in mapping.items() if item.exclude]


def update_candidate_override(
    override_set: OverrideSet,
    *,
    candidate_id: str,
    actor: str,
    force_include: bool | None = None,
    exclude: bool | None = None,
    pin: bool | None = None,
    max_cap_fraction: float | None = None,
    note: str | None = None,
) -> OverrideSet:
    mapping = candidate_override_map(override_set)
    current = mapping.get(candidate_id, CandidateOverride(candidate_id=candidate_id))
    updated = CandidateOverride(
        candidate_id=candidate_id,
        force_include=current.force_include if force_include is None else force_include,
        exclude=current.exclude if exclude is None else exclude,
        pin=current.pin if pin is None else pin,
        max_cap_fraction=current.max_cap_fraction if max_cap_fraction is None else max_cap_fraction,
        note=current.note if note is None else note,
    )
    updated.validate()
    mapping[candidate_id] = updated
    return _rebuild_override_set(
        override_set,
        candidate_overrides=sorted(mapping.values(), key=lambda item: item.candidate_id),
        cluster_overrides=override_set.cluster_overrides,
        audit_entry=OverrideAuditEntry(
            created_at_utc=utc_now_text(),
            actor=actor,
            target_type="candidate",
            target_id=candidate_id,
            action="update_candidate_override",
            changes=updated.to_dict(),
            note=note,
        ),
    )


def update_cluster_override(
    override_set: OverrideSet,
    *,
    cluster_id: str,
    actor: str,
    max_cap_fraction: float | None = None,
    note: str | None = None,
) -> OverrideSet:
    mapping = cluster_override_map(override_set)
    current = mapping.get(cluster_id, ClusterOverride(cluster_id=cluster_id))
    updated = ClusterOverride(
        cluster_id=cluster_id,
        max_cap_fraction=current.max_cap_fraction if max_cap_fraction is None else max_cap_fraction,
        note=current.note if note is None else note,
    )
    updated.validate()
    mapping[cluster_id] = updated
    return _rebuild_override_set(
        override_set,
        candidate_overrides=override_set.candidate_overrides,
        cluster_overrides=sorted(mapping.values(), key=lambda item: item.cluster_id),
        audit_entry=OverrideAuditEntry(
            created_at_utc=utc_now_text(),
            actor=actor,
            target_type="cluster",
            target_id=cluster_id,
            action="update_cluster_override",
            changes=updated.to_dict(),
            note=note,
        ),
    )


def _rebuild_override_set(
    override_set: OverrideSet,
    *,
    candidate_overrides: list[CandidateOverride],
    cluster_overrides: list[ClusterOverride],
    audit_entry: OverrideAuditEntry,
) -> OverrideSet:
    updated = replace(
        override_set,
        updated_at_utc=utc_now_text(),
        candidate_overrides=candidate_overrides,
        cluster_overrides=cluster_overrides,
        audit_entries=[*override_set.audit_entries, audit_entry],
    )
    updated.validate()
    return updated
