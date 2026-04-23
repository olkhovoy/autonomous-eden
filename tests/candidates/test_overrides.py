from __future__ import annotations

from umc_nn.candidates import (
    CandidateRegistry,
    empty_override_set,
    excluded_candidate_ids,
    forced_candidate_ids,
    pinned_candidate_ids,
    update_candidate_override,
    update_cluster_override,
)


def test_override_set_updates_and_audit() -> None:
    override_set = empty_override_set(name="ops", source_cluster_report="clusters_v1")
    override_set = update_candidate_override(
        override_set,
        candidate_id="cand_a",
        actor="operator",
        force_include=True,
        max_cap_fraction=0.25,
        note="watch closely",
    )
    override_set = update_candidate_override(
        override_set,
        candidate_id="cand_b",
        actor="operator",
        pin=True,
        note="must keep",
    )
    override_set = update_candidate_override(
        override_set,
        candidate_id="cand_c",
        actor="operator",
        exclude=True,
        note="remove",
    )
    override_set = update_cluster_override(
        override_set,
        cluster_id="cluster_001",
        actor="operator",
        max_cap_fraction=0.40,
        note="cap this cluster",
    )

    assert forced_candidate_ids(override_set) == ["cand_a", "cand_b"]
    assert pinned_candidate_ids(override_set) == ["cand_b"]
    assert excluded_candidate_ids(override_set) == ["cand_c"]
    assert len(override_set.audit_entries) == 4


def test_registry_saves_and_loads_override_set(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    override_set = empty_override_set(name="ops")
    override_set = update_candidate_override(
        override_set,
        candidate_id="cand_a",
        actor="operator",
        force_include=True,
        max_cap_fraction=0.25,
    )

    path = registry.save_override_set(override_set)
    assert path.exists()
    loaded = registry.load_override_set("ops")
    assert loaded.name == "ops"
    assert forced_candidate_ids(loaded) == ["cand_a"]
