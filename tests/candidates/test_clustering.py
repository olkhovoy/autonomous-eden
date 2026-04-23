from __future__ import annotations

from umc_nn.candidates import (
    CandidateRegistry,
    CandidateTraceView,
    DiversificationReport,
    all_pairwise_diversification,
    build_cluster_report,
    build_trace_view,
    curve_snapshot,
    utc_now_text,
)


def _trace(candidate_id: str, display_name: str, balances: list[float], actions: list[int], max_dd: float) -> CandidateTraceView:
    return CandidateTraceView(
        candidate_id=candidate_id,
        display_name=display_name,
        requested_max_steps=len(actions),
        initial_balance=float(balances[0]),
        balance_history=balances,
        action_history=actions,
        max_drawdown_pct=max_dd,
    )


def test_build_cluster_report_groups_similar_candidates() -> None:
    traces = [
        _trace("a", "A", [100.0, 102.0, 99.0, 101.0], [1, 1, 1], 3.0),
        _trace("b", "B", [100.0, 101.0, 98.5, 100.5], [1, 1, 1], 3.5),
        _trace("c", "C", [100.0, 98.0, 101.0, 99.0], [2, 2, 2], 4.0),
    ]
    diversification = DiversificationReport(
        schema_version="1",
        name="div",
        created_at_utc=utc_now_text(),
        data_path="/tmp/data.npz",
        start_utc="2025-01-01 00:00:00",
        end_utc="2025-01-02 00:00:00",
        start_step=0,
        max_steps=3,
        candidate_ids=["a", "b", "c"],
        candidate_curves=[curve_snapshot(item, max_points=4) for item in traces],
        pair_stats=all_pairwise_diversification(traces),
    )

    report = build_cluster_report(
        diversification,
        name="clusters",
        created_at_utc=utc_now_text(),
        similarity_threshold=0.45,
    )

    assignment = {item.candidate_id: item.cluster_id for item in report.assignments}
    assert assignment["a"] == assignment["b"]
    assert assignment["a"] != assignment["c"]


def test_registry_saves_and_loads_cluster_report(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    traces = [
        _trace("a", "A", [100.0, 102.0, 99.0], [1, 1], 3.0),
        _trace("b", "B", [100.0, 101.0, 98.0], [1, 1], 3.5),
    ]
    diversification = DiversificationReport(
        schema_version="1",
        name="div",
        created_at_utc=utc_now_text(),
        data_path="/tmp/data.npz",
        start_utc="2025-01-01 00:00:00",
        end_utc="2025-01-02 00:00:00",
        start_step=0,
        max_steps=2,
        candidate_ids=["a", "b"],
        candidate_curves=[curve_snapshot(item, max_points=4) for item in traces],
        pair_stats=all_pairwise_diversification(traces),
    )
    report = build_cluster_report(diversification, name="clusters", created_at_utc=utc_now_text(), similarity_threshold=0.1)

    path = registry.save_cluster_report(report)
    assert path.exists()
    loaded = registry.load_cluster_report("clusters")
    assert loaded.name == "clusters"
    assert len(loaded.assignments) == 2
