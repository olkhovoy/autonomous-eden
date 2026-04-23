from __future__ import annotations

from umc_nn.candidates import CandidateRegistry, DiversificationReport, all_pairwise_diversification, curve_snapshot, utc_now_text
from umc_nn.candidates.diversification import CandidateTraceView


def _trace_view(candidate_id: str, display_name: str, balances: list[float], actions: list[int], max_dd: float) -> CandidateTraceView:
    return CandidateTraceView(
        candidate_id=candidate_id,
        display_name=display_name,
        requested_max_steps=len(actions),
        initial_balance=float(balances[0]),
        balance_history=balances,
        action_history=actions,
        max_drawdown_pct=max_dd,
    )


def test_pairwise_diversification_prefers_negative_correlation() -> None:
    a = _trace_view("a", "A", [100.0, 110.0, 99.0, 108.9], [1, 1, 1], 10.0)
    b = _trace_view("b", "B", [100.0, 110.0, 99.0, 108.9], [1, 1, 1], 10.0)
    c = _trace_view("c", "C", [100.0, 90.0, 99.0, 89.1], [2, 2, 2], 10.0)

    pairs = all_pairwise_diversification([a, b, c])
    assert len(pairs) == 3

    same = next(item for item in pairs if {item.left_candidate_id, item.right_candidate_id} == {"a", "b"})
    opposite = next(item for item in pairs if {item.left_candidate_id, item.right_candidate_id} == {"a", "c"})

    assert same.return_corr > 0.99
    assert opposite.return_corr < -0.99
    assert opposite.simultaneous_loss_rate < same.simultaneous_loss_rate
    assert opposite.drawdown_improvement_pct_points > same.drawdown_improvement_pct_points


def test_registry_saves_and_loads_diversification_report(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    traces = [
        _trace_view("a", "A", [100.0, 101.0, 102.0], [1, 1], 0.0),
        _trace_view("b", "B", [100.0, 99.0, 98.0], [2, 2], 2.0),
    ]
    pair = all_pairwise_diversification(traces)[0]
    report = DiversificationReport(
        schema_version="1",
        name="sample_report",
        created_at_utc=utc_now_text(),
        data_path="/tmp/data.npz",
        start_utc="2025-01-01 00:00:00",
        end_utc="2025-01-02 00:00:00",
        start_step=0,
        max_steps=2,
        candidate_ids=["a", "b"],
        candidate_curves=[curve_snapshot(item, max_points=4) for item in traces],
        pair_stats=[pair],
    )

    path = registry.save_diversification_report(report)
    assert path.exists()

    loaded = registry.load_diversification_report("sample_report")
    assert loaded.name == "sample_report"
    assert len(loaded.pair_stats) == 1
    assert len(loaded.candidate_curves) == 2
    assert loaded.pair_stats[0].left_candidate_id == "a"
