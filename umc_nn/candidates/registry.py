from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .schema import (
    AllocatorWorkbenchReport,
    CANDIDATE_STATUSES,
    CandidateFarmReport,
    FarmProgressLog,
    FarmDashboardFeed,
    CandidateIndexEntry,
    CandidateRecord,
    ClusterReport,
    ContinuousSearchCycleReport,
    DashboardFeed,
    CombinationSearchReport,
    DiversificationReport,
    LifecycleReport,
    OverrideSet,
    PortfolioBaselineReport,
    PortfolioLedgerReport,
    RuleSetRecord,
    RollingConveyorReport,
    ResamplingStats,
    ShortlistReport,
    TradeforwardPlan,
    TradeforwardEvaluationReport,
    TradeRecord,
    TracePeriodRecord,
    read_json,
    write_json,
)
from .selection import Rule, evaluate_rules


class CandidateRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.candidates_dir = self.root / "candidates"
        self.rules_dir = self.root / "rules"
        self.trades_dir = self.root / "trades"
        self.traces_dir = self.root / "traces"
        self.resampling_dir = self.root / "resampling"
        self.diversification_dir = self.root / "diversification"
        self.clusters_dir = self.root / "clusters"
        self.shortlists_dir = self.root / "shortlists"
        self.allocations_dir = self.root / "allocations"
        self.combinations_dir = self.root / "combinations"
        self.overrides_dir = self.root / "overrides"
        self.dashboard_dir = self.root / "dashboard"
        self.farm_dashboard_dir = self.root / "farm_dashboard"
        self.farms_dir = self.root / "farms"
        self.farm_progress_dir = self.root / "farm_progress"
        self.cycles_dir = self.root / "cycles"
        self.tradeforward_dir = self.root / "tradeforward"
        self.tradeforward_eval_dir = self.root / "tradeforward_eval"
        self.lifecycle_dir = self.root / "lifecycle"
        self.portfolio_ledger_dir = self.root / "portfolio_ledger"
        self.portfolio_baselines_dir = self.root / "portfolio_baselines"
        self.rolling_dir = self.root / "rolling"
        self.index_path = self.root / "index.json"
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.resampling_dir.mkdir(parents=True, exist_ok=True)
        self.diversification_dir.mkdir(parents=True, exist_ok=True)
        self.clusters_dir.mkdir(parents=True, exist_ok=True)
        self.shortlists_dir.mkdir(parents=True, exist_ok=True)
        self.allocations_dir.mkdir(parents=True, exist_ok=True)
        self.combinations_dir.mkdir(parents=True, exist_ok=True)
        self.overrides_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.farm_dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.farms_dir.mkdir(parents=True, exist_ok=True)
        self.farm_progress_dir.mkdir(parents=True, exist_ok=True)
        self.cycles_dir.mkdir(parents=True, exist_ok=True)
        self.tradeforward_dir.mkdir(parents=True, exist_ok=True)
        self.tradeforward_eval_dir.mkdir(parents=True, exist_ok=True)
        self.lifecycle_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_ledger_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_baselines_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_dir.mkdir(parents=True, exist_ok=True)

    def candidate_path(self, candidate_id: str) -> Path:
        return self.candidates_dir / f"{candidate_id}.json"

    def rule_path(self, name: str) -> Path:
        return self.rules_dir / f"{name}.json"

    def trades_path(self, candidate_id: str) -> Path:
        return self.trades_dir / f"{candidate_id}.json"

    def traces_path(self, candidate_id: str) -> Path:
        return self.traces_dir / f"{candidate_id}.json"

    def resampling_path(self, candidate_id: str) -> Path:
        return self.resampling_dir / f"{candidate_id}.json"

    def diversification_path(self, name: str) -> Path:
        return self.diversification_dir / f"{name}.json"

    def cluster_path(self, name: str) -> Path:
        return self.clusters_dir / f"{name}.json"

    def shortlist_path(self, name: str) -> Path:
        return self.shortlists_dir / f"{name}.json"

    def allocation_path(self, name: str) -> Path:
        return self.allocations_dir / f"{name}.json"

    def combination_path(self, name: str) -> Path:
        return self.combinations_dir / f"{name}.json"

    def override_path(self, name: str) -> Path:
        return self.overrides_dir / f"{name}.json"

    def dashboard_path(self, name: str) -> Path:
        return self.dashboard_dir / f"{name}.json"

    def farm_dashboard_path(self, name: str) -> Path:
        return self.farm_dashboard_dir / f"{name}.json"

    def farm_path(self, name: str) -> Path:
        return self.farms_dir / f"{name}.json"

    def farm_progress_path(self, name: str) -> Path:
        return self.farm_progress_dir / f"{name}.json"

    def cycle_path(self, name: str) -> Path:
        return self.cycles_dir / f"{name}.json"

    def tradeforward_path(self, name: str) -> Path:
        return self.tradeforward_dir / f"{name}.json"

    def tradeforward_eval_path(self, name: str) -> Path:
        return self.tradeforward_eval_dir / f"{name}.json"

    def lifecycle_path(self, name: str) -> Path:
        return self.lifecycle_dir / f"{name}.json"

    def portfolio_ledger_path(self, name: str) -> Path:
        return self.portfolio_ledger_dir / f"{name}.json"

    def portfolio_baselines_path(self, name: str) -> Path:
        return self.portfolio_baselines_dir / f"{name}.json"

    def rolling_path(self, name: str) -> Path:
        return self.rolling_dir / f"{name}.json"

    def add_candidate(self, record: CandidateRecord, *, overwrite: bool = False) -> Path:
        record.validate()
        path = self.candidate_path(record.candidate_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Candidate already exists: {record.candidate_id}")
        write_json(path, record.to_dict())
        self.rebuild_index()
        return path

    def load_candidate(self, candidate_id: str) -> CandidateRecord:
        return CandidateRecord.from_dict(read_json(self.candidate_path(candidate_id)))

    def iter_candidates(self) -> Iterable[CandidateRecord]:
        for path in sorted(self.candidates_dir.glob("*.json")):
            yield CandidateRecord.from_dict(read_json(path))

    def list_candidates(
        self,
        *,
        status: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> list[CandidateRecord]:
        required_tags = set(tags or [])
        result: list[CandidateRecord] = []
        for candidate in self.iter_candidates():
            if status is not None and candidate.status != status:
                continue
            if required_tags and not required_tags.issubset(set(candidate.tags)):
                continue
            result.append(candidate)
        return result

    def update_status(
        self,
        candidate_id: str,
        status: str,
        *,
        note: str | None = None,
        add_tags: Iterable[str] | None = None,
    ) -> CandidateRecord:
        if status not in CANDIDATE_STATUSES:
            raise ValueError(f"Unknown candidate status: {status}")
        candidate = self.load_candidate(candidate_id)
        candidate.status = status
        if add_tags:
            candidate.tags = sorted(set(candidate.tags).union(set(add_tags)))
        if note:
            candidate.notes = note if candidate.notes is None else f"{candidate.notes}\n{note}"
        self.add_candidate(candidate, overwrite=True)
        return candidate

    def attach_trade_records(
        self,
        candidate_id: str,
        trade_records: Iterable[TradeRecord],
        *,
        overwrite: bool = True,
    ) -> Path:
        candidate = self.load_candidate(candidate_id)
        trade_records = list(trade_records)
        path = self.trades_path(candidate_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Trade record artifact already exists: {path}")
        payload = {
            "candidate_id": candidate_id,
            "created_at_utc": candidate.created_at_utc,
            "trade_count": len(trade_records),
            "trades": [trade.to_dict() for trade in trade_records],
        }
        write_json(path, payload)
        candidate.trade_records_path = str(path)
        self.add_candidate(candidate, overwrite=True)
        return path

    def load_trade_records(self, candidate_id: str) -> list[TradeRecord]:
        payload = read_json(self.trades_path(candidate_id))
        return [TradeRecord.from_dict(item) for item in payload.get("trades", [])]

    def attach_trace_records(
        self,
        candidate_id: str,
        trace_records: Iterable[TracePeriodRecord],
        *,
        overwrite: bool = True,
    ) -> Path:
        candidate = self.load_candidate(candidate_id)
        trace_records = list(trace_records)
        path = self.traces_path(candidate_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Trace artifact already exists: {path}")
        payload = {
            "candidate_id": candidate_id,
            "created_at_utc": candidate.created_at_utc,
            "trace_count": len(trace_records),
            "traces": [trace.to_dict() for trace in trace_records],
        }
        write_json(path, payload)
        candidate.trace_records_path = str(path)
        self.add_candidate(candidate, overwrite=True)
        return path

    def load_trace_records(self, candidate_id: str) -> list[TracePeriodRecord]:
        payload = read_json(self.traces_path(candidate_id))
        return [TracePeriodRecord.from_dict(item) for item in payload.get("traces", [])]

    def attach_resampling_results(
        self,
        candidate_id: str,
        results: dict[str, ResamplingStats],
        *,
        overwrite: bool = True,
    ) -> Path:
        candidate = self.load_candidate(candidate_id)
        path = self.resampling_path(candidate_id)
        existing_payload = {"candidate_id": candidate_id, "results": {}}
        if path.exists() and not overwrite:
            raise FileExistsError(f"Resampling artifact already exists: {path}")
        if path.exists() and overwrite:
            existing_payload = read_json(path)

        merged_results = {
            str(name): ResamplingStats.from_dict(stats)
            for name, stats in dict(existing_payload.get("results", {})).items()
        }
        merged_results.update(results)
        write_json(
            path,
            {
                "candidate_id": candidate_id,
                "results": {name: stats.to_dict() for name, stats in merged_results.items()},
            },
        )
        candidate.resampling_results = merged_results
        candidate.resampling_artifact_path = str(path)
        self.add_candidate(candidate, overwrite=True)
        return path

    def save_rule_set(self, rule_set: RuleSetRecord) -> Path:
        path = self.rule_path(rule_set.name)
        write_json(path, rule_set.to_dict())
        return path

    def load_rule_set(self, name: str) -> RuleSetRecord:
        return RuleSetRecord.from_dict(read_json(self.rule_path(name)))

    def save_diversification_report(self, report: DiversificationReport) -> Path:
        path = self.diversification_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_diversification_report(self, name: str) -> DiversificationReport:
        return DiversificationReport.from_dict(read_json(self.diversification_path(name)))

    def save_cluster_report(self, report: ClusterReport) -> Path:
        path = self.cluster_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_cluster_report(self, name: str) -> ClusterReport:
        return ClusterReport.from_dict(read_json(self.cluster_path(name)))

    def save_shortlist_report(self, report: ShortlistReport) -> Path:
        path = self.shortlist_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_shortlist_report(self, name: str) -> ShortlistReport:
        return ShortlistReport.from_dict(read_json(self.shortlist_path(name)))

    def save_allocator_workbench(self, report: AllocatorWorkbenchReport) -> Path:
        path = self.allocation_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_allocator_workbench(self, name: str) -> AllocatorWorkbenchReport:
        return AllocatorWorkbenchReport.from_dict(read_json(self.allocation_path(name)))

    def save_combination_search(self, report: CombinationSearchReport) -> Path:
        path = self.combination_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_combination_search(self, name: str) -> CombinationSearchReport:
        return CombinationSearchReport.from_dict(read_json(self.combination_path(name)))

    def save_override_set(self, override_set: OverrideSet) -> Path:
        override_set.validate()
        path = self.override_path(override_set.name)
        write_json(path, override_set.to_dict())
        return path

    def load_override_set(self, name: str) -> OverrideSet:
        return OverrideSet.from_dict(read_json(self.override_path(name)))

    def save_dashboard_feed(self, feed: DashboardFeed) -> Path:
        path = self.dashboard_path(feed.name)
        write_json(path, feed.to_dict())
        return path

    def load_dashboard_feed(self, name: str) -> DashboardFeed:
        return DashboardFeed.from_dict(read_json(self.dashboard_path(name)))

    def save_farm_dashboard_feed(self, feed: FarmDashboardFeed) -> Path:
        path = self.farm_dashboard_path(feed.name)
        write_json(path, feed.to_dict())
        return path

    def load_farm_dashboard_feed(self, name: str) -> FarmDashboardFeed:
        return FarmDashboardFeed.from_dict(read_json(self.farm_dashboard_path(name)))

    def save_farm_report(self, report: CandidateFarmReport) -> Path:
        path = self.farm_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_farm_report(self, name: str) -> CandidateFarmReport:
        return CandidateFarmReport.from_dict(read_json(self.farm_path(name)))

    def save_farm_progress_log(self, report: FarmProgressLog) -> Path:
        path = self.farm_progress_path(report.farm_name)
        write_json(path, report.to_dict())
        return path

    def load_farm_progress_log(self, name: str) -> FarmProgressLog:
        return FarmProgressLog.from_dict(read_json(self.farm_progress_path(name)))

    def save_cycle_report(self, report: ContinuousSearchCycleReport) -> Path:
        path = self.cycle_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_cycle_report(self, name: str) -> ContinuousSearchCycleReport:
        return ContinuousSearchCycleReport.from_dict(read_json(self.cycle_path(name)))

    def save_tradeforward_plan(self, plan: TradeforwardPlan) -> Path:
        path = self.tradeforward_path(plan.name)
        write_json(path, plan.to_dict())
        return path

    def load_tradeforward_plan(self, name: str) -> TradeforwardPlan:
        return TradeforwardPlan.from_dict(read_json(self.tradeforward_path(name)))

    def save_tradeforward_evaluation(self, report: TradeforwardEvaluationReport) -> Path:
        path = self.tradeforward_eval_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_tradeforward_evaluation(self, name: str) -> TradeforwardEvaluationReport:
        return TradeforwardEvaluationReport.from_dict(read_json(self.tradeforward_eval_path(name)))

    def save_lifecycle_report(self, report: LifecycleReport) -> Path:
        path = self.lifecycle_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_lifecycle_report(self, name: str) -> LifecycleReport:
        return LifecycleReport.from_dict(read_json(self.lifecycle_path(name)))

    def save_portfolio_ledger(self, report: PortfolioLedgerReport) -> Path:
        path = self.portfolio_ledger_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_portfolio_ledger(self, name: str) -> PortfolioLedgerReport:
        return PortfolioLedgerReport.from_dict(read_json(self.portfolio_ledger_path(name)))

    def save_portfolio_baselines(self, report: PortfolioBaselineReport) -> Path:
        path = self.portfolio_baselines_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_portfolio_baselines(self, name: str) -> PortfolioBaselineReport:
        return PortfolioBaselineReport.from_dict(read_json(self.portfolio_baselines_path(name)))

    def save_rolling_conveyor(self, report: RollingConveyorReport) -> Path:
        path = self.rolling_path(report.name)
        write_json(path, report.to_dict())
        return path

    def load_rolling_conveyor(self, name: str) -> RollingConveyorReport:
        return RollingConveyorReport.from_dict(read_json(self.rolling_path(name)))

    def filter_candidates(
        self,
        rules: Iterable[Rule],
        *,
        require_all: bool = True,
        status: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> list[tuple[CandidateRecord, list]]:
        matches: list[tuple[CandidateRecord, list]] = []
        for candidate in self.list_candidates(status=status, tags=tags):
            matched, evaluations = evaluate_rules(candidate, rules, require_all=require_all)
            if matched:
                matches.append((candidate, evaluations))
        return matches

    def rebuild_index(self) -> list[CandidateIndexEntry]:
        entries = [CandidateIndexEntry.from_candidate(candidate) for candidate in self.iter_candidates()]
        entries.sort(key=lambda item: item.created_at_utc)
        write_json(self.index_path, {"entries": [entry.to_dict() for entry in entries]})
        return entries
