from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CANDIDATE_STATUSES = (
    "research",
    "approved",
    "paper",
    "active",
    "draining",
    "retired",
    "rejected",
)


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_candidate_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"cand_{hashlib.sha1(encoded).hexdigest()[:12]}"


def _normalize_counts(raw: Mapping[str, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in raw.items()}


@dataclass(slots=True)
class PeriodSpan:
    name: str
    start_utc: str | None
    end_utc: str | None
    start_step: int
    end_step: int
    requested_max_steps: int
    requested_row_count: int
    row_count: int

    @classmethod
    def from_window(
        cls,
        name: str,
        *,
        start_utc: str | None,
        end_utc: str | None,
        start_step: int,
        max_steps: int,
    ) -> "PeriodSpan":
        return cls(
            name=name,
            start_utc=start_utc,
            end_utc=end_utc,
            start_step=int(start_step),
            end_step=int(start_step + max_steps),
            requested_max_steps=int(max_steps),
            requested_row_count=int(max_steps + 1),
            row_count=int(max_steps + 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PeriodSpan":
        return cls(
            name=str(payload["name"]),
            start_utc=payload.get("start_utc"),
            end_utc=payload.get("end_utc"),
            start_step=int(payload["start_step"]),
            end_step=int(payload["end_step"]),
            requested_max_steps=int(payload["requested_max_steps"]),
            requested_row_count=int(payload.get("requested_row_count", int(payload["requested_max_steps"]) + 1)),
            row_count=int(payload["row_count"]),
        )


@dataclass(slots=True)
class PeriodStats:
    name: str
    start_step: int
    requested_max_steps: int
    steps_run: int
    final_balance: float
    pnl: float
    max_drawdown_pct: float
    trades: int
    wins: int
    win_rate_pct: float
    action_counts: dict[str, int]
    position_counts: dict[str, int]
    start_utc: str | None = None
    end_utc: str | None = None
    beats_flat: bool | None = None
    beats_best_baseline: bool | None = None
    full_window: bool | None = None
    baseline_winner: str | None = None
    end_step: int = -1
    row_count: int = -1

    def __post_init__(self) -> None:
        if self.end_step < 0:
            self.end_step = int(self.start_step + self.steps_run)
        if self.row_count < 0:
            self.row_count = int(self.steps_run + 1)

    @classmethod
    def from_episode(
        cls,
        name: str,
        metrics: Mapping[str, Any],
        *,
        start_utc: str | None = None,
        end_utc: str | None = None,
        beats_flat: bool | None = None,
        beats_best_baseline: bool | None = None,
        full_window: bool | None = None,
        baseline_winner: str | None = None,
    ) -> "PeriodStats":
        return cls(
            name=name,
            start_step=int(metrics["start_step"]),
            end_step=int(metrics["start_step"]) + int(metrics["steps_run"]),
            requested_max_steps=int(metrics["requested_max_steps"]),
            row_count=int(metrics["steps_run"]) + 1,
            steps_run=int(metrics["steps_run"]),
            final_balance=float(metrics["final_balance"]),
            pnl=float(metrics["pnl"]),
            max_drawdown_pct=float(metrics["max_drawdown_pct"]),
            trades=int(metrics["trades"]),
            wins=int(metrics["wins"]),
            win_rate_pct=float(metrics["win_rate_pct"]),
            action_counts=_normalize_counts(metrics["action_counts"]),
            position_counts=_normalize_counts(metrics["position_counts"]),
            start_utc=start_utc,
            end_utc=end_utc,
            beats_flat=beats_flat,
            beats_best_baseline=beats_best_baseline,
            full_window=full_window,
            baseline_winner=baseline_winner,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PeriodStats":
        return cls(
            name=str(payload["name"]),
            start_step=int(payload["start_step"]),
            end_step=int(payload.get("end_step", int(payload["start_step"]) + int(payload["steps_run"]))),
            requested_max_steps=int(payload["requested_max_steps"]),
            row_count=int(payload.get("row_count", int(payload["steps_run"]) + 1)),
            steps_run=int(payload["steps_run"]),
            final_balance=float(payload["final_balance"]),
            pnl=float(payload["pnl"]),
            max_drawdown_pct=float(payload["max_drawdown_pct"]),
            trades=int(payload["trades"]),
            wins=int(payload["wins"]),
            win_rate_pct=float(payload["win_rate_pct"]),
            action_counts=_normalize_counts(payload["action_counts"]),
            position_counts=_normalize_counts(payload["position_counts"]),
            start_utc=payload.get("start_utc"),
            end_utc=payload.get("end_utc"),
            beats_flat=payload.get("beats_flat"),
            beats_best_baseline=payload.get("beats_best_baseline"),
            full_window=payload.get("full_window"),
            baseline_winner=payload.get("baseline_winner"),
        )


@dataclass(slots=True)
class TradeRecord:
    trade_id: str
    period_name: str
    direction: str
    entry_step: int
    exit_step: int
    entry_timestamp_utc: str | None
    exit_timestamp_utc: str | None
    entry_price: float | None
    exit_price: float | None
    quantity: float | None
    gross_pnl: float | None
    net_pnl: float | None
    fees_paid: float | None
    duration_steps: int | None
    entry_balance: float | None = None
    exit_balance: float | None = None
    return_on_equity: float | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeRecord":
        return cls(
            trade_id=str(payload["trade_id"]),
            period_name=str(payload["period_name"]),
            direction=str(payload["direction"]),
            entry_step=int(payload["entry_step"]),
            exit_step=int(payload["exit_step"]),
            entry_timestamp_utc=payload.get("entry_timestamp_utc"),
            exit_timestamp_utc=payload.get("exit_timestamp_utc"),
            entry_price=None if payload.get("entry_price") is None else float(payload["entry_price"]),
            exit_price=None if payload.get("exit_price") is None else float(payload["exit_price"]),
            quantity=None if payload.get("quantity") is None else float(payload["quantity"]),
            gross_pnl=None if payload.get("gross_pnl") is None else float(payload["gross_pnl"]),
            net_pnl=None if payload.get("net_pnl") is None else float(payload["net_pnl"]),
            fees_paid=None if payload.get("fees_paid") is None else float(payload["fees_paid"]),
            duration_steps=None if payload.get("duration_steps") is None else int(payload["duration_steps"]),
            entry_balance=None if payload.get("entry_balance") is None else float(payload["entry_balance"]),
            exit_balance=None if payload.get("exit_balance") is None else float(payload["exit_balance"]),
            return_on_equity=None if payload.get("return_on_equity") is None else float(payload["return_on_equity"]),
            source=payload.get("source"),
        )


@dataclass(slots=True)
class TracePeriodRecord:
    period_name: str
    source: str
    start_step: int
    requested_max_steps: int
    steps_run: int
    start_utc: str | None
    end_utc: str | None
    balance_history: list[float]
    action_history: list[int]
    position_history: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_name": self.period_name,
            "source": self.source,
            "start_step": self.start_step,
            "requested_max_steps": self.requested_max_steps,
            "steps_run": self.steps_run,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "balance_history": list(self.balance_history),
            "action_history": list(self.action_history),
            "position_history": list(self.position_history),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TracePeriodRecord":
        return cls(
            period_name=str(payload["period_name"]),
            source=str(payload["source"]),
            start_step=int(payload["start_step"]),
            requested_max_steps=int(payload["requested_max_steps"]),
            steps_run=int(payload["steps_run"]),
            start_utc=payload.get("start_utc"),
            end_utc=payload.get("end_utc"),
            balance_history=[float(item) for item in payload.get("balance_history", [])],
            action_history=[int(item) for item in payload.get("action_history", [])],
            position_history=[int(item) for item in payload.get("position_history", [])],
        )


@dataclass(slots=True)
class ResamplingStats:
    name: str
    period_name: str
    iterations: int
    sample_size: int
    seed: int
    sizing_mode: str
    fraction: float
    initial_balance: float
    original_trade_count: int
    original_final_balance: float
    original_net_profit: float
    original_max_drawdown_pct: float
    mean_final_balance: float
    median_final_balance: float
    p05_final_balance: float
    p25_final_balance: float
    mean_net_profit: float
    median_net_profit: float
    p05_net_profit: float
    p25_net_profit: float
    mean_max_drawdown_pct: float
    median_max_drawdown_pct: float
    p75_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    profitable_rate: float
    loss_rate: float
    ruin_rate: float
    pessimistic_net_profit: float
    pessimistic_max_drawdown_pct: float
    replay_mode: str = "trade_bootstrap"
    sampler_id: str = "iid_trade_bootstrap"
    overlap_policy: str = "trade_independent"
    steps: int | None = None
    capital_path_distribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResamplingStats":
        return cls(
            name=str(payload["name"]),
            period_name=str(payload["period_name"]),
            iterations=int(payload["iterations"]),
            sample_size=int(payload["sample_size"]),
            seed=int(payload["seed"]),
            sizing_mode=str(payload["sizing_mode"]),
            fraction=float(payload["fraction"]),
            initial_balance=float(payload["initial_balance"]),
            replay_mode=str(payload.get("replay_mode", "trade_bootstrap")),
            sampler_id=str(payload.get("sampler_id", "iid_trade_bootstrap")),
            overlap_policy=str(payload.get("overlap_policy", "trade_independent")),
            steps=None if payload.get("steps") is None else int(payload["steps"]),
            original_trade_count=int(payload["original_trade_count"]),
            original_final_balance=float(payload["original_final_balance"]),
            original_net_profit=float(payload["original_net_profit"]),
            original_max_drawdown_pct=float(payload["original_max_drawdown_pct"]),
            mean_final_balance=float(payload["mean_final_balance"]),
            median_final_balance=float(payload["median_final_balance"]),
            p05_final_balance=float(payload["p05_final_balance"]),
            p25_final_balance=float(payload["p25_final_balance"]),
            mean_net_profit=float(payload["mean_net_profit"]),
            median_net_profit=float(payload["median_net_profit"]),
            p05_net_profit=float(payload["p05_net_profit"]),
            p25_net_profit=float(payload["p25_net_profit"]),
            mean_max_drawdown_pct=float(payload["mean_max_drawdown_pct"]),
            median_max_drawdown_pct=float(payload["median_max_drawdown_pct"]),
            p75_max_drawdown_pct=float(payload["p75_max_drawdown_pct"]),
            p95_max_drawdown_pct=float(payload["p95_max_drawdown_pct"]),
            profitable_rate=float(payload["profitable_rate"]),
            loss_rate=float(payload["loss_rate"]),
            ruin_rate=float(payload["ruin_rate"]),
            pessimistic_net_profit=float(payload["pessimistic_net_profit"]),
            pessimistic_max_drawdown_pct=float(payload["pessimistic_max_drawdown_pct"]),
            capital_path_distribution={
                str(key): float(value)
                for key, value in dict(payload.get("capital_path_distribution", {})).items()
            },
        )


@dataclass(slots=True)
class ExperimentManifest:
    schema_version: str
    created_at_utc: str
    source_script: str
    engine_name: str
    engine_role: str
    representation_name: str
    data_path: str
    checkpoint_path: str
    log_path: str | None
    source_summary_path: str | None
    train_window_utc: dict[str, str] | None
    oos_window_utc: dict[str, str] | None
    engine_family: str | None = None
    engine_config_id: str | None = None
    representation_branch: str | None = None
    fitness_profile: str | None = None
    data_branch_id: str | None = None
    source_coverage_id: str | None = None
    market_scope: str | None = None
    search_config: dict[str, Any] = field(default_factory=dict)
    econ_config: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentManifest":
        return cls(
            schema_version=str(payload["schema_version"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_script=str(payload["source_script"]),
            engine_name=str(payload["engine_name"]),
            engine_role=str(payload["engine_role"]),
            representation_name=str(payload["representation_name"]),
            data_path=str(payload["data_path"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            log_path=payload.get("log_path"),
            source_summary_path=payload.get("source_summary_path"),
            train_window_utc=dict(payload["train_window_utc"]) if payload.get("train_window_utc") else None,
            oos_window_utc=dict(payload["oos_window_utc"]) if payload.get("oos_window_utc") else None,
            engine_family=payload.get("engine_family"),
            engine_config_id=payload.get("engine_config_id"),
            representation_branch=payload.get("representation_branch"),
            fitness_profile=payload.get("fitness_profile"),
            data_branch_id=payload.get("data_branch_id"),
            source_coverage_id=payload.get("source_coverage_id"),
            market_scope=payload.get("market_scope"),
            search_config=dict(payload.get("search_config", {})),
            econ_config=dict(payload.get("econ_config", {})),
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class CandidateRecord:
    schema_version: str
    candidate_id: str
    display_name: str
    engine_name: str
    engine_role: str
    status: str
    created_at_utc: str
    engine_family: str | None = None
    engine_config_id: str | None = None
    representation_branch: str | None = None
    fitness_profile: str | None = None
    data_branch_id: str | None = None
    source_coverage_id: str | None = None
    market_scope: str | None = None
    train_span: PeriodSpan | None = None
    valid_span: PeriodSpan | None = None
    oos_adjacent_span: PeriodSpan | None = None
    forward_span: PeriodSpan | None = None
    total_span: PeriodSpan | None = None
    tags: list[str] = field(default_factory=list)
    manifest: ExperimentManifest | None = None
    periods: dict[str, PeriodStats] = field(default_factory=dict)
    baselines: dict[str, dict[str, PeriodStats]] = field(default_factory=dict)
    selection_flags: dict[str, bool] = field(default_factory=dict)
    trade_records_path: str | None = None
    trace_records_path: str | None = None
    resampling_results: dict[str, ResamplingStats] = field(default_factory=dict)
    resampling_artifact_path: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "engine_name": self.engine_name,
            "engine_role": self.engine_role,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "engine_family": self.engine_family,
            "engine_config_id": self.engine_config_id,
            "representation_branch": self.representation_branch,
            "fitness_profile": self.fitness_profile,
            "data_branch_id": self.data_branch_id,
            "source_coverage_id": self.source_coverage_id,
            "market_scope": self.market_scope,
            "train_span": None if self.train_span is None else self.train_span.to_dict(),
            "valid_span": None if self.valid_span is None else self.valid_span.to_dict(),
            "oos_adjacent_span": None if self.oos_adjacent_span is None else self.oos_adjacent_span.to_dict(),
            "forward_span": None if self.forward_span is None else self.forward_span.to_dict(),
            "total_span": None if self.total_span is None else self.total_span.to_dict(),
            "tags": list(self.tags),
            "manifest": None if self.manifest is None else self.manifest.to_dict(),
            "periods": {name: stats.to_dict() for name, stats in self.periods.items()},
            "baselines": {
                period_name: {policy: stats.to_dict() for policy, stats in baseline_map.items()}
                for period_name, baseline_map in self.baselines.items()
            },
            "selection_flags": dict(self.selection_flags),
            "trade_records_path": self.trade_records_path,
            "trace_records_path": self.trace_records_path,
            "resampling_results": {
                name: stats.to_dict() for name, stats in self.resampling_results.items()
            },
            "resampling_artifact_path": self.resampling_artifact_path,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateRecord":
        return cls(
            schema_version=str(payload["schema_version"]),
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            engine_name=str(payload["engine_name"]),
            engine_role=str(payload["engine_role"]),
            status=str(payload["status"]),
            created_at_utc=str(payload["created_at_utc"]),
            engine_family=payload.get("engine_family"),
            engine_config_id=payload.get("engine_config_id"),
            representation_branch=payload.get("representation_branch"),
            fitness_profile=payload.get("fitness_profile"),
            data_branch_id=payload.get("data_branch_id"),
            source_coverage_id=payload.get("source_coverage_id"),
            market_scope=payload.get("market_scope"),
            train_span=None if payload.get("train_span") is None else PeriodSpan.from_dict(payload["train_span"]),
            valid_span=None if payload.get("valid_span") is None else PeriodSpan.from_dict(payload["valid_span"]),
            oos_adjacent_span=None
            if payload.get("oos_adjacent_span") is None
            else PeriodSpan.from_dict(payload["oos_adjacent_span"]),
            forward_span=None if payload.get("forward_span") is None else PeriodSpan.from_dict(payload["forward_span"]),
            total_span=None if payload.get("total_span") is None else PeriodSpan.from_dict(payload["total_span"]),
            tags=[str(tag) for tag in payload.get("tags", [])],
            manifest=None if payload.get("manifest") is None else ExperimentManifest.from_dict(payload["manifest"]),
            periods={
                name: PeriodStats.from_dict(stats)
                for name, stats in dict(payload.get("periods", {})).items()
            },
            baselines={
                period_name: {
                    policy: PeriodStats.from_dict(stats)
                    for policy, stats in dict(period_payload).items()
                }
                for period_name, period_payload in dict(payload.get("baselines", {})).items()
            },
            selection_flags={str(key): bool(value) for key, value in dict(payload.get("selection_flags", {})).items()},
            trade_records_path=payload.get("trade_records_path"),
            trace_records_path=payload.get("trace_records_path"),
            resampling_results={
                name: ResamplingStats.from_dict(stats)
                for name, stats in dict(payload.get("resampling_results", {})).items()
            },
            resampling_artifact_path=payload.get("resampling_artifact_path"),
            notes=payload.get("notes"),
            metadata=dict(payload.get("metadata", {})),
        )

    def validate(self) -> None:
        if self.status not in CANDIDATE_STATUSES:
            raise ValueError(f"Unknown candidate status: {self.status}")
        if self.manifest is None:
            raise ValueError("manifest is required")

    def path_value_payload(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(slots=True)
class CandidateIndexEntry:
    candidate_id: str
    display_name: str
    engine_name: str
    engine_family: str | None
    representation_branch: str | None
    status: str
    created_at_utc: str
    tags: list[str]
    checkpoint_path: str
    train_pnl: float | None
    oos_pnl: float | None
    train_trades: int | None
    oos_trades: int | None
    oos_positive: bool | None
    oos_beats_flat: bool | None
    has_trades_artifact: bool
    has_resampling: bool

    @classmethod
    def from_candidate(cls, candidate: CandidateRecord) -> "CandidateIndexEntry":
        train = candidate.periods.get("train")
        oos = candidate.periods.get("oos_adjacent") or candidate.periods.get("oos")
        checkpoint_path = candidate.manifest.checkpoint_path if candidate.manifest else ""
        return cls(
            candidate_id=candidate.candidate_id,
            display_name=candidate.display_name,
            engine_name=candidate.engine_name,
            engine_family=candidate.engine_family,
            representation_branch=candidate.representation_branch,
            status=candidate.status,
            created_at_utc=candidate.created_at_utc,
            tags=list(candidate.tags),
            checkpoint_path=checkpoint_path,
            train_pnl=None if train is None else train.pnl,
            oos_pnl=None if oos is None else oos.pnl,
            train_trades=None if train is None else train.trades,
            oos_trades=None if oos is None else oos.trades,
            oos_positive=None if oos is None else oos.pnl > 0.0,
            oos_beats_flat=candidate.selection_flags.get("oos_beats_flat"),
            has_trades_artifact=bool(candidate.trade_records_path),
            has_resampling=bool(candidate.resampling_results),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuleSetRecord:
    schema_version: str
    name: str
    created_at_utc: str
    description: str | None
    require_all: bool
    rules: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleSetRecord":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            description=payload.get("description"),
            require_all=bool(payload.get("require_all", True)),
            rules=[dict(rule) for rule in payload.get("rules", [])],
        )


@dataclass(slots=True)
class DiversificationPairStats:
    left_candidate_id: str
    right_candidate_id: str
    left_display_name: str
    right_display_name: str
    steps: int
    return_corr: float
    downside_corr: float
    simultaneous_loss_rate: float
    simultaneous_drawdown_rate: float
    action_agreement_rate: float
    same_nonflat_rate: float
    opposite_nonflat_rate: float
    equal_weight_final_balance: float
    equal_weight_net_profit: float
    equal_weight_max_drawdown_pct: float
    avg_individual_max_drawdown_pct: float
    drawdown_improvement_pct_points: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiversificationPairStats":
        return cls(
            left_candidate_id=str(payload["left_candidate_id"]),
            right_candidate_id=str(payload["right_candidate_id"]),
            left_display_name=str(payload["left_display_name"]),
            right_display_name=str(payload["right_display_name"]),
            steps=int(payload["steps"]),
            return_corr=float(payload["return_corr"]),
            downside_corr=float(payload["downside_corr"]),
            simultaneous_loss_rate=float(payload["simultaneous_loss_rate"]),
            simultaneous_drawdown_rate=float(payload["simultaneous_drawdown_rate"]),
            action_agreement_rate=float(payload["action_agreement_rate"]),
            same_nonflat_rate=float(payload["same_nonflat_rate"]),
            opposite_nonflat_rate=float(payload["opposite_nonflat_rate"]),
            equal_weight_final_balance=float(payload["equal_weight_final_balance"]),
            equal_weight_net_profit=float(payload["equal_weight_net_profit"]),
            equal_weight_max_drawdown_pct=float(payload["equal_weight_max_drawdown_pct"]),
            avg_individual_max_drawdown_pct=float(payload["avg_individual_max_drawdown_pct"]),
            drawdown_improvement_pct_points=float(payload["drawdown_improvement_pct_points"]),
        )


@dataclass(slots=True)
class CandidateCurveSnapshot:
    candidate_id: str
    display_name: str
    steps_run: int
    sample_indices: list[int]
    normalized_balance_history: list[float]
    final_balance: float
    max_drawdown_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateCurveSnapshot":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            steps_run=int(payload["steps_run"]),
            sample_indices=[int(item) for item in payload.get("sample_indices", [])],
            normalized_balance_history=[float(item) for item in payload.get("normalized_balance_history", [])],
            final_balance=float(payload["final_balance"]),
            max_drawdown_pct=float(payload["max_drawdown_pct"]),
        )


@dataclass(slots=True)
class DiversificationReport:
    schema_version: str
    name: str
    created_at_utc: str
    data_path: str
    start_utc: str
    end_utc: str
    start_step: int
    max_steps: int
    candidate_ids: list[str]
    candidate_curves: list[CandidateCurveSnapshot]
    pair_stats: list[DiversificationPairStats]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "data_path": self.data_path,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "start_step": self.start_step,
            "max_steps": self.max_steps,
            "candidate_ids": list(self.candidate_ids),
            "candidate_curves": [item.to_dict() for item in self.candidate_curves],
            "pair_stats": [item.to_dict() for item in self.pair_stats],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiversificationReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            data_path=str(payload["data_path"]),
            start_utc=str(payload["start_utc"]),
            end_utc=str(payload["end_utc"]),
            start_step=int(payload["start_step"]),
            max_steps=int(payload["max_steps"]),
            candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
            candidate_curves=[CandidateCurveSnapshot.from_dict(item) for item in payload.get("candidate_curves", [])],
            pair_stats=[DiversificationPairStats.from_dict(item) for item in payload.get("pair_stats", [])],
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class ClusterAssignment:
    candidate_id: str
    display_name: str
    cluster_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClusterAssignment":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            cluster_id=str(payload["cluster_id"]),
        )


@dataclass(slots=True)
class ClusterSummary:
    cluster_id: str
    candidate_ids: list[str]
    display_names: list[str]
    cluster_size: int
    mean_return_corr: float
    mean_downside_corr: float
    mean_simultaneous_loss_rate: float
    mean_similarity_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClusterSummary":
        return cls(
            cluster_id=str(payload["cluster_id"]),
            candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
            display_names=[str(item) for item in payload.get("display_names", [])],
            cluster_size=int(payload["cluster_size"]),
            mean_return_corr=float(payload["mean_return_corr"]),
            mean_downside_corr=float(payload["mean_downside_corr"]),
            mean_simultaneous_loss_rate=float(payload["mean_simultaneous_loss_rate"]),
            mean_similarity_score=float(payload["mean_similarity_score"]),
        )


@dataclass(slots=True)
class ClusterReport:
    schema_version: str
    name: str
    created_at_utc: str
    source_diversification_report: str
    data_path: str
    start_utc: str
    end_utc: str
    start_step: int
    max_steps: int
    similarity_threshold: float
    candidate_ids: list[str]
    assignments: list[ClusterAssignment]
    clusters: list[ClusterSummary]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_diversification_report": self.source_diversification_report,
            "data_path": self.data_path,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "start_step": self.start_step,
            "max_steps": self.max_steps,
            "similarity_threshold": self.similarity_threshold,
            "candidate_ids": list(self.candidate_ids),
            "assignments": [item.to_dict() for item in self.assignments],
            "clusters": [item.to_dict() for item in self.clusters],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClusterReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_diversification_report=str(payload["source_diversification_report"]),
            data_path=str(payload["data_path"]),
            start_utc=str(payload["start_utc"]),
            end_utc=str(payload["end_utc"]),
            start_step=int(payload["start_step"]),
            max_steps=int(payload["max_steps"]),
            similarity_threshold=float(payload["similarity_threshold"]),
            candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
            assignments=[ClusterAssignment.from_dict(item) for item in payload.get("assignments", [])],
            clusters=[ClusterSummary.from_dict(item) for item in payload.get("clusters", [])],
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class CandidateOverride:
    candidate_id: str
    force_include: bool = False
    exclude: bool = False
    pin: bool = False
    max_cap_fraction: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateOverride":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            force_include=bool(payload.get("force_include", False)),
            exclude=bool(payload.get("exclude", False)),
            pin=bool(payload.get("pin", False)),
            max_cap_fraction=None if payload.get("max_cap_fraction") is None else float(payload["max_cap_fraction"]),
            note=payload.get("note"),
        )

    def validate(self) -> None:
        if self.exclude and (self.force_include or self.pin):
            raise ValueError(f"Candidate override {self.candidate_id} cannot exclude and include/pin simultaneously")
        if self.max_cap_fraction is not None and not (0.0 < self.max_cap_fraction <= 1.0):
            raise ValueError(f"Candidate override {self.candidate_id} max_cap_fraction must be in (0, 1]")


@dataclass(slots=True)
class ClusterOverride:
    cluster_id: str
    max_cap_fraction: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClusterOverride":
        return cls(
            cluster_id=str(payload["cluster_id"]),
            max_cap_fraction=None if payload.get("max_cap_fraction") is None else float(payload["max_cap_fraction"]),
            note=payload.get("note"),
        )

    def validate(self) -> None:
        if self.max_cap_fraction is not None and not (0.0 < self.max_cap_fraction <= 1.0):
            raise ValueError(f"Cluster override {self.cluster_id} max_cap_fraction must be in (0, 1]")


@dataclass(slots=True)
class OverrideAuditEntry:
    created_at_utc: str
    actor: str
    target_type: str
    target_id: str
    action: str
    changes: dict[str, Any]
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at_utc": self.created_at_utc,
            "actor": self.actor,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "action": self.action,
            "changes": dict(self.changes),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OverrideAuditEntry":
        return cls(
            created_at_utc=str(payload["created_at_utc"]),
            actor=str(payload["actor"]),
            target_type=str(payload["target_type"]),
            target_id=str(payload["target_id"]),
            action=str(payload["action"]),
            changes=dict(payload.get("changes", {})),
            note=payload.get("note"),
        )


@dataclass(slots=True)
class OverrideSet:
    schema_version: str
    name: str
    created_at_utc: str
    updated_at_utc: str
    source_cluster_report: str | None
    description: str | None
    candidate_overrides: list[CandidateOverride]
    cluster_overrides: list[ClusterOverride]
    audit_entries: list[OverrideAuditEntry]

    def validate(self) -> None:
        for item in self.candidate_overrides:
            item.validate()
        for item in self.cluster_overrides:
            item.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "source_cluster_report": self.source_cluster_report,
            "description": self.description,
            "candidate_overrides": [item.to_dict() for item in self.candidate_overrides],
            "cluster_overrides": [item.to_dict() for item in self.cluster_overrides],
            "audit_entries": [item.to_dict() for item in self.audit_entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OverrideSet":
        obj = cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            updated_at_utc=str(payload["updated_at_utc"]),
            source_cluster_report=payload.get("source_cluster_report"),
            description=payload.get("description"),
            candidate_overrides=[CandidateOverride.from_dict(item) for item in payload.get("candidate_overrides", [])],
            cluster_overrides=[ClusterOverride.from_dict(item) for item in payload.get("cluster_overrides", [])],
            audit_entries=[OverrideAuditEntry.from_dict(item) for item in payload.get("audit_entries", [])],
        )
        obj.validate()
        return obj


@dataclass(slots=True)
class ShortlistCandidateScore:
    candidate_id: str
    display_name: str
    selected: bool
    selected_rank: int | None
    base_score: float
    marginal_score: float | None
    brightness_hint: float
    score_components: dict[str, float]
    exception_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "selected": self.selected,
            "selected_rank": self.selected_rank,
            "base_score": self.base_score,
            "marginal_score": self.marginal_score,
            "brightness_hint": self.brightness_hint,
            "score_components": dict(self.score_components),
            "exception_flags": list(self.exception_flags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShortlistCandidateScore":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            selected=bool(payload["selected"]),
            selected_rank=None if payload.get("selected_rank") is None else int(payload["selected_rank"]),
            base_score=float(payload["base_score"]),
            marginal_score=None if payload.get("marginal_score") is None else float(payload["marginal_score"]),
            brightness_hint=float(payload["brightness_hint"]),
            score_components={str(key): float(value) for key, value in dict(payload.get("score_components", {})).items()},
            exception_flags=[str(item) for item in payload.get("exception_flags", [])],
        )


@dataclass(slots=True)
class ShortlistPairScore:
    left_candidate_id: str
    right_candidate_id: str
    compatibility_score: float
    downside_corr: float
    simultaneous_loss_rate: float
    action_agreement_rate: float
    drawdown_bonus: float
    downside_penalty: float
    simultaneous_loss_penalty: float
    action_agreement_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShortlistPairScore":
        return cls(
            left_candidate_id=str(payload["left_candidate_id"]),
            right_candidate_id=str(payload["right_candidate_id"]),
            compatibility_score=float(payload["compatibility_score"]),
            downside_corr=float(payload["downside_corr"]),
            simultaneous_loss_rate=float(payload["simultaneous_loss_rate"]),
            action_agreement_rate=float(payload["action_agreement_rate"]),
            drawdown_bonus=float(payload["drawdown_bonus"]),
            downside_penalty=float(payload["downside_penalty"]),
            simultaneous_loss_penalty=float(payload["simultaneous_loss_penalty"]),
            action_agreement_penalty=float(payload["action_agreement_penalty"]),
        )


@dataclass(slots=True)
class ShortlistReport:
    schema_version: str
    name: str
    created_at_utc: str
    source_diversification_report: str
    data_path: str
    start_utc: str
    end_utc: str
    start_step: int
    max_steps: int
    resampling_name: str
    candidate_ids: list[str]
    selected_candidate_ids: list[str]
    selection_config: dict[str, Any]
    candidate_scores: list[ShortlistCandidateScore]
    selected_pair_scores: list[ShortlistPairScore]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_diversification_report": self.source_diversification_report,
            "data_path": self.data_path,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "start_step": self.start_step,
            "max_steps": self.max_steps,
            "resampling_name": self.resampling_name,
            "candidate_ids": list(self.candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selection_config": dict(self.selection_config),
            "candidate_scores": [item.to_dict() for item in self.candidate_scores],
            "selected_pair_scores": [item.to_dict() for item in self.selected_pair_scores],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShortlistReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_diversification_report=str(payload["source_diversification_report"]),
            data_path=str(payload["data_path"]),
            start_utc=str(payload["start_utc"]),
            end_utc=str(payload["end_utc"]),
            start_step=int(payload["start_step"]),
            max_steps=int(payload["max_steps"]),
            resampling_name=str(payload["resampling_name"]),
            candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
            selected_candidate_ids=[str(item) for item in payload.get("selected_candidate_ids", [])],
            selection_config=dict(payload.get("selection_config", {})),
            candidate_scores=[ShortlistCandidateScore.from_dict(item) for item in payload.get("candidate_scores", [])],
            selected_pair_scores=[ShortlistPairScore.from_dict(item) for item in payload.get("selected_pair_scores", [])],
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class AllocationWeight:
    candidate_id: str
    display_name: str
    raw_score: float
    normalized_share: float
    capital_fraction: float
    cluster_id: str | None
    capped: bool
    cap_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AllocationWeight":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            raw_score=float(payload["raw_score"]),
            normalized_share=float(payload["normalized_share"]),
            capital_fraction=float(payload["capital_fraction"]),
            cluster_id=payload.get("cluster_id"),
            capped=bool(payload["capped"]),
            cap_reason=payload.get("cap_reason"),
        )


@dataclass(slots=True)
class PortfolioResamplingStats:
    name: str
    iterations: int
    block_size: int
    seed: int
    steps: int
    initial_balance: float
    requested_risk_fraction: float
    allocated_risk_fraction: float
    original_final_balance: float
    original_net_profit: float
    original_max_drawdown_pct: float
    mean_final_balance: float
    median_final_balance: float
    p05_final_balance: float
    p25_final_balance: float
    mean_net_profit: float
    median_net_profit: float
    p05_net_profit: float
    p25_net_profit: float
    mean_max_drawdown_pct: float
    median_max_drawdown_pct: float
    p75_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    profitable_rate: float
    loss_rate: float
    ruin_rate: float
    pessimistic_net_profit: float
    pessimistic_max_drawdown_pct: float
    replay_mode: str = "portfolio_step_block_bootstrap"
    sampler_id: str = "moving_block_step_sampler"
    overlap_policy: str = "time_aligned_step_blocks"
    capital_path_distribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioResamplingStats":
        return cls(
            name=str(payload["name"]),
            iterations=int(payload["iterations"]),
            block_size=int(payload["block_size"]),
            seed=int(payload["seed"]),
            steps=int(payload["steps"]),
            initial_balance=float(payload["initial_balance"]),
            requested_risk_fraction=float(payload["requested_risk_fraction"]),
            allocated_risk_fraction=float(payload["allocated_risk_fraction"]),
            original_final_balance=float(payload["original_final_balance"]),
            original_net_profit=float(payload["original_net_profit"]),
            original_max_drawdown_pct=float(payload["original_max_drawdown_pct"]),
            mean_final_balance=float(payload["mean_final_balance"]),
            median_final_balance=float(payload["median_final_balance"]),
            p05_final_balance=float(payload["p05_final_balance"]),
            p25_final_balance=float(payload["p25_final_balance"]),
            mean_net_profit=float(payload["mean_net_profit"]),
            median_net_profit=float(payload["median_net_profit"]),
            p05_net_profit=float(payload["p05_net_profit"]),
            p25_net_profit=float(payload["p25_net_profit"]),
            mean_max_drawdown_pct=float(payload["mean_max_drawdown_pct"]),
            median_max_drawdown_pct=float(payload["median_max_drawdown_pct"]),
            p75_max_drawdown_pct=float(payload["p75_max_drawdown_pct"]),
            p95_max_drawdown_pct=float(payload["p95_max_drawdown_pct"]),
            profitable_rate=float(payload["profitable_rate"]),
            loss_rate=float(payload["loss_rate"]),
            ruin_rate=float(payload["ruin_rate"]),
            pessimistic_net_profit=float(payload["pessimistic_net_profit"]),
            pessimistic_max_drawdown_pct=float(payload["pessimistic_max_drawdown_pct"]),
            replay_mode=str(payload.get("replay_mode", "portfolio_step_block_bootstrap")),
            sampler_id=str(payload.get("sampler_id", "moving_block_step_sampler")),
            overlap_policy=str(payload.get("overlap_policy", "time_aligned_step_blocks")),
            capital_path_distribution={
                str(key): float(value)
                for key, value in dict(payload.get("capital_path_distribution", {})).items()
            },
        )


@dataclass(slots=True)
class AllocatorScenario:
    name: str
    objective_score: float
    requested_risk_fraction: float
    allocated_risk_fraction: float
    reserve_fraction: float
    per_system_cap_fraction: float
    score_mode: str
    curve_sample_indices: list[int]
    normalized_balance_history: list[float]
    weights: list[AllocationWeight]
    resampling: PortfolioResamplingStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objective_score": self.objective_score,
            "requested_risk_fraction": self.requested_risk_fraction,
            "allocated_risk_fraction": self.allocated_risk_fraction,
            "reserve_fraction": self.reserve_fraction,
            "per_system_cap_fraction": self.per_system_cap_fraction,
            "score_mode": self.score_mode,
            "curve_sample_indices": list(self.curve_sample_indices),
            "normalized_balance_history": list(self.normalized_balance_history),
            "weights": [item.to_dict() for item in self.weights],
            "resampling": self.resampling.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AllocatorScenario":
        return cls(
            name=str(payload["name"]),
            objective_score=float(payload["objective_score"]),
            requested_risk_fraction=float(payload["requested_risk_fraction"]),
            allocated_risk_fraction=float(payload["allocated_risk_fraction"]),
            reserve_fraction=float(payload["reserve_fraction"]),
            per_system_cap_fraction=float(payload["per_system_cap_fraction"]),
            score_mode=str(payload["score_mode"]),
            curve_sample_indices=[int(item) for item in payload.get("curve_sample_indices", [])],
            normalized_balance_history=[float(item) for item in payload.get("normalized_balance_history", [])],
            weights=[AllocationWeight.from_dict(item) for item in payload.get("weights", [])],
            resampling=PortfolioResamplingStats.from_dict(payload["resampling"]),
        )


@dataclass(slots=True)
class AllocatorWorkbenchReport:
    schema_version: str
    name: str
    created_at_utc: str
    source_shortlist_report: str
    source_diversification_report: str
    source_cluster_report: str | None
    source_override_set: str | None
    data_path: str
    start_utc: str
    end_utc: str
    start_step: int
    max_steps: int
    selected_candidate_ids: list[str]
    requested_risk_fractions: list[float]
    chosen_scenario_name: str | None
    objective_config: dict[str, Any]
    scenarios: list[AllocatorScenario]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_shortlist_report": self.source_shortlist_report,
            "source_diversification_report": self.source_diversification_report,
            "source_cluster_report": self.source_cluster_report,
            "source_override_set": self.source_override_set,
            "data_path": self.data_path,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "start_step": self.start_step,
            "max_steps": self.max_steps,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "requested_risk_fractions": list(self.requested_risk_fractions),
            "chosen_scenario_name": self.chosen_scenario_name,
            "objective_config": dict(self.objective_config),
            "scenarios": [item.to_dict() for item in self.scenarios],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AllocatorWorkbenchReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_shortlist_report=str(payload["source_shortlist_report"]),
            source_diversification_report=str(payload["source_diversification_report"]),
            source_cluster_report=payload.get("source_cluster_report"),
            source_override_set=payload.get("source_override_set"),
            data_path=str(payload["data_path"]),
            start_utc=str(payload["start_utc"]),
            end_utc=str(payload["end_utc"]),
            start_step=int(payload["start_step"]),
            max_steps=int(payload["max_steps"]),
            selected_candidate_ids=[str(item) for item in payload.get("selected_candidate_ids", [])],
            requested_risk_fractions=[float(item) for item in payload.get("requested_risk_fractions", [])],
            chosen_scenario_name=payload.get("chosen_scenario_name"),
            objective_config=dict(payload.get("objective_config", {})),
            scenarios=[AllocatorScenario.from_dict(item) for item in payload.get("scenarios", [])],
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class CombinationScenario:
    name: str
    subset_candidate_ids: list[str]
    subset_display_names: list[str]
    subset_size: int
    objective_score: float
    requested_risk_fraction: float
    allocated_risk_fraction: float
    reserve_fraction: float
    score_mode: str
    curve_sample_indices: list[int]
    normalized_balance_history: list[float]
    weights: list[AllocationWeight]
    resampling: PortfolioResamplingStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subset_candidate_ids": list(self.subset_candidate_ids),
            "subset_display_names": list(self.subset_display_names),
            "subset_size": self.subset_size,
            "objective_score": self.objective_score,
            "requested_risk_fraction": self.requested_risk_fraction,
            "allocated_risk_fraction": self.allocated_risk_fraction,
            "reserve_fraction": self.reserve_fraction,
            "score_mode": self.score_mode,
            "curve_sample_indices": list(self.curve_sample_indices),
            "normalized_balance_history": list(self.normalized_balance_history),
            "weights": [item.to_dict() for item in self.weights],
            "resampling": self.resampling.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CombinationScenario":
        return cls(
            name=str(payload["name"]),
            subset_candidate_ids=[str(item) for item in payload.get("subset_candidate_ids", [])],
            subset_display_names=[str(item) for item in payload.get("subset_display_names", [])],
            subset_size=int(payload["subset_size"]),
            objective_score=float(payload["objective_score"]),
            requested_risk_fraction=float(payload["requested_risk_fraction"]),
            allocated_risk_fraction=float(payload["allocated_risk_fraction"]),
            reserve_fraction=float(payload["reserve_fraction"]),
            score_mode=str(payload["score_mode"]),
            curve_sample_indices=[int(item) for item in payload.get("curve_sample_indices", [])],
            normalized_balance_history=[float(item) for item in payload.get("normalized_balance_history", [])],
            weights=[AllocationWeight.from_dict(item) for item in payload.get("weights", [])],
            resampling=PortfolioResamplingStats.from_dict(payload["resampling"]),
        )


@dataclass(slots=True)
class CombinationSearchReport:
    schema_version: str
    name: str
    created_at_utc: str
    source_shortlist_report: str
    source_diversification_report: str
    source_cluster_report: str | None
    source_override_set: str | None
    data_path: str
    start_utc: str
    end_utc: str
    start_step: int
    max_steps: int
    pool_candidate_ids: list[str]
    searched_subset_sizes: list[int]
    evaluated_combination_count: int
    evaluated_scenario_count: int
    best_scenario_name: str | None
    objective_config: dict[str, Any]
    scenarios: list[CombinationScenario]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_shortlist_report": self.source_shortlist_report,
            "source_diversification_report": self.source_diversification_report,
            "source_cluster_report": self.source_cluster_report,
            "source_override_set": self.source_override_set,
            "data_path": self.data_path,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "start_step": self.start_step,
            "max_steps": self.max_steps,
            "pool_candidate_ids": list(self.pool_candidate_ids),
            "searched_subset_sizes": list(self.searched_subset_sizes),
            "evaluated_combination_count": self.evaluated_combination_count,
            "evaluated_scenario_count": self.evaluated_scenario_count,
            "best_scenario_name": self.best_scenario_name,
            "objective_config": dict(self.objective_config),
            "scenarios": [item.to_dict() for item in self.scenarios],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CombinationSearchReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_shortlist_report=str(payload["source_shortlist_report"]),
            source_diversification_report=str(payload["source_diversification_report"]),
            source_cluster_report=payload.get("source_cluster_report"),
            source_override_set=payload.get("source_override_set"),
            data_path=str(payload["data_path"]),
            start_utc=str(payload["start_utc"]),
            end_utc=str(payload["end_utc"]),
            start_step=int(payload["start_step"]),
            max_steps=int(payload["max_steps"]),
            pool_candidate_ids=[str(item) for item in payload.get("pool_candidate_ids", [])],
            searched_subset_sizes=[int(item) for item in payload.get("searched_subset_sizes", [])],
            evaluated_combination_count=int(payload["evaluated_combination_count"]),
            evaluated_scenario_count=int(payload["evaluated_scenario_count"]),
            best_scenario_name=payload.get("best_scenario_name"),
            objective_config=dict(payload.get("objective_config", {})),
            scenarios=[CombinationScenario.from_dict(item) for item in payload.get("scenarios", [])],
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class CycleStepRecord:
    name: str
    command: list[str]
    status: str
    log_path: str | None = None
    artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "status": self.status,
            "log_path": self.log_path,
            "artifact": self.artifact,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CycleStepRecord":
        return cls(
            name=str(payload["name"]),
            command=[str(item) for item in payload.get("command", [])],
            status=str(payload["status"]),
            log_path=payload.get("log_path"),
            artifact=payload.get("artifact"),
        )


@dataclass(slots=True)
class ContinuousSearchCycleReport:
    schema_version: str
    name: str
    created_at_utc: str
    mode: str
    output_dir: str
    cycle_tag: str | None
    source_summary_path: str | None
    candidate_ids: list[str]
    report_names: dict[str, str]
    steps: list[CycleStepRecord]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "mode": self.mode,
            "output_dir": self.output_dir,
            "cycle_tag": self.cycle_tag,
            "source_summary_path": self.source_summary_path,
            "candidate_ids": list(self.candidate_ids),
            "report_names": dict(self.report_names),
            "steps": [item.to_dict() for item in self.steps],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContinuousSearchCycleReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            mode=str(payload["mode"]),
            output_dir=str(payload["output_dir"]),
            cycle_tag=payload.get("cycle_tag"),
            source_summary_path=payload.get("source_summary_path"),
            candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
            report_names={str(key): str(value) for key, value in dict(payload.get("report_names", {})).items()},
            steps=[CycleStepRecord.from_dict(item) for item in payload.get("steps", [])],
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class TradeforwardAllocation:
    candidate_id: str
    display_name: str
    checkpoint_path: str
    engine_name: str
    representation_name: str
    status: str
    cluster_id: str | None
    normalized_share: float
    capital_fraction: float
    capped: bool
    cap_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeforwardAllocation":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            engine_name=str(payload["engine_name"]),
            representation_name=str(payload["representation_name"]),
            status=str(payload["status"]),
            cluster_id=payload.get("cluster_id"),
            normalized_share=float(payload["normalized_share"]),
            capital_fraction=float(payload["capital_fraction"]),
            capped=bool(payload["capped"]),
            cap_reason=payload.get("cap_reason"),
        )


@dataclass(slots=True)
class TradeforwardPlan:
    schema_version: str
    name: str
    created_at_utc: str
    selection_mode: str
    scenario_name: str
    data_path: str
    forward_start_utc: str
    forward_end_utc: str
    forward_start_step: int | None
    forward_max_steps: int | None
    requested_risk_fraction: float
    allocated_risk_fraction: float
    reserve_fraction: float
    candidate_ids: list[str]
    allocations: list[TradeforwardAllocation]
    source_cycle_report: str | None = None
    source_allocator_report: str | None = None
    source_combination_report: str | None = None
    source_cluster_report: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "selection_mode": self.selection_mode,
            "scenario_name": self.scenario_name,
            "data_path": self.data_path,
            "forward_start_utc": self.forward_start_utc,
            "forward_end_utc": self.forward_end_utc,
            "forward_start_step": self.forward_start_step,
            "forward_max_steps": self.forward_max_steps,
            "requested_risk_fraction": self.requested_risk_fraction,
            "allocated_risk_fraction": self.allocated_risk_fraction,
            "reserve_fraction": self.reserve_fraction,
            "candidate_ids": list(self.candidate_ids),
            "allocations": [item.to_dict() for item in self.allocations],
            "source_cycle_report": self.source_cycle_report,
            "source_allocator_report": self.source_allocator_report,
            "source_combination_report": self.source_combination_report,
            "source_cluster_report": self.source_cluster_report,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeforwardPlan":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            selection_mode=str(payload["selection_mode"]),
            scenario_name=str(payload["scenario_name"]),
            data_path=str(payload["data_path"]),
            forward_start_utc=str(payload["forward_start_utc"]),
            forward_end_utc=str(payload["forward_end_utc"]),
            forward_start_step=None if payload.get("forward_start_step") is None else int(payload["forward_start_step"]),
            forward_max_steps=None if payload.get("forward_max_steps") is None else int(payload["forward_max_steps"]),
            requested_risk_fraction=float(payload["requested_risk_fraction"]),
            allocated_risk_fraction=float(payload["allocated_risk_fraction"]),
            reserve_fraction=float(payload["reserve_fraction"]),
            candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
            allocations=[TradeforwardAllocation.from_dict(item) for item in payload.get("allocations", [])],
            source_cycle_report=payload.get("source_cycle_report"),
            source_allocator_report=payload.get("source_allocator_report"),
            source_combination_report=payload.get("source_combination_report"),
            source_cluster_report=payload.get("source_cluster_report"),
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class TradeforwardExpectation:
    selection_mode: str
    scenario_name: str
    objective_score: float
    expected_original_net_profit: float
    expected_original_max_drawdown_pct: float
    expected_p05_net_profit: float
    expected_p95_max_drawdown_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeforwardExpectation":
        return cls(
            selection_mode=str(payload["selection_mode"]),
            scenario_name=str(payload["scenario_name"]),
            objective_score=float(payload["objective_score"]),
            expected_original_net_profit=float(payload["expected_original_net_profit"]),
            expected_original_max_drawdown_pct=float(payload["expected_original_max_drawdown_pct"]),
            expected_p05_net_profit=float(payload["expected_p05_net_profit"]),
            expected_p95_max_drawdown_pct=float(payload["expected_p95_max_drawdown_pct"]),
        )


@dataclass(slots=True)
class TradeforwardCandidateEvaluation:
    candidate_id: str
    display_name: str
    cluster_id: str | None
    checkpoint_path: str
    capital_fraction: float
    normalized_share: float
    capped: bool
    cap_reason: str | None
    final_balance: float
    pnl: float
    max_drawdown_pct: float
    trades: int
    wins: int
    win_rate_pct: float
    curve_sample_indices: list[int]
    normalized_balance_history: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "cluster_id": self.cluster_id,
            "checkpoint_path": self.checkpoint_path,
            "capital_fraction": self.capital_fraction,
            "normalized_share": self.normalized_share,
            "capped": self.capped,
            "cap_reason": self.cap_reason,
            "final_balance": self.final_balance,
            "pnl": self.pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate_pct": self.win_rate_pct,
            "curve_sample_indices": list(self.curve_sample_indices),
            "normalized_balance_history": list(self.normalized_balance_history),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeforwardCandidateEvaluation":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            cluster_id=payload.get("cluster_id"),
            checkpoint_path=str(payload["checkpoint_path"]),
            capital_fraction=float(payload["capital_fraction"]),
            normalized_share=float(payload["normalized_share"]),
            capped=bool(payload["capped"]),
            cap_reason=payload.get("cap_reason"),
            final_balance=float(payload["final_balance"]),
            pnl=float(payload["pnl"]),
            max_drawdown_pct=float(payload["max_drawdown_pct"]),
            trades=int(payload["trades"]),
            wins=int(payload["wins"]),
            win_rate_pct=float(payload["win_rate_pct"]),
            curve_sample_indices=[int(item) for item in payload.get("curve_sample_indices", [])],
            normalized_balance_history=[float(item) for item in payload.get("normalized_balance_history", [])],
        )


@dataclass(slots=True)
class TradeforwardPortfolioEvaluation:
    initial_balance: float
    final_balance: float
    pnl: float
    max_drawdown_pct: float
    requested_risk_fraction: float
    allocated_risk_fraction: float
    reserve_fraction: float
    component_trade_count_total: int
    component_win_count_total: int
    curve_sample_indices: list[int]
    normalized_balance_history: list[float]
    actual_minus_expected_original_net_profit: float | None = None
    actual_minus_expected_original_max_drawdown_pct: float | None = None
    actual_minus_expected_p05_net_profit: float | None = None
    actual_minus_expected_p95_max_drawdown_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "pnl": self.pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "requested_risk_fraction": self.requested_risk_fraction,
            "allocated_risk_fraction": self.allocated_risk_fraction,
            "reserve_fraction": self.reserve_fraction,
            "component_trade_count_total": self.component_trade_count_total,
            "component_win_count_total": self.component_win_count_total,
            "curve_sample_indices": list(self.curve_sample_indices),
            "normalized_balance_history": list(self.normalized_balance_history),
            "actual_minus_expected_original_net_profit": self.actual_minus_expected_original_net_profit,
            "actual_minus_expected_original_max_drawdown_pct": self.actual_minus_expected_original_max_drawdown_pct,
            "actual_minus_expected_p05_net_profit": self.actual_minus_expected_p05_net_profit,
            "actual_minus_expected_p95_max_drawdown_pct": self.actual_minus_expected_p95_max_drawdown_pct,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeforwardPortfolioEvaluation":
        return cls(
            initial_balance=float(payload["initial_balance"]),
            final_balance=float(payload["final_balance"]),
            pnl=float(payload["pnl"]),
            max_drawdown_pct=float(payload["max_drawdown_pct"]),
            requested_risk_fraction=float(payload["requested_risk_fraction"]),
            allocated_risk_fraction=float(payload["allocated_risk_fraction"]),
            reserve_fraction=float(payload["reserve_fraction"]),
            component_trade_count_total=int(payload["component_trade_count_total"]),
            component_win_count_total=int(payload["component_win_count_total"]),
            curve_sample_indices=[int(item) for item in payload.get("curve_sample_indices", [])],
            normalized_balance_history=[float(item) for item in payload.get("normalized_balance_history", [])],
            actual_minus_expected_original_net_profit=None
            if payload.get("actual_minus_expected_original_net_profit") is None
            else float(payload["actual_minus_expected_original_net_profit"]),
            actual_minus_expected_original_max_drawdown_pct=None
            if payload.get("actual_minus_expected_original_max_drawdown_pct") is None
            else float(payload["actual_minus_expected_original_max_drawdown_pct"]),
            actual_minus_expected_p05_net_profit=None
            if payload.get("actual_minus_expected_p05_net_profit") is None
            else float(payload["actual_minus_expected_p05_net_profit"]),
            actual_minus_expected_p95_max_drawdown_pct=None
            if payload.get("actual_minus_expected_p95_max_drawdown_pct") is None
            else float(payload["actual_minus_expected_p95_max_drawdown_pct"]),
        )


@dataclass(slots=True)
class TradeforwardEvaluationReport:
    schema_version: str
    name: str
    created_at_utc: str
    source_plan: str
    selection_mode: str
    scenario_name: str
    data_path: str
    forward_start_utc: str
    forward_end_utc: str
    forward_start_step: int | None
    forward_max_steps: int | None
    candidate_ids: list[str]
    expectation: TradeforwardExpectation | None
    portfolio: TradeforwardPortfolioEvaluation
    candidate_evaluations: list[TradeforwardCandidateEvaluation]
    source_cycle_report: str | None = None
    source_allocator_report: str | None = None
    source_combination_report: str | None = None
    source_cluster_report: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_plan": self.source_plan,
            "selection_mode": self.selection_mode,
            "scenario_name": self.scenario_name,
            "data_path": self.data_path,
            "forward_start_utc": self.forward_start_utc,
            "forward_end_utc": self.forward_end_utc,
            "forward_start_step": self.forward_start_step,
            "forward_max_steps": self.forward_max_steps,
            "candidate_ids": list(self.candidate_ids),
            "expectation": None if self.expectation is None else self.expectation.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "candidate_evaluations": [item.to_dict() for item in self.candidate_evaluations],
            "source_cycle_report": self.source_cycle_report,
            "source_allocator_report": self.source_allocator_report,
            "source_combination_report": self.source_combination_report,
            "source_cluster_report": self.source_cluster_report,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TradeforwardEvaluationReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_plan=str(payload["source_plan"]),
            selection_mode=str(payload["selection_mode"]),
            scenario_name=str(payload["scenario_name"]),
            data_path=str(payload["data_path"]),
            forward_start_utc=str(payload["forward_start_utc"]),
            forward_end_utc=str(payload["forward_end_utc"]),
            forward_start_step=None if payload.get("forward_start_step") is None else int(payload["forward_start_step"]),
            forward_max_steps=None if payload.get("forward_max_steps") is None else int(payload["forward_max_steps"]),
            candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
            expectation=None if payload.get("expectation") is None else TradeforwardExpectation.from_dict(payload["expectation"]),
            portfolio=TradeforwardPortfolioEvaluation.from_dict(payload["portfolio"]),
            candidate_evaluations=[
                TradeforwardCandidateEvaluation.from_dict(item)
                for item in payload.get("candidate_evaluations", [])
            ],
            source_cycle_report=payload.get("source_cycle_report"),
            source_allocator_report=payload.get("source_allocator_report"),
            source_combination_report=payload.get("source_combination_report"),
            source_cluster_report=payload.get("source_cluster_report"),
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class RollingCycleOutcome:
    cycle_index: int
    cycle_name: str
    cycle_report_name: str
    tradeforward_plan_name: str
    tradeforward_evaluation_name: str
    selection_start_utc: str
    selection_end_utc: str
    forward_start_utc: str
    forward_end_utc: str
    candidate_ids: list[str]
    selected_candidate_ids: list[str]
    requested_risk_fraction: float
    allocated_risk_fraction: float
    reserve_fraction: float
    portfolio_pnl: float
    portfolio_final_balance: float
    portfolio_max_drawdown_pct: float
    cycle_return_fraction: float
    ledger_balance_before: float
    ledger_balance_after: float
    actual_minus_expected_original_net_profit: float | None = None
    actual_minus_expected_original_max_drawdown_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "cycle_name": self.cycle_name,
            "cycle_report_name": self.cycle_report_name,
            "tradeforward_plan_name": self.tradeforward_plan_name,
            "tradeforward_evaluation_name": self.tradeforward_evaluation_name,
            "selection_start_utc": self.selection_start_utc,
            "selection_end_utc": self.selection_end_utc,
            "forward_start_utc": self.forward_start_utc,
            "forward_end_utc": self.forward_end_utc,
            "candidate_ids": list(self.candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "requested_risk_fraction": self.requested_risk_fraction,
            "allocated_risk_fraction": self.allocated_risk_fraction,
            "reserve_fraction": self.reserve_fraction,
            "portfolio_pnl": self.portfolio_pnl,
            "portfolio_final_balance": self.portfolio_final_balance,
            "portfolio_max_drawdown_pct": self.portfolio_max_drawdown_pct,
            "cycle_return_fraction": self.cycle_return_fraction,
            "ledger_balance_before": self.ledger_balance_before,
            "ledger_balance_after": self.ledger_balance_after,
            "actual_minus_expected_original_net_profit": self.actual_minus_expected_original_net_profit,
            "actual_minus_expected_original_max_drawdown_pct": self.actual_minus_expected_original_max_drawdown_pct,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RollingCycleOutcome":
        return cls(
            cycle_index=int(payload["cycle_index"]),
            cycle_name=str(payload["cycle_name"]),
            cycle_report_name=str(payload["cycle_report_name"]),
            tradeforward_plan_name=str(payload["tradeforward_plan_name"]),
            tradeforward_evaluation_name=str(payload["tradeforward_evaluation_name"]),
            selection_start_utc=str(payload["selection_start_utc"]),
            selection_end_utc=str(payload["selection_end_utc"]),
            forward_start_utc=str(payload["forward_start_utc"]),
            forward_end_utc=str(payload["forward_end_utc"]),
            candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
            selected_candidate_ids=[str(item) for item in payload.get("selected_candidate_ids", [])],
            requested_risk_fraction=float(payload["requested_risk_fraction"]),
            allocated_risk_fraction=float(payload["allocated_risk_fraction"]),
            reserve_fraction=float(payload["reserve_fraction"]),
            portfolio_pnl=float(payload["portfolio_pnl"]),
            portfolio_final_balance=float(payload["portfolio_final_balance"]),
            portfolio_max_drawdown_pct=float(payload["portfolio_max_drawdown_pct"]),
            cycle_return_fraction=float(payload["cycle_return_fraction"]),
            ledger_balance_before=float(payload["ledger_balance_before"]),
            ledger_balance_after=float(payload["ledger_balance_after"]),
            actual_minus_expected_original_net_profit=None
            if payload.get("actual_minus_expected_original_net_profit") is None
            else float(payload["actual_minus_expected_original_net_profit"]),
            actual_minus_expected_original_max_drawdown_pct=None
            if payload.get("actual_minus_expected_original_max_drawdown_pct") is None
            else float(payload["actual_minus_expected_original_max_drawdown_pct"]),
        )


@dataclass(slots=True)
class RollingConveyorReport:
    schema_version: str
    name: str
    created_at_utc: str
    mode: str
    selection_days: int
    forward_days: int
    step_days: int
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    positive_cycle_count: int
    evaluated_cycle_count: int
    ledger_cycle_indices: list[int]
    ledger_balance_history: list[float]
    cycle_outcomes: list[RollingCycleOutcome]
    selector: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "mode": self.mode,
            "selection_days": self.selection_days,
            "forward_days": self.forward_days,
            "step_days": self.step_days,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "positive_cycle_count": self.positive_cycle_count,
            "evaluated_cycle_count": self.evaluated_cycle_count,
            "ledger_cycle_indices": list(self.ledger_cycle_indices),
            "ledger_balance_history": list(self.ledger_balance_history),
            "cycle_outcomes": [item.to_dict() for item in self.cycle_outcomes],
            "selector": dict(self.selector),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RollingConveyorReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            mode=str(payload["mode"]),
            selection_days=int(payload["selection_days"]),
            forward_days=int(payload["forward_days"]),
            step_days=int(payload["step_days"]),
            initial_balance=float(payload["initial_balance"]),
            final_balance=float(payload["final_balance"]),
            total_pnl=float(payload["total_pnl"]),
            total_return_pct=float(payload["total_return_pct"]),
            max_drawdown_pct=float(payload["max_drawdown_pct"]),
            positive_cycle_count=int(payload["positive_cycle_count"]),
            evaluated_cycle_count=int(payload["evaluated_cycle_count"]),
            ledger_cycle_indices=[int(item) for item in payload.get("ledger_cycle_indices", [])],
            ledger_balance_history=[float(item) for item in payload.get("ledger_balance_history", [])],
            cycle_outcomes=[RollingCycleOutcome.from_dict(item) for item in payload.get("cycle_outcomes", [])],
            selector=dict(payload.get("selector", {})),
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class LifecycleDecisionRecord:
    cycle_index: int
    cycle_name: str
    candidate_id: str
    display_name: str
    selected: bool
    previous_status: str
    next_status: str
    action: str
    reason_codes: list[str]
    trades: int | None = None
    pnl: float | None = None
    max_drawdown_pct: float | None = None
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    consecutive_idle_cycles: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "cycle_name": self.cycle_name,
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "selected": self.selected,
            "previous_status": self.previous_status,
            "next_status": self.next_status,
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "trades": self.trades,
            "pnl": self.pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_idle_cycles": self.consecutive_idle_cycles,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleDecisionRecord":
        return cls(
            cycle_index=int(payload["cycle_index"]),
            cycle_name=str(payload["cycle_name"]),
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            selected=bool(payload["selected"]),
            previous_status=str(payload["previous_status"]),
            next_status=str(payload["next_status"]),
            action=str(payload["action"]),
            reason_codes=[str(item) for item in payload.get("reason_codes", [])],
            trades=None if payload.get("trades") is None else int(payload["trades"]),
            pnl=None if payload.get("pnl") is None else float(payload["pnl"]),
            max_drawdown_pct=None
            if payload.get("max_drawdown_pct") is None
            else float(payload["max_drawdown_pct"]),
            consecutive_successes=int(payload.get("consecutive_successes", 0)),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
            consecutive_idle_cycles=int(payload.get("consecutive_idle_cycles", 0)),
        )


@dataclass(slots=True)
class LifecycleCandidateSummary:
    candidate_id: str
    display_name: str
    initial_status: str
    final_status: str
    total_cycles_seen: int
    selected_cycles: int
    successful_forward_cycles: int
    failed_forward_cycles: int
    no_trade_cycles: int
    idle_cycles: int
    transition_count: int
    last_selected_cycle_index: int | None = None
    last_transition_cycle_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "initial_status": self.initial_status,
            "final_status": self.final_status,
            "total_cycles_seen": self.total_cycles_seen,
            "selected_cycles": self.selected_cycles,
            "successful_forward_cycles": self.successful_forward_cycles,
            "failed_forward_cycles": self.failed_forward_cycles,
            "no_trade_cycles": self.no_trade_cycles,
            "idle_cycles": self.idle_cycles,
            "transition_count": self.transition_count,
            "last_selected_cycle_index": self.last_selected_cycle_index,
            "last_transition_cycle_index": self.last_transition_cycle_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleCandidateSummary":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            initial_status=str(payload["initial_status"]),
            final_status=str(payload["final_status"]),
            total_cycles_seen=int(payload["total_cycles_seen"]),
            selected_cycles=int(payload["selected_cycles"]),
            successful_forward_cycles=int(payload["successful_forward_cycles"]),
            failed_forward_cycles=int(payload["failed_forward_cycles"]),
            no_trade_cycles=int(payload["no_trade_cycles"]),
            idle_cycles=int(payload["idle_cycles"]),
            transition_count=int(payload["transition_count"]),
            last_selected_cycle_index=None
            if payload.get("last_selected_cycle_index") is None
            else int(payload["last_selected_cycle_index"]),
            last_transition_cycle_index=None
            if payload.get("last_transition_cycle_index") is None
            else int(payload["last_transition_cycle_index"]),
        )


@dataclass(slots=True)
class LifecycleReport:
    schema_version: str
    name: str
    created_at_utc: str
    source_rolling_report: str
    source_tradeforward_evaluations: list[str]
    config: dict[str, Any]
    candidate_ids: list[str]
    applied_status_updates: bool
    final_status_counts: dict[str, int]
    candidate_summaries: list[LifecycleCandidateSummary]
    decisions: list[LifecycleDecisionRecord]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_rolling_report": self.source_rolling_report,
            "source_tradeforward_evaluations": list(self.source_tradeforward_evaluations),
            "config": dict(self.config),
            "candidate_ids": list(self.candidate_ids),
            "applied_status_updates": self.applied_status_updates,
            "final_status_counts": {str(key): int(value) for key, value in self.final_status_counts.items()},
            "candidate_summaries": [item.to_dict() for item in self.candidate_summaries],
            "decisions": [item.to_dict() for item in self.decisions],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_rolling_report=str(payload["source_rolling_report"]),
            source_tradeforward_evaluations=[str(item) for item in payload.get("source_tradeforward_evaluations", [])],
            config=dict(payload.get("config", {})),
            candidate_ids=[str(item) for item in payload.get("candidate_ids", [])],
            applied_status_updates=bool(payload["applied_status_updates"]),
            final_status_counts={str(key): int(value) for key, value in dict(payload.get("final_status_counts", {})).items()},
            candidate_summaries=[
                LifecycleCandidateSummary.from_dict(item) for item in payload.get("candidate_summaries", [])
            ],
            decisions=[LifecycleDecisionRecord.from_dict(item) for item in payload.get("decisions", [])],
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class PortfolioRebalanceChange:
    candidate_id: str
    display_name: str
    cluster_id: str | None
    previous_capital_fraction: float
    target_capital_fraction: float
    delta_capital_fraction: float
    rebalance_action: str
    status_before_cycle: str | None = None
    status_after_cycle: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "cluster_id": self.cluster_id,
            "previous_capital_fraction": self.previous_capital_fraction,
            "target_capital_fraction": self.target_capital_fraction,
            "delta_capital_fraction": self.delta_capital_fraction,
            "rebalance_action": self.rebalance_action,
            "status_before_cycle": self.status_before_cycle,
            "status_after_cycle": self.status_after_cycle,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioRebalanceChange":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            cluster_id=payload.get("cluster_id"),
            previous_capital_fraction=float(payload["previous_capital_fraction"]),
            target_capital_fraction=float(payload["target_capital_fraction"]),
            delta_capital_fraction=float(payload["delta_capital_fraction"]),
            rebalance_action=str(payload["rebalance_action"]),
            status_before_cycle=payload.get("status_before_cycle"),
            status_after_cycle=payload.get("status_after_cycle"),
        )


@dataclass(slots=True)
class PortfolioCycleLedgerEntry:
    cycle_index: int
    cycle_name: str
    selection_start_utc: str
    selection_end_utc: str
    forward_start_utc: str
    forward_end_utc: str
    tradeforward_plan_name: str
    tradeforward_evaluation_name: str
    ledger_balance_before: float
    ledger_balance_after_rebalance: float
    ledger_balance_after_cycle: float
    gross_cycle_pnl: float
    net_cycle_pnl: float
    portfolio_max_drawdown_pct: float
    requested_risk_fraction: float
    allocated_risk_fraction: float
    reserve_fraction_before: float
    reserve_fraction_after: float
    buy_turnover_fraction: float
    sell_turnover_fraction: float
    gross_turnover_fraction: float
    estimated_rebalance_cost: float
    previous_active_candidate_ids: list[str]
    target_candidate_ids: list[str]
    added_candidate_ids: list[str]
    removed_candidate_ids: list[str]
    increased_candidate_ids: list[str]
    decreased_candidate_ids: list[str]
    unchanged_candidate_ids: list[str]
    non_tradable_selected_candidate_ids: list[str]
    status_counts_before: dict[str, int]
    status_counts_after: dict[str, int]
    cluster_exposure_before: dict[str, float]
    cluster_exposure_after: dict[str, float]
    rebalance_changes: list[PortfolioRebalanceChange]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "cycle_name": self.cycle_name,
            "selection_start_utc": self.selection_start_utc,
            "selection_end_utc": self.selection_end_utc,
            "forward_start_utc": self.forward_start_utc,
            "forward_end_utc": self.forward_end_utc,
            "tradeforward_plan_name": self.tradeforward_plan_name,
            "tradeforward_evaluation_name": self.tradeforward_evaluation_name,
            "ledger_balance_before": self.ledger_balance_before,
            "ledger_balance_after_rebalance": self.ledger_balance_after_rebalance,
            "ledger_balance_after_cycle": self.ledger_balance_after_cycle,
            "gross_cycle_pnl": self.gross_cycle_pnl,
            "net_cycle_pnl": self.net_cycle_pnl,
            "portfolio_max_drawdown_pct": self.portfolio_max_drawdown_pct,
            "requested_risk_fraction": self.requested_risk_fraction,
            "allocated_risk_fraction": self.allocated_risk_fraction,
            "reserve_fraction_before": self.reserve_fraction_before,
            "reserve_fraction_after": self.reserve_fraction_after,
            "buy_turnover_fraction": self.buy_turnover_fraction,
            "sell_turnover_fraction": self.sell_turnover_fraction,
            "gross_turnover_fraction": self.gross_turnover_fraction,
            "estimated_rebalance_cost": self.estimated_rebalance_cost,
            "previous_active_candidate_ids": list(self.previous_active_candidate_ids),
            "target_candidate_ids": list(self.target_candidate_ids),
            "added_candidate_ids": list(self.added_candidate_ids),
            "removed_candidate_ids": list(self.removed_candidate_ids),
            "increased_candidate_ids": list(self.increased_candidate_ids),
            "decreased_candidate_ids": list(self.decreased_candidate_ids),
            "unchanged_candidate_ids": list(self.unchanged_candidate_ids),
            "non_tradable_selected_candidate_ids": list(self.non_tradable_selected_candidate_ids),
            "status_counts_before": {str(key): int(value) for key, value in self.status_counts_before.items()},
            "status_counts_after": {str(key): int(value) for key, value in self.status_counts_after.items()},
            "cluster_exposure_before": {str(key): float(value) for key, value in self.cluster_exposure_before.items()},
            "cluster_exposure_after": {str(key): float(value) for key, value in self.cluster_exposure_after.items()},
            "rebalance_changes": [item.to_dict() for item in self.rebalance_changes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioCycleLedgerEntry":
        return cls(
            cycle_index=int(payload["cycle_index"]),
            cycle_name=str(payload["cycle_name"]),
            selection_start_utc=str(payload["selection_start_utc"]),
            selection_end_utc=str(payload["selection_end_utc"]),
            forward_start_utc=str(payload["forward_start_utc"]),
            forward_end_utc=str(payload["forward_end_utc"]),
            tradeforward_plan_name=str(payload["tradeforward_plan_name"]),
            tradeforward_evaluation_name=str(payload["tradeforward_evaluation_name"]),
            ledger_balance_before=float(payload["ledger_balance_before"]),
            ledger_balance_after_rebalance=float(payload["ledger_balance_after_rebalance"]),
            ledger_balance_after_cycle=float(payload["ledger_balance_after_cycle"]),
            gross_cycle_pnl=float(payload["gross_cycle_pnl"]),
            net_cycle_pnl=float(payload["net_cycle_pnl"]),
            portfolio_max_drawdown_pct=float(payload["portfolio_max_drawdown_pct"]),
            requested_risk_fraction=float(payload["requested_risk_fraction"]),
            allocated_risk_fraction=float(payload["allocated_risk_fraction"]),
            reserve_fraction_before=float(payload["reserve_fraction_before"]),
            reserve_fraction_after=float(payload["reserve_fraction_after"]),
            buy_turnover_fraction=float(payload["buy_turnover_fraction"]),
            sell_turnover_fraction=float(payload["sell_turnover_fraction"]),
            gross_turnover_fraction=float(payload["gross_turnover_fraction"]),
            estimated_rebalance_cost=float(payload["estimated_rebalance_cost"]),
            previous_active_candidate_ids=[str(item) for item in payload.get("previous_active_candidate_ids", [])],
            target_candidate_ids=[str(item) for item in payload.get("target_candidate_ids", [])],
            added_candidate_ids=[str(item) for item in payload.get("added_candidate_ids", [])],
            removed_candidate_ids=[str(item) for item in payload.get("removed_candidate_ids", [])],
            increased_candidate_ids=[str(item) for item in payload.get("increased_candidate_ids", [])],
            decreased_candidate_ids=[str(item) for item in payload.get("decreased_candidate_ids", [])],
            unchanged_candidate_ids=[str(item) for item in payload.get("unchanged_candidate_ids", [])],
            non_tradable_selected_candidate_ids=[
                str(item) for item in payload.get("non_tradable_selected_candidate_ids", [])
            ],
            status_counts_before={str(key): int(value) for key, value in dict(payload.get("status_counts_before", {})).items()},
            status_counts_after={str(key): int(value) for key, value in dict(payload.get("status_counts_after", {})).items()},
            cluster_exposure_before={
                str(key): float(value) for key, value in dict(payload.get("cluster_exposure_before", {})).items()
            },
            cluster_exposure_after={
                str(key): float(value) for key, value in dict(payload.get("cluster_exposure_after", {})).items()
            },
            rebalance_changes=[PortfolioRebalanceChange.from_dict(item) for item in payload.get("rebalance_changes", [])],
        )


@dataclass(slots=True)
class PortfolioCandidateLedgerSummary:
    candidate_id: str
    display_name: str
    final_status: str
    selected_cycles: int
    added_cycles: int
    removed_cycles: int
    increased_cycles: int
    decreased_cycles: int
    unchanged_cycles: int
    gross_target_capital_fraction: float
    average_target_capital_fraction: float
    max_target_capital_fraction: float
    ending_capital_fraction: float
    first_selected_cycle_index: int | None = None
    last_selected_cycle_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_name": self.display_name,
            "final_status": self.final_status,
            "selected_cycles": self.selected_cycles,
            "added_cycles": self.added_cycles,
            "removed_cycles": self.removed_cycles,
            "increased_cycles": self.increased_cycles,
            "decreased_cycles": self.decreased_cycles,
            "unchanged_cycles": self.unchanged_cycles,
            "gross_target_capital_fraction": self.gross_target_capital_fraction,
            "average_target_capital_fraction": self.average_target_capital_fraction,
            "max_target_capital_fraction": self.max_target_capital_fraction,
            "ending_capital_fraction": self.ending_capital_fraction,
            "first_selected_cycle_index": self.first_selected_cycle_index,
            "last_selected_cycle_index": self.last_selected_cycle_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioCandidateLedgerSummary":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_name=str(payload["display_name"]),
            final_status=str(payload["final_status"]),
            selected_cycles=int(payload["selected_cycles"]),
            added_cycles=int(payload["added_cycles"]),
            removed_cycles=int(payload["removed_cycles"]),
            increased_cycles=int(payload["increased_cycles"]),
            decreased_cycles=int(payload["decreased_cycles"]),
            unchanged_cycles=int(payload["unchanged_cycles"]),
            gross_target_capital_fraction=float(payload["gross_target_capital_fraction"]),
            average_target_capital_fraction=float(payload["average_target_capital_fraction"]),
            max_target_capital_fraction=float(payload["max_target_capital_fraction"]),
            ending_capital_fraction=float(payload["ending_capital_fraction"]),
            first_selected_cycle_index=None
            if payload.get("first_selected_cycle_index") is None
            else int(payload["first_selected_cycle_index"]),
            last_selected_cycle_index=None
            if payload.get("last_selected_cycle_index") is None
            else int(payload["last_selected_cycle_index"]),
        )


@dataclass(slots=True)
class PortfolioLedgerReport:
    schema_version: str
    name: str
    created_at_utc: str
    source_rolling_report: str
    source_lifecycle_report: str
    config: dict[str, Any]
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    ledger_cycle_indices: list[int]
    ledger_balance_history: list[float]
    total_buy_turnover_fraction: float
    total_sell_turnover_fraction: float
    total_gross_turnover_fraction: float
    average_gross_turnover_fraction: float
    total_estimated_rebalance_cost: float
    average_allocated_risk_fraction: float
    average_reserve_fraction: float
    total_churn_count: int
    peak_cluster_exposure_fraction: float
    non_tradable_selection_count: int
    final_active_candidate_ids: list[str]
    final_status_counts: dict[str, int]
    candidate_summaries: list[PortfolioCandidateLedgerSummary]
    cycle_entries: list[PortfolioCycleLedgerEntry]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_rolling_report": self.source_rolling_report,
            "source_lifecycle_report": self.source_lifecycle_report,
            "config": dict(self.config),
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "ledger_cycle_indices": list(self.ledger_cycle_indices),
            "ledger_balance_history": list(self.ledger_balance_history),
            "total_buy_turnover_fraction": self.total_buy_turnover_fraction,
            "total_sell_turnover_fraction": self.total_sell_turnover_fraction,
            "total_gross_turnover_fraction": self.total_gross_turnover_fraction,
            "average_gross_turnover_fraction": self.average_gross_turnover_fraction,
            "total_estimated_rebalance_cost": self.total_estimated_rebalance_cost,
            "average_allocated_risk_fraction": self.average_allocated_risk_fraction,
            "average_reserve_fraction": self.average_reserve_fraction,
            "total_churn_count": self.total_churn_count,
            "peak_cluster_exposure_fraction": self.peak_cluster_exposure_fraction,
            "non_tradable_selection_count": self.non_tradable_selection_count,
            "final_active_candidate_ids": list(self.final_active_candidate_ids),
            "final_status_counts": {str(key): int(value) for key, value in self.final_status_counts.items()},
            "candidate_summaries": [item.to_dict() for item in self.candidate_summaries],
            "cycle_entries": [item.to_dict() for item in self.cycle_entries],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioLedgerReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_rolling_report=str(payload["source_rolling_report"]),
            source_lifecycle_report=str(payload["source_lifecycle_report"]),
            config=dict(payload.get("config", {})),
            initial_balance=float(payload["initial_balance"]),
            final_balance=float(payload["final_balance"]),
            total_pnl=float(payload["total_pnl"]),
            total_return_pct=float(payload["total_return_pct"]),
            max_drawdown_pct=float(payload["max_drawdown_pct"]),
            ledger_cycle_indices=[int(item) for item in payload.get("ledger_cycle_indices", [])],
            ledger_balance_history=[float(item) for item in payload.get("ledger_balance_history", [])],
            total_buy_turnover_fraction=float(payload["total_buy_turnover_fraction"]),
            total_sell_turnover_fraction=float(payload["total_sell_turnover_fraction"]),
            total_gross_turnover_fraction=float(payload["total_gross_turnover_fraction"]),
            average_gross_turnover_fraction=float(payload["average_gross_turnover_fraction"]),
            total_estimated_rebalance_cost=float(payload["total_estimated_rebalance_cost"]),
            average_allocated_risk_fraction=float(payload["average_allocated_risk_fraction"]),
            average_reserve_fraction=float(payload["average_reserve_fraction"]),
            total_churn_count=int(payload["total_churn_count"]),
            peak_cluster_exposure_fraction=float(payload["peak_cluster_exposure_fraction"]),
            non_tradable_selection_count=int(payload["non_tradable_selection_count"]),
            final_active_candidate_ids=[str(item) for item in payload.get("final_active_candidate_ids", [])],
            final_status_counts={str(key): int(value) for key, value in dict(payload.get("final_status_counts", {})).items()},
            candidate_summaries=[
                PortfolioCandidateLedgerSummary.from_dict(item) for item in payload.get("candidate_summaries", [])
            ],
            cycle_entries=[PortfolioCycleLedgerEntry.from_dict(item) for item in payload.get("cycle_entries", [])],
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class PortfolioBaselineCycleResult:
    baseline_name: str
    cycle_index: int
    cycle_name: str
    selected_candidate_ids: list[str]
    selected_display_names: list[str]
    ledger_balance_before: float
    ledger_balance_after_rebalance: float
    ledger_balance_after_cycle: float
    gross_cycle_pnl: float
    net_cycle_pnl: float
    max_drawdown_pct: float
    allocated_risk_fraction: float
    reserve_fraction_before: float
    reserve_fraction_after: float
    buy_turnover_fraction: float
    sell_turnover_fraction: float
    gross_turnover_fraction: float
    estimated_rebalance_cost: float
    allocation_map: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "cycle_index": self.cycle_index,
            "cycle_name": self.cycle_name,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_display_names": list(self.selected_display_names),
            "ledger_balance_before": self.ledger_balance_before,
            "ledger_balance_after_rebalance": self.ledger_balance_after_rebalance,
            "ledger_balance_after_cycle": self.ledger_balance_after_cycle,
            "gross_cycle_pnl": self.gross_cycle_pnl,
            "net_cycle_pnl": self.net_cycle_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "allocated_risk_fraction": self.allocated_risk_fraction,
            "reserve_fraction_before": self.reserve_fraction_before,
            "reserve_fraction_after": self.reserve_fraction_after,
            "buy_turnover_fraction": self.buy_turnover_fraction,
            "sell_turnover_fraction": self.sell_turnover_fraction,
            "gross_turnover_fraction": self.gross_turnover_fraction,
            "estimated_rebalance_cost": self.estimated_rebalance_cost,
            "allocation_map": {str(key): float(value) for key, value in self.allocation_map.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioBaselineCycleResult":
        return cls(
            baseline_name=str(payload["baseline_name"]),
            cycle_index=int(payload["cycle_index"]),
            cycle_name=str(payload["cycle_name"]),
            selected_candidate_ids=[str(item) for item in payload.get("selected_candidate_ids", [])],
            selected_display_names=[str(item) for item in payload.get("selected_display_names", [])],
            ledger_balance_before=float(payload["ledger_balance_before"]),
            ledger_balance_after_rebalance=float(payload["ledger_balance_after_rebalance"]),
            ledger_balance_after_cycle=float(payload["ledger_balance_after_cycle"]),
            gross_cycle_pnl=float(payload["gross_cycle_pnl"]),
            net_cycle_pnl=float(payload["net_cycle_pnl"]),
            max_drawdown_pct=float(payload["max_drawdown_pct"]),
            allocated_risk_fraction=float(payload["allocated_risk_fraction"]),
            reserve_fraction_before=float(payload["reserve_fraction_before"]),
            reserve_fraction_after=float(payload["reserve_fraction_after"]),
            buy_turnover_fraction=float(payload["buy_turnover_fraction"]),
            sell_turnover_fraction=float(payload["sell_turnover_fraction"]),
            gross_turnover_fraction=float(payload["gross_turnover_fraction"]),
            estimated_rebalance_cost=float(payload["estimated_rebalance_cost"]),
            allocation_map={str(key): float(value) for key, value in dict(payload.get("allocation_map", {})).items()},
        )


@dataclass(slots=True)
class PortfolioBaselineResult:
    baseline_name: str
    baseline_kind: str
    description: str
    selector_summary: dict[str, Any]
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    total_buy_turnover_fraction: float
    total_sell_turnover_fraction: float
    total_gross_turnover_fraction: float
    average_reserve_fraction: float
    total_estimated_rebalance_cost: float
    curve_sample_indices: list[int]
    normalized_balance_history: list[float]
    cycle_results: list[PortfolioBaselineCycleResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "baseline_kind": self.baseline_kind,
            "description": self.description,
            "selector_summary": dict(self.selector_summary),
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_buy_turnover_fraction": self.total_buy_turnover_fraction,
            "total_sell_turnover_fraction": self.total_sell_turnover_fraction,
            "total_gross_turnover_fraction": self.total_gross_turnover_fraction,
            "average_reserve_fraction": self.average_reserve_fraction,
            "total_estimated_rebalance_cost": self.total_estimated_rebalance_cost,
            "curve_sample_indices": list(self.curve_sample_indices),
            "normalized_balance_history": list(self.normalized_balance_history),
            "cycle_results": [item.to_dict() for item in self.cycle_results],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioBaselineResult":
        return cls(
            baseline_name=str(payload["baseline_name"]),
            baseline_kind=str(payload["baseline_kind"]),
            description=str(payload["description"]),
            selector_summary=dict(payload.get("selector_summary", {})),
            initial_balance=float(payload["initial_balance"]),
            final_balance=float(payload["final_balance"]),
            total_pnl=float(payload["total_pnl"]),
            total_return_pct=float(payload["total_return_pct"]),
            max_drawdown_pct=float(payload["max_drawdown_pct"]),
            total_buy_turnover_fraction=float(payload["total_buy_turnover_fraction"]),
            total_sell_turnover_fraction=float(payload["total_sell_turnover_fraction"]),
            total_gross_turnover_fraction=float(payload["total_gross_turnover_fraction"]),
            average_reserve_fraction=float(payload["average_reserve_fraction"]),
            total_estimated_rebalance_cost=float(payload["total_estimated_rebalance_cost"]),
            curve_sample_indices=[int(item) for item in payload.get("curve_sample_indices", [])],
            normalized_balance_history=[float(item) for item in payload.get("normalized_balance_history", [])],
            cycle_results=[PortfolioBaselineCycleResult.from_dict(item) for item in payload.get("cycle_results", [])],
        )


@dataclass(slots=True)
class PortfolioBaselineComparison:
    baseline_name: str
    baseline_kind: str
    conveyor_total_pnl: float
    baseline_total_pnl: float
    pnl_delta: float
    conveyor_total_return_pct: float
    baseline_total_return_pct: float
    return_pct_delta: float
    conveyor_max_drawdown_pct: float
    baseline_max_drawdown_pct: float
    drawdown_advantage_pct: float
    beats_by_pnl: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "baseline_kind": self.baseline_kind,
            "conveyor_total_pnl": self.conveyor_total_pnl,
            "baseline_total_pnl": self.baseline_total_pnl,
            "pnl_delta": self.pnl_delta,
            "conveyor_total_return_pct": self.conveyor_total_return_pct,
            "baseline_total_return_pct": self.baseline_total_return_pct,
            "return_pct_delta": self.return_pct_delta,
            "conveyor_max_drawdown_pct": self.conveyor_max_drawdown_pct,
            "baseline_max_drawdown_pct": self.baseline_max_drawdown_pct,
            "drawdown_advantage_pct": self.drawdown_advantage_pct,
            "beats_by_pnl": self.beats_by_pnl,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioBaselineComparison":
        return cls(
            baseline_name=str(payload["baseline_name"]),
            baseline_kind=str(payload["baseline_kind"]),
            conveyor_total_pnl=float(payload["conveyor_total_pnl"]),
            baseline_total_pnl=float(payload["baseline_total_pnl"]),
            pnl_delta=float(payload["pnl_delta"]),
            conveyor_total_return_pct=float(payload["conveyor_total_return_pct"]),
            baseline_total_return_pct=float(payload["baseline_total_return_pct"]),
            return_pct_delta=float(payload["return_pct_delta"]),
            conveyor_max_drawdown_pct=float(payload["conveyor_max_drawdown_pct"]),
            baseline_max_drawdown_pct=float(payload["baseline_max_drawdown_pct"]),
            drawdown_advantage_pct=float(payload["drawdown_advantage_pct"]),
            beats_by_pnl=bool(payload["beats_by_pnl"]),
        )


@dataclass(slots=True)
class PortfolioGateResult:
    overall_pass: bool
    config: dict[str, Any]
    checks: dict[str, bool]
    beaten_baselines: list[str]
    failed_required_baselines: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pass": self.overall_pass,
            "config": dict(self.config),
            "checks": {str(key): bool(value) for key, value in self.checks.items()},
            "beaten_baselines": list(self.beaten_baselines),
            "failed_required_baselines": list(self.failed_required_baselines),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioGateResult":
        return cls(
            overall_pass=bool(payload["overall_pass"]),
            config=dict(payload.get("config", {})),
            checks={str(key): bool(value) for key, value in dict(payload.get("checks", {})).items()},
            beaten_baselines=[str(item) for item in payload.get("beaten_baselines", [])],
            failed_required_baselines=[str(item) for item in payload.get("failed_required_baselines", [])],
        )


@dataclass(slots=True)
class PortfolioBaselineReport:
    schema_version: str
    name: str
    created_at_utc: str
    source_portfolio_ledger_report: str
    source_rolling_report: str
    config: dict[str, Any]
    conveyor_total_pnl: float
    conveyor_total_return_pct: float
    conveyor_max_drawdown_pct: float
    baselines: list[PortfolioBaselineResult]
    comparisons: list[PortfolioBaselineComparison]
    gate: PortfolioGateResult
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_portfolio_ledger_report": self.source_portfolio_ledger_report,
            "source_rolling_report": self.source_rolling_report,
            "config": dict(self.config),
            "conveyor_total_pnl": self.conveyor_total_pnl,
            "conveyor_total_return_pct": self.conveyor_total_return_pct,
            "conveyor_max_drawdown_pct": self.conveyor_max_drawdown_pct,
            "baselines": [item.to_dict() for item in self.baselines],
            "comparisons": [item.to_dict() for item in self.comparisons],
            "gate": self.gate.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioBaselineReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_portfolio_ledger_report=str(payload["source_portfolio_ledger_report"]),
            source_rolling_report=str(payload["source_rolling_report"]),
            config=dict(payload.get("config", {})),
            conveyor_total_pnl=float(payload["conveyor_total_pnl"]),
            conveyor_total_return_pct=float(payload["conveyor_total_return_pct"]),
            conveyor_max_drawdown_pct=float(payload["conveyor_max_drawdown_pct"]),
            baselines=[PortfolioBaselineResult.from_dict(item) for item in payload.get("baselines", [])],
            comparisons=[PortfolioBaselineComparison.from_dict(item) for item in payload.get("comparisons", [])],
            gate=PortfolioGateResult.from_dict(payload["gate"]),
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class DashboardFeed:
    schema_version: str
    name: str
    created_at_utc: str
    source_shortlist_report: str | None
    source_diversification_report: str | None
    source_cluster_report: str | None
    source_override_set: str | None
    source_allocator_report: str | None
    source_combination_report: str | None
    summary: dict[str, Any]
    candidates: list[dict[str, Any]]
    monitoring: dict[str, Any] = field(default_factory=dict)
    broom: dict[str, Any] | None = None
    clusters: list[dict[str, Any]] = field(default_factory=list)
    overrides: dict[str, Any] | None = None
    allocator: dict[str, Any] | None = None
    combinations: dict[str, Any] | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_shortlist_report": self.source_shortlist_report,
            "source_diversification_report": self.source_diversification_report,
            "source_cluster_report": self.source_cluster_report,
            "source_override_set": self.source_override_set,
            "source_allocator_report": self.source_allocator_report,
            "source_combination_report": self.source_combination_report,
            "summary": dict(self.summary),
            "candidates": [dict(item) for item in self.candidates],
            "monitoring": dict(self.monitoring),
            "broom": None if self.broom is None else dict(self.broom),
            "clusters": [dict(item) for item in self.clusters],
            "overrides": None if self.overrides is None else dict(self.overrides),
            "allocator": None if self.allocator is None else dict(self.allocator),
            "combinations": None if self.combinations is None else dict(self.combinations),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DashboardFeed":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_shortlist_report=payload.get("source_shortlist_report"),
            source_diversification_report=payload.get("source_diversification_report"),
            source_cluster_report=payload.get("source_cluster_report"),
            source_override_set=payload.get("source_override_set"),
            source_allocator_report=payload.get("source_allocator_report"),
            source_combination_report=payload.get("source_combination_report"),
            summary=dict(payload.get("summary", {})),
            candidates=[dict(item) for item in payload.get("candidates", [])],
            monitoring=dict(payload.get("monitoring", {})),
            broom=None if payload.get("broom") is None else dict(payload["broom"]),
            clusters=[dict(item) for item in payload.get("clusters", [])],
            overrides=None if payload.get("overrides") is None else dict(payload["overrides"]),
            allocator=None if payload.get("allocator") is None else dict(payload["allocator"]),
            combinations=None if payload.get("combinations") is None else dict(payload["combinations"]),
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class CandidateFarmScenarioReport:
    scenario_name: str
    status: str
    mode: str
    output_dir: str
    config: dict[str, Any]
    log_paths: dict[str, str]
    updated_at_utc: str | None = None
    progress_stage: str | None = None
    rolling_report_name: str | None = None
    lifecycle_report_name: str | None = None
    portfolio_ledger_report_name: str | None = None
    portfolio_baselines_report_name: str | None = None
    selector: dict[str, Any] = field(default_factory=dict)
    candidate_pool_ids: list[str] = field(default_factory=list)
    selected_candidate_ids: list[str] = field(default_factory=list)
    final_status_counts: dict[str, int] = field(default_factory=dict)
    gate_pass: bool | None = None
    beaten_baselines: list[str] = field(default_factory=list)
    failed_required_baselines: list[str] = field(default_factory=list)
    total_pnl: float | None = None
    total_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    evaluated_cycle_count: int | None = None
    positive_cycle_count: int | None = None
    error_message: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "status": self.status,
            "mode": self.mode,
            "output_dir": self.output_dir,
            "config": dict(self.config),
            "log_paths": {str(key): str(value) for key, value in self.log_paths.items()},
            "updated_at_utc": self.updated_at_utc,
            "progress_stage": self.progress_stage,
            "rolling_report_name": self.rolling_report_name,
            "lifecycle_report_name": self.lifecycle_report_name,
            "portfolio_ledger_report_name": self.portfolio_ledger_report_name,
            "portfolio_baselines_report_name": self.portfolio_baselines_report_name,
            "selector": dict(self.selector),
            "candidate_pool_ids": list(self.candidate_pool_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "final_status_counts": {str(key): int(value) for key, value in self.final_status_counts.items()},
            "gate_pass": self.gate_pass,
            "beaten_baselines": list(self.beaten_baselines),
            "failed_required_baselines": list(self.failed_required_baselines),
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "evaluated_cycle_count": self.evaluated_cycle_count,
            "positive_cycle_count": self.positive_cycle_count,
            "error_message": self.error_message,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateFarmScenarioReport":
        return cls(
            scenario_name=str(payload["scenario_name"]),
            status=str(payload["status"]),
            mode=str(payload["mode"]),
            output_dir=str(payload["output_dir"]),
            config=dict(payload.get("config", {})),
            log_paths={str(key): str(value) for key, value in dict(payload.get("log_paths", {})).items()},
            updated_at_utc=payload.get("updated_at_utc"),
            progress_stage=payload.get("progress_stage"),
            rolling_report_name=payload.get("rolling_report_name"),
            lifecycle_report_name=payload.get("lifecycle_report_name"),
            portfolio_ledger_report_name=payload.get("portfolio_ledger_report_name"),
            portfolio_baselines_report_name=payload.get("portfolio_baselines_report_name"),
            selector=dict(payload.get("selector", {})),
            candidate_pool_ids=[str(item) for item in payload.get("candidate_pool_ids", [])],
            selected_candidate_ids=[str(item) for item in payload.get("selected_candidate_ids", [])],
            final_status_counts={str(key): int(value) for key, value in dict(payload.get("final_status_counts", {})).items()},
            gate_pass=None if payload.get("gate_pass") is None else bool(payload["gate_pass"]),
            beaten_baselines=[str(item) for item in payload.get("beaten_baselines", [])],
            failed_required_baselines=[str(item) for item in payload.get("failed_required_baselines", [])],
            total_pnl=None if payload.get("total_pnl") is None else float(payload["total_pnl"]),
            total_return_pct=None if payload.get("total_return_pct") is None else float(payload["total_return_pct"]),
            max_drawdown_pct=None if payload.get("max_drawdown_pct") is None else float(payload["max_drawdown_pct"]),
            evaluated_cycle_count=None
            if payload.get("evaluated_cycle_count") is None
            else int(payload["evaluated_cycle_count"]),
            positive_cycle_count=None
            if payload.get("positive_cycle_count") is None
            else int(payload["positive_cycle_count"]),
            error_message=payload.get("error_message"),
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class CandidateFarmReport:
    schema_version: str
    name: str
    created_at_utc: str
    source_manifest_path: str | None
    registry_root: str
    scenarios: list[CandidateFarmScenarioReport]
    summary: dict[str, Any]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_manifest_path": self.source_manifest_path,
            "registry_root": self.registry_root,
            "scenarios": [item.to_dict() for item in self.scenarios],
            "summary": dict(self.summary),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateFarmReport":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_manifest_path=payload.get("source_manifest_path"),
            registry_root=str(payload["registry_root"]),
            scenarios=[CandidateFarmScenarioReport.from_dict(item) for item in payload.get("scenarios", [])],
            summary=dict(payload.get("summary", {})),
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class FarmProgressEvent:
    sequence: int
    created_at_utc: str
    scenario_name: str
    status: str
    progress_stage: str | None = None
    event_kind: str = "heartbeat"
    gate_pass: bool | None = None
    total_pnl: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FarmProgressEvent":
        return cls(
            sequence=int(payload["sequence"]),
            created_at_utc=str(payload["created_at_utc"]),
            scenario_name=str(payload["scenario_name"]),
            status=str(payload["status"]),
            progress_stage=payload.get("progress_stage"),
            event_kind=str(payload.get("event_kind", "heartbeat")),
            gate_pass=None if payload.get("gate_pass") is None else bool(payload["gate_pass"]),
            total_pnl=None if payload.get("total_pnl") is None else float(payload["total_pnl"]),
            note=payload.get("note"),
        )


@dataclass(slots=True)
class FarmProgressLog:
    schema_version: str
    farm_name: str
    created_at_utc: str
    updated_at_utc: str
    events: list[FarmProgressEvent]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "farm_name": self.farm_name,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "events": [item.to_dict() for item in self.events],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FarmProgressLog":
        return cls(
            schema_version=str(payload["schema_version"]),
            farm_name=str(payload["farm_name"]),
            created_at_utc=str(payload["created_at_utc"]),
            updated_at_utc=str(payload["updated_at_utc"]),
            events=[FarmProgressEvent.from_dict(item) for item in payload.get("events", [])],
            notes=payload.get("notes"),
        )


@dataclass(slots=True)
class FarmDashboardFeed:
    schema_version: str
    name: str
    created_at_utc: str
    source_farm_report: str
    summary: dict[str, Any]
    scenarios: list[dict[str, Any]]
    monitoring: dict[str, Any] = field(default_factory=dict)
    broom: dict[str, Any] | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "source_farm_report": self.source_farm_report,
            "summary": dict(self.summary),
            "scenarios": [dict(item) for item in self.scenarios],
            "monitoring": dict(self.monitoring),
            "broom": None if self.broom is None else dict(self.broom),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FarmDashboardFeed":
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            created_at_utc=str(payload["created_at_utc"]),
            source_farm_report=str(payload["source_farm_report"]),
            summary=dict(payload.get("summary", {})),
            scenarios=[dict(item) for item in payload.get("scenarios", [])],
            monitoring=dict(payload.get("monitoring", {})),
            broom=None if payload.get("broom") is None else dict(payload["broom"]),
            notes=payload.get("notes"),
        )


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
