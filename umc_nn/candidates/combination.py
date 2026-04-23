from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from .schema import (
    CandidateRecord,
    DiversificationPairStats,
    DiversificationReport,
    ResamplingStats,
    ShortlistCandidateScore,
    ShortlistPairScore,
    ShortlistReport,
)


@dataclass(slots=True)
class ShortlistConfig:
    resampling_name: str
    max_candidates: int = 4
    min_marginal_score: float = 0.25
    max_pair_downside_corr: float = 0.65
    max_pair_simultaneous_loss_rate: float = 0.70

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "resampling_name": self.resampling_name,
            "max_candidates": self.max_candidates,
            "min_marginal_score": self.min_marginal_score,
            "max_pair_downside_corr": self.max_pair_downside_corr,
            "max_pair_simultaneous_loss_rate": self.max_pair_simultaneous_loss_rate,
        }


def build_shortlist_report(
    *,
    name: str,
    created_at_utc: str,
    candidates: Iterable[CandidateRecord],
    diversification_report: DiversificationReport,
    config: ShortlistConfig,
    notes: str | None = None,
) -> ShortlistReport:
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    candidate_ids = list(diversification_report.candidate_ids)
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidate_map]
    if missing:
        raise ValueError(f"Missing candidate records for shortlist: {missing}")

    pair_lookup: dict[frozenset[str], tuple[float, ShortlistPairScore]] = {}
    pair_scores_by_candidate: dict[str, list[float]] = {candidate_id: [] for candidate_id in candidate_ids}
    for pair in diversification_report.pair_stats:
        pair_score = compatibility_breakdown(pair)
        key = frozenset((pair.left_candidate_id, pair.right_candidate_id))
        pair_lookup[key] = (pair_score.compatibility_score, pair_score)
        pair_scores_by_candidate[pair.left_candidate_id].append(pair_score.compatibility_score)
        pair_scores_by_candidate[pair.right_candidate_id].append(pair_score.compatibility_score)

    candidate_scores: dict[str, ShortlistCandidateScore] = {}
    for candidate_id in candidate_ids:
        candidate = candidate_map[candidate_id]
        base_score = base_candidate_score(
            candidate,
            _resolve_resampling(candidate, config.resampling_name),
            pair_scores_by_candidate[candidate_id],
        )
        candidate_scores[candidate_id] = ShortlistCandidateScore(
            candidate_id=candidate_id,
            display_name=candidate.display_name,
            selected=False,
            selected_rank=None,
            base_score=float(sum(base_score.values())),
            marginal_score=None,
            brightness_hint=0.0,
            score_components=base_score,
            exception_flags=[],
        )

    selected_ids = greedy_select_candidates(
        candidate_ids=candidate_ids,
        candidate_scores=candidate_scores,
        pair_lookup=pair_lookup,
        config=config,
    )

    for rank, candidate_id in enumerate(selected_ids, start=1):
        selected = candidate_scores[candidate_id]
        selected.selected = True
        selected.selected_rank = rank
        selected.marginal_score = selection_marginal_score(
            candidate_id,
            selected_ids[: rank - 1],
            candidate_scores,
            pair_lookup,
        )

    exception_flags = detect_exception_flags(
        candidate_scores=list(candidate_scores.values()),
        pair_scores_by_candidate=pair_scores_by_candidate,
    )
    for candidate_id, flags in exception_flags.items():
        candidate_scores[candidate_id].exception_flags = flags

    apply_brightness_hints(candidate_scores.values())

    selected_pair_scores: list[ShortlistPairScore] = []
    for left_id, right_id in combinations(selected_ids, 2):
        _, pair_score = pair_lookup[frozenset((left_id, right_id))]
        selected_pair_scores.append(pair_score)
    selected_pair_scores.sort(key=lambda item: item.compatibility_score, reverse=True)

    ordered_candidate_scores = sorted(
        candidate_scores.values(),
        key=lambda item: (
            not item.selected,
            item.selected_rank if item.selected_rank is not None else 10**9,
            -item.base_score,
        ),
    )

    return ShortlistReport(
        schema_version="1",
        name=name,
        created_at_utc=created_at_utc,
        source_diversification_report=diversification_report.name,
        data_path=diversification_report.data_path,
        start_utc=diversification_report.start_utc,
        end_utc=diversification_report.end_utc,
        start_step=diversification_report.start_step,
        max_steps=diversification_report.max_steps,
        resampling_name=config.resampling_name,
        candidate_ids=candidate_ids,
        selected_candidate_ids=selected_ids,
        selection_config=config.to_dict(),
        candidate_scores=ordered_candidate_scores,
        selected_pair_scores=selected_pair_scores,
        notes=notes,
    )


def base_candidate_score(
    candidate: CandidateRecord,
    resampling: ResamplingStats,
    pair_scores: list[float],
) -> dict[str, float]:
    train = candidate.periods.get("train")
    oos = candidate.periods.get("oos_adjacent") or candidate.periods.get("oos")
    if train is None or oos is None:
        raise ValueError(f"Candidate {candidate.candidate_id} must have train and oos periods")
    initial_balance = 10000.0
    if candidate.manifest is not None:
        initial_balance = float(candidate.manifest.econ_config.get("initial_balance", initial_balance))

    oos_return = oos.pnl / max(initial_balance, 1e-12)
    pessimistic_return = resampling.pessimistic_net_profit / max(initial_balance, 1e-12)
    pair_potential = 0.0
    if pair_scores:
        pair_potential = float(np.mean(sorted(pair_scores, reverse=True)[: min(2, len(pair_scores))]))

    return {
        "oos_return_score": 3.0 * _squash(oos_return, scale=0.03),
        "oos_beats_flat_score": 0.75 if candidate.selection_flags.get("oos_beats_flat") else 0.0,
        "oos_activity_score": 0.75 * min(oos.trades / 40.0, 1.0),
        "oos_full_window_bonus": 0.25 if oos.full_window else 0.0,
        "resampling_pessimistic_score": 2.5 * _squash(pessimistic_return, scale=0.02),
        "resampling_profitability_score": 1.5 * ((resampling.profitable_rate - 0.5) * 2.0),
        "resampling_drawdown_penalty": -0.75 * min(resampling.pessimistic_max_drawdown_pct / 5.0, 2.0),
        "resampling_ruin_penalty": -3.0 * resampling.ruin_rate,
        "diversifier_potential_score": 0.6 * pair_potential,
        "flat_collapse_penalty": -2.0 if oos.trades == 0 else 0.0,
    }


def compatibility_breakdown(pair: DiversificationPairStats) -> ShortlistPairScore:
    drawdown_bonus = 0.30 * pair.drawdown_improvement_pct_points
    downside_penalty = 3.0 * max(0.0, pair.downside_corr - 0.15)
    simultaneous_loss_penalty = 2.5 * max(0.0, pair.simultaneous_loss_rate - 0.25)
    action_agreement_penalty = 1.0 * max(0.0, pair.action_agreement_rate - 0.80)
    compatibility_score = (
        drawdown_bonus
        + 0.75 * max(0.0, -pair.return_corr)
        + 0.50 * pair.opposite_nonflat_rate
        - downside_penalty
        - simultaneous_loss_penalty
        - action_agreement_penalty
        - 0.50 * max(0.0, pair.same_nonflat_rate - 0.50)
    )
    return ShortlistPairScore(
        left_candidate_id=pair.left_candidate_id,
        right_candidate_id=pair.right_candidate_id,
        compatibility_score=compatibility_score,
        downside_corr=pair.downside_corr,
        simultaneous_loss_rate=pair.simultaneous_loss_rate,
        action_agreement_rate=pair.action_agreement_rate,
        drawdown_bonus=drawdown_bonus,
        downside_penalty=downside_penalty,
        simultaneous_loss_penalty=simultaneous_loss_penalty,
        action_agreement_penalty=action_agreement_penalty,
    )


def greedy_select_candidates(
    *,
    candidate_ids: list[str],
    candidate_scores: dict[str, ShortlistCandidateScore],
    pair_lookup: dict[frozenset[str], tuple[float, ShortlistPairScore]],
    config: ShortlistConfig,
) -> list[str]:
    remaining = list(candidate_ids)
    remaining.sort(key=lambda candidate_id: candidate_scores[candidate_id].base_score, reverse=True)
    selected: list[str] = []

    while remaining and len(selected) < config.max_candidates:
        best_candidate_id: str | None = None
        best_score = float("-inf")
        for candidate_id in remaining:
            if violates_hard_pair_limits(candidate_id, selected, pair_lookup, config):
                continue
            marginal = selection_marginal_score(candidate_id, selected, candidate_scores, pair_lookup)
            if marginal > best_score:
                best_score = marginal
                best_candidate_id = candidate_id
        if best_candidate_id is None:
            break
        if selected and best_score < config.min_marginal_score:
            break
        selected.append(best_candidate_id)
        remaining.remove(best_candidate_id)

    return selected


def selection_marginal_score(
    candidate_id: str,
    selected_ids: list[str],
    candidate_scores: dict[str, ShortlistCandidateScore],
    pair_lookup: dict[frozenset[str], tuple[float, ShortlistPairScore]],
) -> float:
    base_score = candidate_scores[candidate_id].base_score
    if not selected_ids:
        return base_score
    pair_scores = [pair_lookup[frozenset((candidate_id, selected_id))][0] for selected_id in selected_ids]
    return base_score + float(np.mean(pair_scores))


def violates_hard_pair_limits(
    candidate_id: str,
    selected_ids: list[str],
    pair_lookup: dict[frozenset[str], tuple[float, ShortlistPairScore]],
    config: ShortlistConfig,
) -> bool:
    for selected_id in selected_ids:
        pair_score = pair_lookup[frozenset((candidate_id, selected_id))][1]
        if pair_score.downside_corr > config.max_pair_downside_corr:
            return True
        if pair_score.simultaneous_loss_rate > config.max_pair_simultaneous_loss_rate:
            return True
    return False


def detect_exception_flags(
    *,
    candidate_scores: list[ShortlistCandidateScore],
    pair_scores_by_candidate: dict[str, list[float]],
) -> dict[str, list[str]]:
    result = {candidate.candidate_id: [] for candidate in candidate_scores}
    ranked_by_base = sorted(candidate_scores, key=lambda item: item.base_score, reverse=True)
    ranked_by_pair = sorted(
        candidate_scores,
        key=lambda item: float(np.mean(sorted(pair_scores_by_candidate[item.candidate_id], reverse=True)[:2])) if pair_scores_by_candidate[item.candidate_id] else -999.0,
        reverse=True,
    )

    top_base = {item.candidate_id for item in ranked_by_base[:2]}
    top_pair = {item.candidate_id for item in ranked_by_pair[:2]}
    for candidate in candidate_scores:
        components = candidate.score_components
        if components.get("oos_return_score", 0.0) > 0.0 and components.get("resampling_pessimistic_score", 0.0) < 0.0:
            result[candidate.candidate_id].append("oos_positive_vs_fragile_train")
        if components.get("flat_collapse_penalty", 0.0) < 0.0:
            result[candidate.candidate_id].append("oos_flat_collapse")
        if candidate.candidate_id in top_pair and candidate.candidate_id not in top_base:
            result[candidate.candidate_id].append("diversifier_outlier")
        if candidate.candidate_id in top_base and components.get("oos_return_score", 0.0) <= 0.0:
            result[candidate.candidate_id].append("near_miss_quality")
    return result


def apply_brightness_hints(candidate_scores: Iterable[ShortlistCandidateScore]) -> None:
    items = list(candidate_scores)
    if not items:
        return
    raw_scores = [
        candidate.marginal_score if candidate.selected and candidate.marginal_score is not None else candidate.base_score
        for candidate in items
    ]
    low = min(raw_scores)
    high = max(raw_scores)
    for candidate, raw in zip(items, raw_scores, strict=True):
        if high - low < 1e-9:
            candidate.brightness_hint = 1.0
        else:
            candidate.brightness_hint = 0.2 + 0.8 * ((raw - low) / (high - low))


def _resolve_resampling(candidate: CandidateRecord, resampling_name: str) -> ResamplingStats:
    if resampling_name not in candidate.resampling_results:
        available = ", ".join(sorted(candidate.resampling_results))
        raise KeyError(
            f"Candidate {candidate.candidate_id} has no resampling '{resampling_name}'. "
            f"Available: {available}"
        )
    return candidate.resampling_results[resampling_name]


def _squash(value: float, *, scale: float) -> float:
    if abs(scale) < 1e-12:
        return 0.0
    return float(np.tanh(value / scale))
