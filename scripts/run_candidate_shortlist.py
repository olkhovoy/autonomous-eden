#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, ShortlistConfig, build_shortlist_report, utc_now_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a diversified shortlist from a diversification report and candidate registry.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--diversification-report", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--resampling-name", default="train_bootstrap_f1.00")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--min-marginal-score", type=float, default=0.25)
    parser.add_argument("--max-pair-downside-corr", type=float, default=0.65)
    parser.add_argument("--max-pair-simultaneous-loss-rate", type=float, default=0.70)
    parser.add_argument("--notes")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    diversification_report = registry.load_diversification_report(args.diversification_report)
    candidates = [registry.load_candidate(candidate_id) for candidate_id in diversification_report.candidate_ids]
    config = ShortlistConfig(
        resampling_name=args.resampling_name,
        max_candidates=args.max_candidates,
        min_marginal_score=args.min_marginal_score,
        max_pair_downside_corr=args.max_pair_downside_corr,
        max_pair_simultaneous_loss_rate=args.max_pair_simultaneous_loss_rate,
    )
    report = build_shortlist_report(
        name=args.report_name,
        created_at_utc=utc_now_text(),
        candidates=candidates,
        diversification_report=diversification_report,
        config=config,
        notes=args.notes,
    )
    path = registry.save_shortlist_report(report)

    print(f"Saved shortlist report to {path}")
    print("Selected candidates:")
    for item in report.candidate_scores:
        if not item.selected:
            continue
        flags = f" flags={','.join(item.exception_flags)}" if item.exception_flags else ""
        print(
            f"  rank={item.selected_rank} {item.display_name} "
            f"base={item.base_score:+.3f} marginal={item.marginal_score:+.3f} "
            f"brightness={item.brightness_hint:.2f}{flags}"
        )

    if report.selected_pair_scores:
        print("Selected pair compatibility:")
        for pair in report.selected_pair_scores:
            print(
                f"  {pair.left_candidate_id} <-> {pair.right_candidate_id} "
                f"compat={pair.compatibility_score:+.3f} "
                f"downside={pair.downside_corr:+.3f} "
                f"sim_loss={pair.simultaneous_loss_rate:.3f}"
            )


if __name__ == "__main__":
    main()
