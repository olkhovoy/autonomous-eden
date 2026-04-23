#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, build_cluster_report, utc_now_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cluster assignments from a diversification report.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--diversification-report", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.40)
    parser.add_argument("--notes")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    diversification = registry.load_diversification_report(args.diversification_report)
    report = build_cluster_report(
        diversification,
        name=args.report_name,
        created_at_utc=utc_now_text(),
        similarity_threshold=args.similarity_threshold,
        notes=args.notes,
    )
    path = registry.save_cluster_report(report)
    print(f"Saved cluster report to {path}")
    for cluster in report.clusters:
        print(
            f"{cluster.cluster_id} | size={cluster.cluster_size} "
            f"members={','.join(cluster.display_names)} "
            f"sim={cluster.mean_similarity_score:.3f} "
            f"downside={cluster.mean_downside_corr:.3f} "
            f"sim_loss={cluster.mean_simultaneous_loss_rate:.3f}"
        )


if __name__ == "__main__":
    main()
