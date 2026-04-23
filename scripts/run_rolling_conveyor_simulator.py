#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import (
    CandidateRegistry,
    build_rolling_conveyor_report,
    build_rolling_window_specs,
)


MANAGED_CYCLE_ARGS = {
    "--cycle-name",
    "--mode",
    "--registry-root",
    "--selection-start-utc",
    "--selection-end-utc",
    "--forward-start-utc",
    "--forward-end-utc",
    "--forward-days",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple continuous-search cycles across sequential windows and stitch them into a rolling conveyor report.",
        epilog=(
            "Additional unknown args are forwarded to scripts/run_continuous_search_cycle.py. "
            "Do not pass cycle-managed flags such as --cycle-name or window boundaries."
        ),
    )
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--mode", choices=["reuse", "generate"], default="reuse")
    parser.add_argument("--selection-start-utc", required=True)
    parser.add_argument("--selection-days", type=int, default=7)
    parser.add_argument("--forward-days", type=int, default=7)
    parser.add_argument("--step-days", type=int)
    parser.add_argument("--num-cycles", type=int, default=3)
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--curve-points", type=int, default=256)
    parser.add_argument("--notes")
    return parser


def _run_logged(cmd: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        subprocess.run(cmd, cwd=str(ROOT), check=True, stdout=log_file, stderr=subprocess.STDOUT)


def _extract_selector(forwarded_args: list[str]) -> dict[str, object]:
    selector: dict[str, object] = {"forwarded_cycle_args": list(forwarded_args)}
    candidate_ids: list[str] = []
    tags: list[str] = []
    status: str | None = None
    i = 0
    while i < len(forwarded_args):
        token = forwarded_args[i]
        next_value = None if i + 1 >= len(forwarded_args) else forwarded_args[i + 1]
        if token == "--candidate-id" and next_value is not None:
            candidate_ids.append(next_value)
            i += 2
            continue
        if token == "--tag" and next_value is not None:
            tags.append(next_value)
            i += 2
            continue
        if token == "--status" and next_value is not None:
            status = next_value
            i += 2
            continue
        i += 1
    if candidate_ids:
        selector["candidate_ids"] = candidate_ids
    if tags:
        selector["tags"] = tags
    if status is not None:
        selector["status"] = status
    return selector


def main() -> None:
    parser = _build_parser()
    args, forwarded_cycle_args = parser.parse_known_args()

    conflicts = sorted(set(forwarded_cycle_args).intersection(MANAGED_CYCLE_ARGS))
    if conflicts:
        parser.error(f"These args are managed by the rolling simulator and must not be forwarded: {', '.join(conflicts)}")

    resolved_step_days = args.forward_days if args.step_days is None else args.step_days
    specs = build_rolling_window_specs(
        report_name=args.report_name,
        selection_start_utc=args.selection_start_utc,
        selection_days=args.selection_days,
        forward_days=args.forward_days,
        cycle_count=args.num_cycles,
        step_days=resolved_step_days,
    )
    registry = CandidateRegistry(ROOT / args.registry_root)
    output_dir = ROOT / "checkpoints" / "rolling_conveyor" / args.report_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cycle_reports = []
    tradeforward_plans = []
    tradeforward_evaluations = []

    for spec in specs:
        cycle_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_continuous_search_cycle.py"),
            "--registry-root",
            args.registry_root,
            "--cycle-name",
            spec.cycle_name,
            "--mode",
            args.mode,
            "--selection-start-utc",
            spec.selection_start_utc,
            "--selection-end-utc",
            spec.selection_end_utc,
            "--forward-start-utc",
            spec.forward_start_utc,
            "--forward-end-utc",
            spec.forward_end_utc,
            *forwarded_cycle_args,
        ]
        if args.notes:
            cycle_cmd.extend(["--notes", args.notes])
        _run_logged(cycle_cmd, log_path=output_dir / f"{spec.cycle_name}_cycle.log")

        cycle_report = registry.load_cycle_report(spec.cycle_name)
        plan_name = cycle_report.report_names["tradeforward"]
        eval_name = f"{plan_name}_eval"
        eval_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_tradeforward_plan.py"),
            "--registry-root",
            args.registry_root,
            "--plan-name",
            plan_name,
            "--report-name",
            eval_name,
            "--curve-points",
            str(args.curve_points),
        ]
        if args.notes:
            eval_cmd.extend(["--notes", args.notes])
        _run_logged(eval_cmd, log_path=output_dir / f"{spec.cycle_name}_tradeforward_eval.log")

        cycle_reports.append(cycle_report)
        tradeforward_plans.append(registry.load_tradeforward_plan(plan_name))
        tradeforward_evaluations.append(registry.load_tradeforward_evaluation(eval_name))

    report = build_rolling_conveyor_report(
        registry,
        args.report_name,
        mode=args.mode,
        cycle_reports=cycle_reports,
        tradeforward_plans=tradeforward_plans,
        tradeforward_evaluations=tradeforward_evaluations,
        selection_start_utc=args.selection_start_utc,
        selection_days=args.selection_days,
        forward_days=args.forward_days,
        step_days=resolved_step_days,
        initial_balance=args.initial_balance,
        selector=_extract_selector(forwarded_cycle_args),
        notes=args.notes,
    )
    path = registry.save_rolling_conveyor(report)

    print(f"Saved rolling conveyor report to {path}")
    print(
        f"cycles={report.evaluated_cycle_count}/{args.num_cycles} "
        f"balance={report.final_balance:.2f} "
        f"pnl={report.total_pnl:+.2f} "
        f"return={report.total_return_pct:+.2f}% "
        f"max_dd={report.max_drawdown_pct:.2f}%"
    )
    for item in report.cycle_outcomes:
        print(
            f"  cycle_{item.cycle_index:02d} "
            f"{item.forward_start_utc}->{item.forward_end_utc} "
            f"pnl={item.portfolio_pnl:+.2f} "
            f"dd={item.portfolio_max_drawdown_pct:.2f}% "
            f"selected={len(item.selected_candidate_ids)}"
        )


if __name__ == "__main__":
    main()
