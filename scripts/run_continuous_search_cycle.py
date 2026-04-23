#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import (
    CandidateRegistry,
    ContinuousSearchCycleReport,
    CycleStepRecord,
    build_tradeforward_plan,
    utc_now_text,
)


def _parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _default_cycle_name() -> str:
    return datetime.now(timezone.utc).strftime("cycle_%Y%m%d_%H%M%S")


def _selection_span_days(start_utc: str, end_utc: str) -> int:
    delta = _parse_utc(end_utc) - _parse_utc(start_utc)
    if delta.total_seconds() <= 0:
        raise ValueError("selection_end_utc must be after selection_start_utc")
    if delta.total_seconds() % 86400 != 0:
        raise ValueError("Selection span must be an integer number of days for generator window derivation")
    return int(delta.total_seconds() // 86400)


def _generator_window_plan(args: argparse.Namespace) -> tuple[list[str], int]:
    if args.window_start:
        return list(args.window_start), int(args.oos_days)
    if args.generator_window_mode == "explicit":
        return [], int(args.oos_days)

    selection_days = _selection_span_days(args.selection_start_utc, args.selection_end_utc)
    offsets = args.generator_window_offset_days or [0]
    starts: list[str] = []
    selection_start = _parse_utc(args.selection_start_utc)
    for offset_days in offsets:
        if offset_days > 0:
            raise ValueError("generator_window_offset_days must be <= 0 to avoid leaking future review data")
        train_start = selection_start + timedelta(days=offset_days - int(args.train_days))
        starts.append(_format_utc(train_start))
    return sorted(set(starts)), selection_days


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one continuous candidate-search cycle and produce downstream portfolio artifacts.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--cycle-name", default=_default_cycle_name())
    parser.add_argument("--mode", choices=["reuse", "generate"], default="reuse")
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--status")
    parser.add_argument("--selection-start-utc", required=True)
    parser.add_argument("--selection-end-utc", required=True)
    parser.add_argument("--forward-start-utc")
    parser.add_argument("--forward-end-utc")
    parser.add_argument("--forward-days", type=int, default=7)
    parser.add_argument("--resampling-fraction", action="append", type=float)
    parser.add_argument("--resampling-iterations", type=int, default=500)
    parser.add_argument("--resampling-seed", type=int, default=42)
    parser.add_argument("--resampling-name-prefix", default="bootstrap")
    parser.add_argument("--similarity-threshold", type=float, default=0.40)
    parser.add_argument("--shortlist-max-candidates", type=int, default=4)
    parser.add_argument("--shortlist-min-marginal-score", type=float, default=0.25)
    parser.add_argument("--shortlist-max-pair-downside-corr", type=float, default=0.65)
    parser.add_argument("--shortlist-max-pair-sim-loss", type=float, default=0.70)
    parser.add_argument("--risk-fraction", action="append", type=float)
    parser.add_argument("--per-system-cap", type=float, default=0.35)
    parser.add_argument("--default-cluster-cap", type=float, default=0.50)
    parser.add_argument("--combination-max-pool-size", type=int, default=6)
    parser.add_argument("--combination-min-subset-size", type=int, default=1)
    parser.add_argument("--combination-max-subset-size", type=int, default=3)
    parser.add_argument("--portfolio-iterations", type=int, default=300)
    parser.add_argument("--portfolio-block-size", type=int, default=64)
    parser.add_argument("--objective-max-dd", type=float, default=15.0)
    parser.add_argument("--curve-points", type=int, default=256)
    parser.add_argument("--tradeforward-mode", choices=["combination", "allocator"], default="combination")
    parser.add_argument("--notes")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--data-path", default="data/BTCUSDT_parquet_neurobars_autoresearch.npz")
    parser.add_argument("--window-start", action="append")
    parser.add_argument(
        "--generator-window-mode",
        choices=["selection_anchored", "explicit"],
        default="selection_anchored",
    )
    parser.add_argument("--generator-window-offset-days", action="append", type=int)
    parser.add_argument("--train-days", type=int, default=7)
    parser.add_argument("--oos-days", type=int, default=7)
    parser.add_argument("--runs-per-window", type=int, default=2)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--skip-filtering-phase", action="store_true")
    parser.add_argument("--engine-name", default="monolith")
    parser.add_argument("--engine-role", default="candidate_engine")
    parser.add_argument("--representation-name", default="neurobars_autoresearch_fused32")
    parser.add_argument("--engine-family", default="umc")
    parser.add_argument("--engine-hidden-dim", type=int, default=64)
    parser.add_argument("--engine-alpha", type=float, default=0.5)
    parser.add_argument("--action-head-mode", default="argmax")
    parser.add_argument("--action-threshold", type=float, default=0.55)
    parser.add_argument("--fitness-profile", default="hunter")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--activity-target-trades", type=int, default=12)
    parser.add_argument("--trade-band-low", type=int, default=None)
    parser.add_argument("--trade-band-high", type=int, default=None)
    parser.add_argument("--trade-band-floor", type=float, default=0.25)
    parser.add_argument("--valid-days", type=int, default=2)
    parser.add_argument("--market-scope", default="crypto:BTCUSDT")
    parser.add_argument("--source-coverage-id", default=None)
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--maker-fee-rate", type=float, default=None)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--execution-fee-mode", default="taker")
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--position-sizing-mode", default="fraction_of_equity")
    parser.add_argument("--position-notional-fraction", type=float, default=1.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--fixed-position-qty", type=float, default=1.0)
    return parser


def _candidate_ids_from_args(registry: CandidateRegistry, args: argparse.Namespace) -> list[str]:
    if args.candidate_id:
        return list(args.candidate_id)
    return [candidate.candidate_id for candidate in registry.list_candidates(status=args.status, tags=args.tag)]


def _run_step(
    name: str,
    cmd: list[str],
    *,
    log_path: Path,
    dry_run: bool,
) -> CycleStepRecord:
    if dry_run:
        return CycleStepRecord(name=name, command=cmd, status="planned", log_path=str(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        subprocess.run(cmd, cwd=str(ROOT), check=True, stdout=log_file, stderr=subprocess.STDOUT)
    return CycleStepRecord(name=name, command=cmd, status="completed", log_path=str(log_path))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    cycle_name = args.cycle_name
    cycle_tag = f"cycle:{cycle_name}" if args.mode == "generate" else None
    output_dir = ROOT / "checkpoints" / "continuous_cycles" / cycle_name
    output_dir.mkdir(parents=True, exist_ok=True)

    report_names = {
        "diversification": f"{cycle_name}_div",
        "clusters": f"{cycle_name}_clusters",
        "shortlist": f"{cycle_name}_shortlist",
        "combinations": f"{cycle_name}_combinations",
        "allocator": f"{cycle_name}_allocator",
        "dashboard": f"{cycle_name}_dashboard",
        "tradeforward": f"{cycle_name}_tradeforward",
    }

    steps: list[CycleStepRecord] = []
    summary_path = output_dir / "summary.json"
    candidate_ids: list[str] = []

    if args.mode == "generate":
        generator_window_starts, generator_oos_days = _generator_window_plan(args)
        probe_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "probe_monolith_walkforward.py"),
            "--data-path",
            args.data_path,
            "--train-days",
            str(args.train_days),
            "--valid-days",
            str(args.valid_days),
            "--oos-days",
            str(generator_oos_days),
            "--runs-per-window",
            str(args.runs_per_window),
            "--generations",
            str(args.generations),
            "--population-size",
            str(args.population_size),
            *([] if not args.skip_filtering_phase else ["--skip-filtering-phase"]),
            "--engine-family",
            args.engine_family,
            "--engine-hidden-dim",
            str(args.engine_hidden_dim),
            "--engine-alpha",
            str(args.engine_alpha),
            "--action-head-mode",
            args.action_head_mode,
            "--action-threshold",
            str(args.action_threshold),
            "--fitness-profile",
            args.fitness_profile,
            "--min-trades",
            str(args.min_trades),
            "--activity-target-trades",
            str(args.activity_target_trades),
            "--output-dir",
            str(output_dir),
            "--initial-balance",
            str(args.initial_balance),
            "--exchange",
            args.exchange,
            "--execution-fee-mode",
            args.execution_fee_mode,
            "--slippage",
            str(args.slippage),
            "--position-sizing-mode",
            args.position_sizing_mode,
            "--position-notional-fraction",
            str(args.position_notional_fraction),
            "--leverage",
            str(args.leverage),
            "--fixed-position-qty",
            str(args.fixed_position_qty),
        ]
        if args.trade_band_low is not None:
            probe_cmd.extend(["--trade-band-low", str(args.trade_band_low)])
        if args.trade_band_high is not None:
            probe_cmd.extend(["--trade-band-high", str(args.trade_band_high)])
        probe_cmd.extend(["--trade-band-floor", str(args.trade_band_floor)])
        for item in generator_window_starts:
            probe_cmd.extend(["--window-start", item])
        if args.maker_fee_rate is not None:
            probe_cmd.extend(["--maker-fee-rate", str(args.maker_fee_rate)])
        if args.taker_fee_rate is not None:
            probe_cmd.extend(["--taker-fee-rate", str(args.taker_fee_rate)])
        steps.append(_run_step("probe_walkforward", probe_cmd, log_path=output_dir / "01_probe.log", dry_run=args.dry_run))

        import_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "register_walkforward_candidates.py"),
            "--summary-path",
            str(summary_path),
            "--registry-root",
            args.registry_root,
            "--engine-name",
            args.engine_name,
            "--engine-role",
            args.engine_role,
            "--representation-name",
            args.representation_name,
            "--engine-family",
            args.engine_family,
            "--engine-hidden-dim",
            str(args.engine_hidden_dim),
            "--engine-alpha",
            str(args.engine_alpha),
            "--action-head-mode",
            args.action_head_mode,
            "--action-threshold",
            str(args.action_threshold),
            "--fitness-profile",
            args.fitness_profile,
            "--min-trades",
            str(args.min_trades),
            "--activity-target-trades",
            str(args.activity_target_trades),
            "--valid-days",
            str(args.valid_days),
            "--market-scope",
            args.market_scope,
            "--status",
            "research",
            "--tag",
            "continuous",
            "--tag",
            cycle_tag,
        ]
        if args.trade_band_low is not None:
            import_cmd.extend(["--trade-band-low", str(args.trade_band_low)])
        if args.trade_band_high is not None:
            import_cmd.extend(["--trade-band-high", str(args.trade_band_high)])
        import_cmd.extend(["--trade-band-floor", str(args.trade_band_floor)])
        if args.source_coverage_id:
            import_cmd.extend(["--source-coverage-id", args.source_coverage_id])
        for item in args.tag or []:
            import_cmd.extend(["--tag", item])
        if args.notes:
            import_cmd.extend(["--notes", args.notes])
        steps.append(_run_step("register_candidates", import_cmd, log_path=output_dir / "02_register.log", dry_run=args.dry_run))

        if not args.dry_run:
            candidate_ids = [candidate.candidate_id for candidate in registry.list_candidates(tags=[cycle_tag])]
    else:
        candidate_ids = _candidate_ids_from_args(registry, args)

    if not candidate_ids and not args.dry_run:
        raise ValueError("No candidates selected for the cycle")

    candidate_id_args: list[str] = []
    for candidate_id in candidate_ids:
        candidate_id_args.extend(["--candidate-id", candidate_id])

    export_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_candidate_trades.py"),
        "--registry-root",
        args.registry_root,
        *candidate_id_args,
    ]
    steps.append(_run_step("export_trades", export_cmd, log_path=output_dir / "03_export_trades.log", dry_run=args.dry_run))

    fractions = args.resampling_fraction or [1.0]
    resample_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_candidate_resampling.py"),
        "--registry-root",
        args.registry_root,
        "--period",
        "train",
        "--iterations",
        str(args.resampling_iterations),
        "--seed",
        str(args.resampling_seed),
        "--name-prefix",
        args.resampling_name_prefix,
        *candidate_id_args,
    ]
    for fraction in fractions:
        resample_cmd.extend(["--fraction", str(fraction)])
    steps.append(_run_step("train_resampling", resample_cmd, log_path=output_dir / "04_resampling.log", dry_run=args.dry_run))

    diversification_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_candidate_diversification.py"),
        "--registry-root",
        args.registry_root,
        "--start-utc",
        args.selection_start_utc,
        "--end-utc",
        args.selection_end_utc,
        "--report-name",
        report_names["diversification"],
        "--curve-points",
        str(max(args.curve_points, 512)),
        *candidate_id_args,
    ]
    steps.append(_run_step("diversification", diversification_cmd, log_path=output_dir / "05_diversification.log", dry_run=args.dry_run))

    clustering_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_candidate_clustering.py"),
        "--registry-root",
        args.registry_root,
        "--diversification-report",
        report_names["diversification"],
        "--report-name",
        report_names["clusters"],
        "--similarity-threshold",
        str(args.similarity_threshold),
    ]
    steps.append(_run_step("clusters", clustering_cmd, log_path=output_dir / "06_clusters.log", dry_run=args.dry_run))

    resampling_name = f"{args.resampling_name_prefix}_f{fractions[0]:.2f}"
    shortlist_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_candidate_shortlist.py"),
        "--registry-root",
        args.registry_root,
        "--diversification-report",
        report_names["diversification"],
        "--report-name",
        report_names["shortlist"],
        "--resampling-name",
        resampling_name,
        "--max-candidates",
        str(args.shortlist_max_candidates),
        "--min-marginal-score",
        str(args.shortlist_min_marginal_score),
        "--max-pair-downside-corr",
        str(args.shortlist_max_pair_downside_corr),
        "--max-pair-simultaneous-loss-rate",
        str(args.shortlist_max_pair_sim_loss),
    ]
    steps.append(_run_step("shortlist", shortlist_cmd, log_path=output_dir / "07_shortlist.log", dry_run=args.dry_run))

    risk_fractions = args.risk_fraction or [0.25, 0.50, 0.75]
    combinations_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_shortlist_combinations.py"),
        "--registry-root",
        args.registry_root,
        "--shortlist-report",
        report_names["shortlist"],
        "--report-name",
        report_names["combinations"],
        "--cluster-report",
        report_names["clusters"],
        "--max-pool-size",
        str(args.combination_max_pool_size),
        "--min-subset-size",
        str(args.combination_min_subset_size),
        "--max-subset-size",
        str(args.combination_max_subset_size),
        "--per-system-cap",
        str(args.per_system_cap),
        "--default-cluster-cap",
        str(args.default_cluster_cap),
        "--iterations",
        str(args.portfolio_iterations),
        "--block-size",
        str(args.portfolio_block_size),
        "--objective-max-dd",
        str(args.objective_max_dd),
        "--curve-points",
        str(args.curve_points),
    ]
    for risk in risk_fractions:
        combinations_cmd.extend(["--risk-fraction", str(risk)])
    steps.append(_run_step("combinations", combinations_cmd, log_path=output_dir / "08_combinations.log", dry_run=args.dry_run))

    allocator_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_allocator_workbench.py"),
        "--registry-root",
        args.registry_root,
        "--shortlist-report",
        report_names["shortlist"],
        "--report-name",
        report_names["allocator"],
        "--cluster-report",
        report_names["clusters"],
        "--per-system-cap",
        str(args.per_system_cap),
        "--default-cluster-cap",
        str(args.default_cluster_cap),
        "--iterations",
        str(args.portfolio_iterations),
        "--block-size",
        str(args.portfolio_block_size),
        "--objective-max-dd",
        str(args.objective_max_dd),
        "--curve-points",
        str(args.curve_points),
    ]
    for risk in risk_fractions:
        allocator_cmd.extend(["--risk-fraction", str(risk)])
    steps.append(_run_step("allocator", allocator_cmd, log_path=output_dir / "09_allocator.log", dry_run=args.dry_run))

    dashboard_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_operator_dashboard_feed.py"),
        "--registry-root",
        args.registry_root,
        "--feed-name",
        report_names["dashboard"],
        "--shortlist-report",
        report_names["shortlist"],
        "--diversification-report",
        report_names["diversification"],
        "--cluster-report",
        report_names["clusters"],
        "--allocator-report",
        report_names["allocator"],
        "--combination-report",
        report_names["combinations"],
        "--resampling-name",
        resampling_name,
    ]
    steps.append(_run_step("dashboard", dashboard_cmd, log_path=output_dir / "10_dashboard.log", dry_run=args.dry_run))

    forward_start_utc = args.forward_start_utc or args.selection_end_utc
    forward_end_utc = args.forward_end_utc or _format_utc(_parse_utc(forward_start_utc) + timedelta(days=args.forward_days))
    tradeforward_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_tradeforward_plan.py"),
        "--registry-root",
        args.registry_root,
        "--plan-name",
        report_names["tradeforward"],
        "--forward-start-utc",
        forward_start_utc,
        "--forward-end-utc",
        forward_end_utc,
    ]
    if args.tradeforward_mode == "combination":
        tradeforward_cmd.extend(["--combination-report", report_names["combinations"]])
    else:
        tradeforward_cmd.extend(["--allocator-report", report_names["allocator"]])
    if args.notes:
        tradeforward_cmd.extend(["--notes", args.notes])
    steps.append(_run_step("tradeforward", tradeforward_cmd, log_path=output_dir / "11_tradeforward.log", dry_run=args.dry_run))

    if not args.dry_run and args.mode == "reuse":
        candidate_ids = _candidate_ids_from_args(registry, args)

    cycle_report = ContinuousSearchCycleReport(
        schema_version="1",
        name=cycle_name,
        created_at_utc=utc_now_text(),
        mode=args.mode,
        output_dir=str(output_dir),
        cycle_tag=cycle_tag,
        source_summary_path=None if args.mode != "generate" else str(summary_path),
        candidate_ids=candidate_ids,
        report_names=report_names,
        steps=steps,
        notes=args.notes,
    )
    cycle_path = registry.save_cycle_report(cycle_report)

    if not args.dry_run:
        cycle_ref = cycle_report.name
        combination_report = registry.load_combination_search(report_names["combinations"])
        allocator_report = registry.load_allocator_workbench(report_names["allocator"])
        tradeforward = build_tradeforward_plan(
            registry,
            report_names["tradeforward"],
            forward_start_utc=forward_start_utc,
            forward_end_utc=forward_end_utc,
            allocator_report=allocator_report if args.tradeforward_mode == "allocator" else None,
            combination_report=combination_report if args.tradeforward_mode == "combination" else None,
            source_cycle_report=cycle_ref,
            notes=args.notes,
        )
        registry.save_tradeforward_plan(tradeforward)

    print(f"Saved cycle report to {cycle_path}")
    print(f"mode={args.mode} steps={len(steps)} dry_run={args.dry_run}")
    if candidate_ids:
        print(f"candidates={len(candidate_ids)}")
    for step in steps:
        print(f"{step.status:9} {step.name:<16} {Path(step.log_path).name if step.log_path else '-'}")


if __name__ == "__main__":
    main()
