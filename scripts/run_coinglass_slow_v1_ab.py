#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BRANCH_BASELINE = "baseline_market_only_v1"
BRANCH_COINGLASS = "coinglass_slow_v1"


def _branch_config(branch: str) -> dict[str, str]:
    if branch == BRANCH_BASELINE:
        return {
            "data_dir": "data/generated/BTCUSDT/baseline_market_only_v1/data",
            "cache_dir": "cache/neurobars_autoresearch/baseline_market_only_v1",
            "checkpoint_path": "checkpoints/neurobars_autoresearch_baseline_market_only_v1.pt",
            "metrics_path": "checkpoints/neurobars_autoresearch_baseline_market_only_v1_metrics.json",
            "npz_path": "data/BTCUSDT_parquet_neurobars_baseline_market_only_v1.npz",
            "representation_name": "baseline_market_only_v1_fused32",
        }
    if branch == BRANCH_COINGLASS:
        return {
            "data_dir": "data/generated/BTCUSDT/coinglass_slow_v1/data",
            "cache_dir": "cache/neurobars_autoresearch/coinglass_slow_v1",
            "checkpoint_path": "checkpoints/neurobars_autoresearch_coinglass_slow_v1.pt",
            "metrics_path": "checkpoints/neurobars_autoresearch_coinglass_slow_v1_metrics.json",
            "npz_path": "data/BTCUSDT_parquet_neurobars_coinglass_slow_v1.npz",
            "representation_name": "coinglass_slow_v1_fused32",
        }
    raise ValueError(f"Unsupported branch: {branch}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CoinGlass slow-context v1 A/B ladder.")
    parser.add_argument("--branch", choices=[BRANCH_BASELINE, BRANCH_COINGLASS, "both"], default="both")
    parser.add_argument("--stage", choices=["build", "train", "export", "probe", "rolling", "all"], default="all")
    parser.add_argument("--source-mode", choices=["legacy", "live"], default="legacy")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--selection-start-utc", default="2025-05-01 00:00:00")
    parser.add_argument("--selection-days", type=int, default=7)
    parser.add_argument("--forward-days", type=int, default=7)
    parser.add_argument("--num-cycles", type=int, default=2)
    parser.add_argument("--runs-per-window", type=int, default=2)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--time-budget", type=int, default=300)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--notes")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _selected_branches(branch_arg: str) -> list[str]:
    if branch_arg == "both":
        return [BRANCH_BASELINE, BRANCH_COINGLASS]
    return [branch_arg]


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def _run_build(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_coinglass_slow_context_v1.py"),
        "--source-mode",
        args.source_mode,
    ]
    _run(cmd)


def _run_train(branch: str, args: argparse.Namespace) -> None:
    cfg = _branch_config(branch)
    env = os.environ.copy()
    env["NEUROBARS_AUTORESEARCH_DATA_DIR"] = str((ROOT / cfg["data_dir"]).resolve())
    env["NEUROBARS_AUTORESEARCH_CACHE_DIR"] = str((ROOT / cfg["cache_dir"]).resolve())
    env["NEUROBARS_AUTORESEARCH_CHECKPOINT_PATH"] = str((ROOT / cfg["checkpoint_path"]).resolve())
    env["NEUROBARS_AUTORESEARCH_METRICS_PATH"] = str((ROOT / cfg["metrics_path"]).resolve())
    env["NEUROBARS_AUTORESEARCH_TIME_BUDGET"] = str(args.time_budget)
    _run([sys.executable, str(ROOT / "experiments" / "neurobars_autoresearch" / "train.py")], env=env)


def _run_export(branch: str) -> None:
    cfg = _branch_config(branch)
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "neurobars_autoresearch" / "export_neurobars.py"),
        "--checkpoint-path",
        cfg["checkpoint_path"],
        "--data-dir",
        cfg["data_dir"],
        "--output-path",
        cfg["npz_path"],
        "--cache-dir",
        cfg["cache_dir"],
    ]
    _run(cmd)


def _run_probe(branch: str, args: argparse.Namespace) -> None:
    cfg = _branch_config(branch)
    probe_dir = ROOT / "checkpoints" / "monolith_walkforward_probe" / branch
    probe_dir.mkdir(parents=True, exist_ok=True)
    summary_path = probe_dir / "summary.json"
    probe_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "probe_monolith_walkforward.py"),
        "--data-path",
        cfg["npz_path"],
        "--runs-per-window",
        str(args.runs_per_window),
        "--generations",
        str(args.generations),
        "--population-size",
        str(args.population_size),
        "--exchange",
        args.exchange,
        "--output-dir",
        str(probe_dir),
    ]
    _run(probe_cmd)

    import_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "register_walkforward_candidates.py"),
        "--summary-path",
        str(summary_path),
        "--registry-root",
        args.registry_root,
        "--representation-name",
        cfg["representation_name"],
        "--tag",
        "coinglass_ab_v1",
        "--tag",
        branch,
    ]
    if args.notes:
        import_cmd.extend(["--notes", args.notes])
    if args.overwrite:
        import_cmd.append("--overwrite")
    _run(import_cmd)


def _run_rolling(branch: str, args: argparse.Namespace) -> None:
    cfg = _branch_config(branch)
    report_name = f"{branch}_rolling"
    lifecycle_name = f"{branch}_lifecycle"
    ledger_name = f"{branch}_portfolio"
    baselines_name = f"{branch}_baselines"

    rolling_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_rolling_conveyor_simulator.py"),
        "--registry-root",
        args.registry_root,
        "--report-name",
        report_name,
        "--mode",
        "reuse",
        "--selection-start-utc",
        args.selection_start_utc,
        "--selection-days",
        str(args.selection_days),
        "--forward-days",
        str(args.forward_days),
        "--num-cycles",
        str(args.num_cycles),
        "--tag",
        branch,
        "--tag",
        "coinglass_ab_v1",
    ]
    if args.notes:
        rolling_cmd.extend(["--notes", args.notes])
    _run(rolling_cmd)

    lifecycle_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_lifecycle_state_machine.py"),
        "--registry-root",
        args.registry_root,
        "--rolling-report",
        report_name,
        "--report-name",
        lifecycle_name,
    ]
    if args.notes:
        lifecycle_cmd.extend(["--notes", args.notes])
    _run(lifecycle_cmd)

    ledger_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_portfolio_ledger.py"),
        "--registry-root",
        args.registry_root,
        "--rolling-report",
        report_name,
        "--lifecycle-report",
        lifecycle_name,
        "--report-name",
        ledger_name,
    ]
    if args.notes:
        ledger_cmd.extend(["--notes", args.notes])
    _run(ledger_cmd)

    baselines_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_portfolio_baselines.py"),
        "--registry-root",
        args.registry_root,
        "--portfolio-ledger-report",
        ledger_name,
        "--report-name",
        baselines_name,
    ]
    if args.notes:
        baselines_cmd.extend(["--notes", args.notes])
    _run(baselines_cmd)

    print(
        f"[{branch}] completed rolling gate ladder "
        f"data_path={cfg['npz_path']} report={baselines_name}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    stages = ["build", "train", "export", "probe", "rolling"] if args.stage == "all" else [args.stage]
    branches = _selected_branches(args.branch)

    if "build" in stages:
        _run_build(args)

    for branch in branches:
        if "train" in stages:
            _run_train(branch, args)
        if "export" in stages:
            _run_export(branch)
        if "probe" in stages:
            _run_probe(branch, args)
        if "rolling" in stages:
            _run_rolling(branch, args)


if __name__ == "__main__":
    main()
