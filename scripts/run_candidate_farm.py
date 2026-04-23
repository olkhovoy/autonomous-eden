#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import (
    CandidateRegistry,
    FarmProgressEvent,
    FarmProgressLog,
    build_candidate_farm_report,
    build_farm_dashboard_feed,
    build_candidate_farm_scenario_report,
    expand_farm_manifest_scenarios,
    utc_now_text,
)


MANAGED_ROLLING_KEYS = {"registry_root", "report_name", "notes"}
MANAGED_LIFECYCLE_KEYS = {"registry_root", "rolling_report", "report_name", "notes"}
MANAGED_LEDGER_KEYS = {"registry_root", "rolling_report", "lifecycle_report", "report_name", "notes"}
MANAGED_BASELINE_KEYS = {"registry_root", "portfolio_ledger_report", "report_name", "notes"}


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("._") or "scenario"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a manifest-driven candidate farm over multiple rolling conveyor scenarios."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--registry-root", default=None)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=30.0)
    parser.add_argument("--dashboard-feed-name", default=None)
    parser.add_argument("--dashboard-sync-path", default=None)
    parser.add_argument("--dashboard-max-scenarios", type=int, default=500)
    parser.add_argument("--dashboard-max-broom-lines", type=int, default=240)
    parser.add_argument("--notes", default=None)
    return parser


def _scenario_notes(global_notes: str | None, scenario_notes: str | None, cli_notes: str | None) -> str | None:
    parts = [item for item in (global_notes, scenario_notes, cli_notes) if item]
    if not parts:
        return None
    return " | ".join(parts)


def _args_from_mapping(mapping: dict[str, Any], *, managed_keys: set[str]) -> list[str]:
    args: list[str] = []
    for key, value in mapping.items():
        if key in managed_keys:
            raise ValueError(f"Manifest key '{key}' is managed by the farm runner and must not be set explicitly")
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def _run_logged(cmd: list[str], *, log_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        subprocess.run(cmd, cwd=str(ROOT), check=True, stdout=log_file, stderr=subprocess.STDOUT)


def _run_logged_with_heartbeat(
    cmd: list[str],
    *,
    log_path: Path,
    dry_run: bool,
    heartbeat_interval_seconds: float,
    on_heartbeat: Callable[[], None] | None,
) -> None:
    if dry_run:
        return
    if heartbeat_interval_seconds <= 0 or on_heartbeat is None:
        _run_logged(cmd, log_path=log_path, dry_run=dry_run)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT)
        last_heartbeat = time.monotonic()
        while True:
            result = process.poll()
            if result is not None:
                if result != 0:
                    raise subprocess.CalledProcessError(result, cmd)
                return
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval_seconds:
                on_heartbeat()
                last_heartbeat = now
            time.sleep(min(1.0, heartbeat_interval_seconds))


def _load_optional_artifacts(
    registry: CandidateRegistry,
    *,
    rolling_name: str,
    lifecycle_name: str,
    ledger_name: str,
    baseline_name: str,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    rolling = registry.load_rolling_conveyor(rolling_name) if registry.rolling_path(rolling_name).exists() else None
    lifecycle = registry.load_lifecycle_report(lifecycle_name) if registry.lifecycle_path(lifecycle_name).exists() else None
    ledger = registry.load_portfolio_ledger(ledger_name) if registry.portfolio_ledger_path(ledger_name).exists() else None
    baselines = (
        registry.load_portfolio_baselines(baseline_name)
        if registry.portfolio_baselines_path(baseline_name).exists()
        else None
    )
    return rolling, lifecycle, ledger, baselines


def _save_farm_snapshot(
    registry: CandidateRegistry,
    *,
    farm_name: str,
    registry_root: str,
    scenario_reports: list[Any],
    source_manifest_path: Path,
    notes: str | None,
    progress_events: list[FarmProgressEvent] | None = None,
    dashboard_feed_name: str | None = None,
    dashboard_sync_path: Path | None = None,
    dashboard_max_scenarios: int | None = 500,
    dashboard_max_broom_lines: int | None = 240,
) -> None:
    report = build_candidate_farm_report(
        name=farm_name,
        registry_root=str((ROOT / registry_root).resolve()),
        scenarios=scenario_reports,
        source_manifest_path=str(source_manifest_path),
        notes=notes,
    )
    registry.save_farm_report(report)
    if progress_events:
        progress_log = FarmProgressLog(
            schema_version="1",
            farm_name=farm_name,
            created_at_utc=progress_events[0].created_at_utc,
            updated_at_utc=progress_events[-1].created_at_utc,
            events=list(progress_events),
            notes=notes,
        )
        registry.save_farm_progress_log(progress_log)
    if dashboard_feed_name:
        feed = build_farm_dashboard_feed(
            registry,
            dashboard_feed_name,
            farm_report=report,
            max_scenarios=dashboard_max_scenarios,
            max_broom_lines=dashboard_max_broom_lines,
            notes=notes,
        )
        registry.save_farm_dashboard_feed(feed)
        if dashboard_sync_path is not None:
            dashboard_sync_path.parent.mkdir(parents=True, exist_ok=True)
            dashboard_sync_path.write_text(json.dumps(feed.to_dict(), indent=2, sort_keys=True) + "\n")


def _append_progress_event(
    progress_events: list[FarmProgressEvent],
    *,
    scenario_report: Any,
    event_kind: str,
    note: str | None = None,
) -> None:
    progress_events.append(
        FarmProgressEvent(
            sequence=len(progress_events) + 1,
            created_at_utc=str(scenario_report.updated_at_utc),
            scenario_name=str(scenario_report.scenario_name),
            status=str(scenario_report.status),
            progress_stage=scenario_report.progress_stage,
            event_kind=event_kind,
            gate_pass=scenario_report.gate_pass,
            total_pnl=scenario_report.total_pnl,
            note=note,
        )
    )


def _all_scenario_artifacts_exist(
    registry: CandidateRegistry,
    *,
    rolling_name: str,
    lifecycle_name: str,
    ledger_name: str,
    baseline_name: str,
) -> bool:
    return (
        registry.rolling_path(rolling_name).exists()
        and registry.lifecycle_path(lifecycle_name).exists()
        and registry.portfolio_ledger_path(ledger_name).exists()
        and registry.portfolio_baselines_path(baseline_name).exists()
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    farm_name = args.report_name or manifest.get("name") or manifest.get("farm_name") or manifest_path.stem
    registry_root = args.registry_root or manifest.get("registry_root") or "candidate_registry"
    dashboard_feed_name = args.dashboard_feed_name or f"{farm_name}_feed"
    dashboard_sync_path = None if args.dashboard_sync_path is None else (ROOT / args.dashboard_sync_path).resolve()
    dashboard_max_scenarios = None if args.dashboard_max_scenarios <= 0 else args.dashboard_max_scenarios
    dashboard_max_broom_lines = None if args.dashboard_max_broom_lines <= 0 else args.dashboard_max_broom_lines
    registry = CandidateRegistry(ROOT / registry_root)
    global_notes = manifest.get("notes")
    scenarios = expand_farm_manifest_scenarios(manifest)
    if not scenarios:
        raise ValueError("Manifest contains no scenarios")
    scenario_names = [str(item["name"]) for item in scenarios]
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError("Manifest expands to duplicate scenario names")

    farm_output_dir = ROOT / "checkpoints" / "candidate_farms" / _safe_name(farm_name)
    farm_output_dir.mkdir(parents=True, exist_ok=True)

    scenario_reports = []
    scenario_runtime: list[dict[str, Any]] = []
    progress_events: list[FarmProgressEvent] = []
    for scenario in scenarios:
        scenario_name = str(scenario["name"])
        safe_scenario_name = _safe_name(scenario_name)
        output_dir = farm_output_dir / safe_scenario_name
        output_dir.mkdir(parents=True, exist_ok=True)

        rolling_args = dict(scenario.get("rolling_args", {}))
        lifecycle_args = dict(scenario.get("lifecycle_args", {}))
        ledger_args = dict(scenario.get("ledger_args", {}))
        baseline_args = dict(scenario.get("baseline_args", {}))
        forwarded_cycle_args = [str(item) for item in scenario.get("forwarded_cycle_args", [])]
        notes = _scenario_notes(global_notes, scenario.get("notes"), args.notes)
        mode = str(rolling_args.get("mode", "reuse"))

        rolling_name = f"{farm_name}__{safe_scenario_name}__rolling"
        lifecycle_name = f"{farm_name}__{safe_scenario_name}__lifecycle"
        ledger_name = f"{farm_name}__{safe_scenario_name}__portfolio"
        baseline_name = f"{farm_name}__{safe_scenario_name}__baselines"
        log_paths = {
            "rolling": str(output_dir / "01_rolling.log"),
            "lifecycle": str(output_dir / "02_lifecycle.log"),
            "portfolio_ledger": str(output_dir / "03_portfolio_ledger.log"),
            "portfolio_baselines": str(output_dir / "04_portfolio_baselines.log"),
        }
        config = {
            "rolling_args": rolling_args,
            "forwarded_cycle_args": forwarded_cycle_args,
            "lifecycle_args": lifecycle_args,
            "ledger_args": ledger_args,
            "baseline_args": baseline_args,
        }
        scenario_runtime.append(
            {
                "scenario_name": scenario_name,
                "safe_scenario_name": safe_scenario_name,
                "output_dir": output_dir,
                "rolling_args": rolling_args,
                "lifecycle_args": lifecycle_args,
                "ledger_args": ledger_args,
                "baseline_args": baseline_args,
                "forwarded_cycle_args": forwarded_cycle_args,
                "notes": notes,
                "mode": mode,
                "rolling_name": rolling_name,
                "lifecycle_name": lifecycle_name,
                "ledger_name": ledger_name,
                "baseline_name": baseline_name,
                "log_paths": log_paths,
                "config": config,
            }
        )
        initial_report = build_candidate_farm_scenario_report(
            scenario_name=scenario_name,
            status="planned",
            mode=mode,
            output_dir=str(output_dir),
            config=config,
            log_paths=log_paths,
            updated_at_utc=None,
            progress_stage="queued",
            rolling_report_name=rolling_name,
            lifecycle_report_name=lifecycle_name,
            portfolio_ledger_report_name=ledger_name,
            portfolio_baselines_report_name=baseline_name,
            notes=notes,
        )
        scenario_reports.append(initial_report)
        _append_progress_event(progress_events, scenario_report=initial_report, event_kind="queued")

    _save_farm_snapshot(
        registry,
        farm_name=farm_name,
        registry_root=registry_root,
        scenario_reports=scenario_reports,
        source_manifest_path=manifest_path,
        notes=_scenario_notes(global_notes, None, args.notes),
        progress_events=progress_events,
        dashboard_feed_name=dashboard_feed_name,
        dashboard_sync_path=dashboard_sync_path,
        dashboard_max_scenarios=dashboard_max_scenarios,
        dashboard_max_broom_lines=dashboard_max_broom_lines,
    )

    for index, scenario in enumerate(scenario_runtime):
        scenario_name = scenario["scenario_name"]
        safe_scenario_name = scenario["safe_scenario_name"]
        output_dir = scenario["output_dir"]
        rolling_args = scenario["rolling_args"]
        lifecycle_args = scenario["lifecycle_args"]
        ledger_args = scenario["ledger_args"]
        baseline_args = scenario["baseline_args"]
        forwarded_cycle_args = scenario["forwarded_cycle_args"]
        notes = scenario["notes"]
        mode = scenario["mode"]
        rolling_name = scenario["rolling_name"]
        lifecycle_name = scenario["lifecycle_name"]
        ledger_name = scenario["ledger_name"]
        baseline_name = scenario["baseline_name"]
        log_paths = scenario["log_paths"]
        config = scenario["config"]

        def emit_heartbeat() -> None:
            scenario_reports[index].updated_at_utc = utc_now_text()
            _append_progress_event(progress_events, scenario_report=scenario_reports[index], event_kind="heartbeat")
            _save_farm_snapshot(
                registry,
                farm_name=farm_name,
                registry_root=registry_root,
                scenario_reports=scenario_reports,
                source_manifest_path=manifest_path,
                notes=_scenario_notes(global_notes, None, args.notes),
                progress_events=progress_events,
                dashboard_feed_name=dashboard_feed_name,
                dashboard_sync_path=dashboard_sync_path,
                dashboard_max_scenarios=dashboard_max_scenarios,
                dashboard_max_broom_lines=dashboard_max_broom_lines,
            )
        try:
            if args.resume_completed and not args.dry_run and _all_scenario_artifacts_exist(
                registry,
                rolling_name=rolling_name,
                lifecycle_name=lifecycle_name,
                ledger_name=ledger_name,
                baseline_name=baseline_name,
            ):
                rolling_report, lifecycle_report, ledger_report, baseline_report = _load_optional_artifacts(
                    registry,
                    rolling_name=rolling_name,
                    lifecycle_name=lifecycle_name,
                    ledger_name=ledger_name,
                    baseline_name=baseline_name,
                )
                scenario_reports[index] = build_candidate_farm_scenario_report(
                    scenario_name=scenario_name,
                    status="reused",
                    mode=mode,
                    output_dir=str(output_dir),
                    config=config,
                    log_paths=log_paths,
                    progress_stage="completed",
                    rolling_report=rolling_report,
                    lifecycle_report=lifecycle_report,
                    portfolio_ledger_report=ledger_report,
                    portfolio_baselines_report=baseline_report,
                    notes=notes,
                )
                _append_progress_event(progress_events, scenario_report=scenario_reports[index], event_kind="reused")
                _save_farm_snapshot(
                    registry,
                    farm_name=farm_name,
                    registry_root=registry_root,
                    scenario_reports=scenario_reports,
                    source_manifest_path=manifest_path,
                    notes=_scenario_notes(global_notes, None, args.notes),
                    progress_events=progress_events,
                    dashboard_feed_name=dashboard_feed_name,
                    dashboard_sync_path=dashboard_sync_path,
                    dashboard_max_scenarios=dashboard_max_scenarios,
                    dashboard_max_broom_lines=dashboard_max_broom_lines,
                )
                print(
                    f"[reused] {scenario_name} gate={baseline_report.gate.overall_pass if baseline_report else None} "
                    f"pnl={baseline_report.conveyor_total_pnl if baseline_report else float('nan'):+.2f}"
                )
                continue

            scenario_reports[index] = build_candidate_farm_scenario_report(
                scenario_name=scenario_name,
                status="running",
                mode=mode,
                output_dir=str(output_dir),
                config=config,
                log_paths=log_paths,
                progress_stage="rolling",
                rolling_report_name=rolling_name,
                lifecycle_report_name=lifecycle_name,
                portfolio_ledger_report_name=ledger_name,
                portfolio_baselines_report_name=baseline_name,
                notes=notes,
            )
            _append_progress_event(progress_events, scenario_report=scenario_reports[index], event_kind="stage")
            _save_farm_snapshot(
                registry,
                farm_name=farm_name,
                registry_root=registry_root,
                scenario_reports=scenario_reports,
                source_manifest_path=manifest_path,
                notes=_scenario_notes(global_notes, None, args.notes),
                progress_events=progress_events,
                dashboard_feed_name=dashboard_feed_name,
                dashboard_sync_path=dashboard_sync_path,
                dashboard_max_scenarios=dashboard_max_scenarios,
                dashboard_max_broom_lines=dashboard_max_broom_lines,
            )

            rolling_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_rolling_conveyor_simulator.py"),
                "--registry-root",
                registry_root,
                "--report-name",
                rolling_name,
                *_args_from_mapping(rolling_args, managed_keys=MANAGED_ROLLING_KEYS),
                *forwarded_cycle_args,
            ]
            if notes:
                rolling_cmd.extend(["--notes", notes])
            _run_logged_with_heartbeat(
                rolling_cmd,
                log_path=Path(log_paths["rolling"]),
                dry_run=args.dry_run,
                heartbeat_interval_seconds=args.heartbeat_interval_seconds,
                on_heartbeat=emit_heartbeat,
            )

            if args.dry_run:
                scenario_reports[index] = build_candidate_farm_scenario_report(
                    scenario_name=scenario_name,
                    status="planned",
                    mode=mode,
                    output_dir=str(output_dir),
                    config=config,
                    log_paths=log_paths,
                    progress_stage="rolling_planned",
                    rolling_report_name=rolling_name,
                    lifecycle_report_name=lifecycle_name,
                    portfolio_ledger_report_name=ledger_name,
                    portfolio_baselines_report_name=baseline_name,
                    notes=notes,
                )
                _append_progress_event(progress_events, scenario_report=scenario_reports[index], event_kind="dry_run")
                _save_farm_snapshot(
                    registry,
                    farm_name=farm_name,
                    registry_root=registry_root,
                    scenario_reports=scenario_reports,
                    source_manifest_path=manifest_path,
                    notes=_scenario_notes(global_notes, None, args.notes),
                    progress_events=progress_events,
                    dashboard_feed_name=dashboard_feed_name,
                    dashboard_sync_path=dashboard_sync_path,
                    dashboard_max_scenarios=dashboard_max_scenarios,
                    dashboard_max_broom_lines=dashboard_max_broom_lines,
                )
                continue

            scenario_reports[index] = build_candidate_farm_scenario_report(
                scenario_name=scenario_name,
                status="running",
                mode=mode,
                output_dir=str(output_dir),
                config=config,
                log_paths=log_paths,
                progress_stage="lifecycle",
                rolling_report_name=rolling_name,
                lifecycle_report_name=lifecycle_name,
                portfolio_ledger_report_name=ledger_name,
                portfolio_baselines_report_name=baseline_name,
                notes=notes,
            )
            _append_progress_event(progress_events, scenario_report=scenario_reports[index], event_kind="stage")
            _save_farm_snapshot(
                registry,
                farm_name=farm_name,
                registry_root=registry_root,
                scenario_reports=scenario_reports,
                source_manifest_path=manifest_path,
                notes=_scenario_notes(global_notes, None, args.notes),
                progress_events=progress_events,
                dashboard_feed_name=dashboard_feed_name,
                dashboard_sync_path=dashboard_sync_path,
                dashboard_max_scenarios=dashboard_max_scenarios,
                dashboard_max_broom_lines=dashboard_max_broom_lines,
            )

            lifecycle_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_lifecycle_state_machine.py"),
                "--registry-root",
                registry_root,
                "--rolling-report",
                rolling_name,
                "--report-name",
                lifecycle_name,
                *_args_from_mapping(lifecycle_args, managed_keys=MANAGED_LIFECYCLE_KEYS),
            ]
            if notes:
                lifecycle_cmd.extend(["--notes", notes])
            _run_logged_with_heartbeat(
                lifecycle_cmd,
                log_path=Path(log_paths["lifecycle"]),
                dry_run=False,
                heartbeat_interval_seconds=args.heartbeat_interval_seconds,
                on_heartbeat=emit_heartbeat,
            )

            scenario_reports[index] = build_candidate_farm_scenario_report(
                scenario_name=scenario_name,
                status="running",
                mode=mode,
                output_dir=str(output_dir),
                config=config,
                log_paths=log_paths,
                progress_stage="portfolio_ledger",
                rolling_report_name=rolling_name,
                lifecycle_report_name=lifecycle_name,
                portfolio_ledger_report_name=ledger_name,
                portfolio_baselines_report_name=baseline_name,
                notes=notes,
            )
            _append_progress_event(progress_events, scenario_report=scenario_reports[index], event_kind="stage")
            _save_farm_snapshot(
                registry,
                farm_name=farm_name,
                registry_root=registry_root,
                scenario_reports=scenario_reports,
                source_manifest_path=manifest_path,
                notes=_scenario_notes(global_notes, None, args.notes),
                progress_events=progress_events,
                dashboard_feed_name=dashboard_feed_name,
                dashboard_sync_path=dashboard_sync_path,
                dashboard_max_scenarios=dashboard_max_scenarios,
                dashboard_max_broom_lines=dashboard_max_broom_lines,
            )

            ledger_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_portfolio_ledger.py"),
                "--registry-root",
                registry_root,
                "--rolling-report",
                rolling_name,
                "--lifecycle-report",
                lifecycle_name,
                "--report-name",
                ledger_name,
                *_args_from_mapping(ledger_args, managed_keys=MANAGED_LEDGER_KEYS),
            ]
            if notes:
                ledger_cmd.extend(["--notes", notes])
            _run_logged_with_heartbeat(
                ledger_cmd,
                log_path=Path(log_paths["portfolio_ledger"]),
                dry_run=False,
                heartbeat_interval_seconds=args.heartbeat_interval_seconds,
                on_heartbeat=emit_heartbeat,
            )

            scenario_reports[index] = build_candidate_farm_scenario_report(
                scenario_name=scenario_name,
                status="running",
                mode=mode,
                output_dir=str(output_dir),
                config=config,
                log_paths=log_paths,
                progress_stage="portfolio_baselines",
                rolling_report_name=rolling_name,
                lifecycle_report_name=lifecycle_name,
                portfolio_ledger_report_name=ledger_name,
                portfolio_baselines_report_name=baseline_name,
                notes=notes,
            )
            _append_progress_event(progress_events, scenario_report=scenario_reports[index], event_kind="stage")
            _save_farm_snapshot(
                registry,
                farm_name=farm_name,
                registry_root=registry_root,
                scenario_reports=scenario_reports,
                source_manifest_path=manifest_path,
                notes=_scenario_notes(global_notes, None, args.notes),
                progress_events=progress_events,
                dashboard_feed_name=dashboard_feed_name,
                dashboard_sync_path=dashboard_sync_path,
                dashboard_max_scenarios=dashboard_max_scenarios,
                dashboard_max_broom_lines=dashboard_max_broom_lines,
            )

            baseline_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_portfolio_baselines.py"),
                "--registry-root",
                registry_root,
                "--portfolio-ledger-report",
                ledger_name,
                "--report-name",
                baseline_name,
                *_args_from_mapping(baseline_args, managed_keys=MANAGED_BASELINE_KEYS),
            ]
            if notes:
                baseline_cmd.extend(["--notes", notes])
            _run_logged_with_heartbeat(
                baseline_cmd,
                log_path=Path(log_paths["portfolio_baselines"]),
                dry_run=False,
                heartbeat_interval_seconds=args.heartbeat_interval_seconds,
                on_heartbeat=emit_heartbeat,
            )

            rolling_report, lifecycle_report, ledger_report, baseline_report = _load_optional_artifacts(
                registry,
                rolling_name=rolling_name,
                lifecycle_name=lifecycle_name,
                ledger_name=ledger_name,
                baseline_name=baseline_name,
            )
            scenario_reports[index] = build_candidate_farm_scenario_report(
                scenario_name=scenario_name,
                status="completed",
                mode=mode,
                output_dir=str(output_dir),
                config=config,
                log_paths=log_paths,
                progress_stage="completed",
                rolling_report=rolling_report,
                lifecycle_report=lifecycle_report,
                portfolio_ledger_report=ledger_report,
                portfolio_baselines_report=baseline_report,
                notes=notes,
            )
            _append_progress_event(progress_events, scenario_report=scenario_reports[index], event_kind="completed")
            _save_farm_snapshot(
                registry,
                farm_name=farm_name,
                registry_root=registry_root,
                scenario_reports=scenario_reports,
                source_manifest_path=manifest_path,
                notes=_scenario_notes(global_notes, None, args.notes),
                progress_events=progress_events,
                dashboard_feed_name=dashboard_feed_name,
                dashboard_sync_path=dashboard_sync_path,
                dashboard_max_scenarios=dashboard_max_scenarios,
                dashboard_max_broom_lines=dashboard_max_broom_lines,
            )
            print(
                f"[completed] {scenario_name} gate={baseline_report.gate.overall_pass if baseline_report else None} "
                f"pnl={baseline_report.conveyor_total_pnl if baseline_report else float('nan'):+.2f}"
            )
        except Exception as exc:
            rolling_report, lifecycle_report, ledger_report, baseline_report = _load_optional_artifacts(
                registry,
                rolling_name=rolling_name,
                lifecycle_name=lifecycle_name,
                ledger_name=ledger_name,
                baseline_name=baseline_name,
            )
            scenario_reports[index] = build_candidate_farm_scenario_report(
                scenario_name=scenario_name,
                status="failed",
                mode=mode,
                output_dir=str(output_dir),
                config=config,
                log_paths=log_paths,
                progress_stage="failed",
                rolling_report=rolling_report,
                lifecycle_report=lifecycle_report,
                portfolio_ledger_report=ledger_report,
                portfolio_baselines_report=baseline_report,
                rolling_report_name=rolling_name,
                lifecycle_report_name=lifecycle_name,
                portfolio_ledger_report_name=ledger_name,
                portfolio_baselines_report_name=baseline_name,
                error_message=str(exc),
                notes=notes,
            )
            _append_progress_event(progress_events, scenario_report=scenario_reports[index], event_kind="failed", note=str(exc))
            _save_farm_snapshot(
                registry,
                farm_name=farm_name,
                registry_root=registry_root,
                scenario_reports=scenario_reports,
                source_manifest_path=manifest_path,
                notes=_scenario_notes(global_notes, None, args.notes),
                progress_events=progress_events,
                dashboard_feed_name=dashboard_feed_name,
                dashboard_sync_path=dashboard_sync_path,
                dashboard_max_scenarios=dashboard_max_scenarios,
                dashboard_max_broom_lines=dashboard_max_broom_lines,
            )
            print(f"[failed] {scenario_name}: {exc}")
            if not args.continue_on_error:
                raise

    farm_report = build_candidate_farm_report(
        name=farm_name,
        registry_root=str((ROOT / registry_root).resolve()),
        scenarios=scenario_reports,
        source_manifest_path=str(manifest_path),
        notes=_scenario_notes(global_notes, None, args.notes),
    )
    path = registry.save_farm_report(farm_report)
    print(f"Saved candidate farm report to {path}")
    print(
        f"scenarios={farm_report.summary['scenario_count']} "
        f"completed={farm_report.summary['completed_or_reused_scenarios']} "
        f"gate_pass={farm_report.summary['gate_pass_count']} "
        f"unique_pool={farm_report.summary['total_unique_candidate_pool_ids']}"
    )
    if farm_report.summary.get("best_scenario_by_pnl"):
        print(f"best_by_pnl={farm_report.summary['best_scenario_by_pnl']}")


if __name__ == "__main__":
    main()
