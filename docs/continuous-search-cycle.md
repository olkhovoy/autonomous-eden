# Continuous Search Cycle

Last updated: 2026-03-10

## Purpose

The research conveyor needs one repeatable operational unit.
That unit is the continuous search cycle.

Each cycle should:

1. source candidates from either generation or an existing pool
2. refresh downstream robustness and diversification artifacts
3. rebuild shortlist, combinations, allocator, and dashboard
4. emit a tradeforward handoff artifact for the next unseen window

This is now the per-window building block used by the rolling conveyor
simulator.

## Current Implementation

Launcher:

1. [run_continuous_search_cycle.py](/home/user/mcs/scripts/run_continuous_search_cycle.py)

Registry artifact directory:

1. [candidate_registry/cycles](/home/user/mcs/candidate_registry/cycles)

Current cycle record schema:

1. [schema.py](/home/user/mcs/umc_nn/candidates/schema.py)

## Modes

### `reuse`

Use an existing candidate pool selected by:

1. `--candidate-id`
2. `--tag`
3. `--status`

This is the safest smoke-test mode and the best way to operate on an already
generated research pool.

### `generate`

Run the current walk-forward probe first, then import candidates and continue
through the same downstream pipeline.

By default, `generate` is now `selection_anchored`.
That means if explicit `--window-start` values are not provided, the cycle
derives probe windows from the current selection span:

1. adjacent train/OOS windows are aligned to the current review period
2. `--generator-window-offset-days 0` means OOS is exactly the selection window
3. negative offsets allow older adjacent windows without leaking future review data

This keeps generated candidates tied to the current historical conveyor window
instead of falling back to unrelated hard-coded probe dates.

## Real Smoke Run

Current real cycle:

1. [reuse_cycle_20250508.json](/home/user/mcs/candidate_registry/cycles/reuse_cycle_20250508.json)

What it produced:

1. refreshed diversification, cluster, shortlist, combination, allocator, and
   dashboard artifacts
2. a tradeforward handoff plan:
   [reuse_cycle_20250508_tradeforward.json](/home/user/mcs/candidate_registry/tradeforward/reuse_cycle_20250508_tradeforward.json)
3. a refreshed dashboard feed:
   [reuse_cycle_20250508_dashboard.json](/home/user/mcs/candidate_registry/dashboard/reuse_cycle_20250508_dashboard.json)

Current rolling use of this cycle layer:

1. [reuse_rolling_20250501.json](/home/user/mcs/candidate_registry/rolling/reuse_rolling_20250501.json)

Current generate dry-run artifact:

1. [generate_anchor_dryrun_20250508.json](/home/user/mcs/candidate_registry/cycles/generate_anchor_dryrun_20250508.json)

## Usage

Reuse mode:

```bash
.venv/bin/python scripts/run_continuous_search_cycle.py \
  --registry-root candidate_registry \
  --cycle-name reuse_cycle_20250508 \
  --mode reuse \
  --tag walkforward \
  --tag probe \
  --tag fused32 \
  --selection-start-utc '2025-05-08 00:00:00' \
  --selection-end-utc '2025-05-15 00:00:00'
```

Generate-mode planning:

```bash
.venv/bin/python scripts/run_continuous_search_cycle.py \
  --registry-root candidate_registry \
  --cycle-name dryrun_generate_smoke \
  --mode generate \
  --window-start '2025-05-01 00:00:00' \
  --selection-start-utc '2025-05-08 00:00:00' \
  --selection-end-utc '2025-05-15 00:00:00' \
  --generator-window-mode selection_anchored \
  --generator-window-offset-days 0 \
  --generator-window-offset-days -7 \
  --runs-per-window 2 \
  --generations 1 \
  --population-size 4 \
  --dry-run
```

## Limits

Still pending:

1. persistent daemon mode
2. higher-throughput candidate generation beyond the current probe launcher
3. lifecycle actions after tradeforward
4. scheduler/UI integration
5. long-running daemonized orchestration over many rolling simulations

Follow-up roadmap:

1. [next-development-roadmap.md](./next-development-roadmap.md)
2. [multi-market-architecture.md](./multi-market-architecture.md)
