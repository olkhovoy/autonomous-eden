# Candidate-Farm Expansion

Last updated: 2026-03-10

## Purpose

The conveyor now has historical portfolio gates.
That means broader candidate generation should no longer be judged by isolated
candidate PnL, but by full downstream conveyor outcomes:

1. rolling selection and tradeforward
2. lifecycle state transitions
3. portfolio ledger and rebalance behavior
4. portfolio-level baselines and gates

The first practical farm layer therefore sits above `rolling`, not below it.

## Current Implementation

Core builder:

1. [farm.py](/home/user/mcs/umc_nn/candidates/farm.py)

Launcher:

1. [run_candidate_farm.py](/home/user/mcs/scripts/run_candidate_farm.py)

Registry artifact directory:

1. [candidate_registry/farms](/home/user/mcs/candidate_registry/farms)

Current smoke artifact:

1. [reuse_farm_smoke_light_202505.json](/home/user/mcs/candidate_registry/farms/reuse_farm_smoke_light_202505.json)
2. farm dashboard feed:
   [reuse_farm_smoke_light_202505_feed.json](/home/user/mcs/candidate_registry/farm_dashboard/reuse_farm_smoke_light_202505_feed.json)

## Manifest Model

The farm runner is manifest-driven.
Each scenario specifies:

1. `rolling_args`
2. `forwarded_cycle_args`
3. optional `lifecycle_args`
4. optional `ledger_args`
5. optional `baseline_args`

This keeps the farm configuration outside chat state and lets the same runner
scale from lightweight smoke runs to longer historical batches.

The manifest now supports:

1. `defaults`
2. explicit `scenarios`
3. `scenario_templates`
4. linked `cases`
5. cartesian `matrix` sweeps

`cases` are for values that should move together, for example
`selection_label + selection_start_utc`.
`matrix` is for true search grids such as
`generations x population_size x train_days`.

List values from `defaults` and scenario/template blocks are appended.
For CLI-like arrays such as `forwarded_cycle_args`, that means later flags win,
which is useful for template sweeps that override a default budget.

## Scenario Output

For each scenario the runner captures:

1. rolling report
2. lifecycle report
3. portfolio ledger report
4. portfolio baselines report
5. gate verdict
6. candidate-pool and selected-candidate coverage
7. per-scenario logs and output directory

During execution the runner also writes progress snapshots.
Each scenario now exposes:

1. `status`: `planned`, `running`, `completed`, `reused`, or `failed`
2. `progress_stage`: `queued`, `rolling`, `lifecycle`, `portfolio_ledger`,
   `portfolio_baselines`, `completed`, or `failed`
3. `updated_at_utc`

That snapshot contract is the basis for later live farm monitoring.

The runner now also writes an append-only farm progress log:

1. [candidate_registry/farm_progress](/home/user/mcs/candidate_registry/farm_progress)
2. example:
   [reuse_farm_smoke_light_202505.json](/home/user/mcs/candidate_registry/farm_progress/reuse_farm_smoke_light_202505.json)

This keeps stage transitions and final scenario outcomes outside chat state and
lets the UI compute heartbeat and stagnation metrics without parsing stdout.

The farm report then summarizes:

1. completed vs failed scenarios
2. gate pass count and rate
3. unique candidate-pool coverage
4. unique selected-candidate coverage
5. best scenario by PnL
6. best gate-passing scenario by PnL
7. lowest drawdown scenario
8. ranked scenario list

## Real Smoke Run

Current lightweight smoke farm:

1. [reuse_farm_smoke_light_202505.json](/home/user/mcs/candidate_registry/farms/reuse_farm_smoke_light_202505.json)
2. [reuse_farm_smoke_light_202505__reuse_may01_light__rolling.json](/home/user/mcs/candidate_registry/rolling/reuse_farm_smoke_light_202505__reuse_may01_light__rolling.json)
3. [reuse_farm_smoke_light_202505__reuse_may01_light__lifecycle.json](/home/user/mcs/candidate_registry/lifecycle/reuse_farm_smoke_light_202505__reuse_may01_light__lifecycle.json)
4. [reuse_farm_smoke_light_202505__reuse_may01_light__portfolio.json](/home/user/mcs/candidate_registry/portfolio_ledger/reuse_farm_smoke_light_202505__reuse_may01_light__portfolio.json)
5. [reuse_farm_smoke_light_202505__reuse_may01_light__baselines.json](/home/user/mcs/candidate_registry/portfolio_baselines/reuse_farm_smoke_light_202505__reuse_may01_light__baselines.json)

Summary:

1. `2` scenarios completed
2. `1/2` passed the baseline gate
3. best gate-passing scenario: `reuse_may01_light`
4. best scenario PnL: about `+232.00`
5. second scenario stayed profitable at about `+196.95`, but failed the gate
6. total unique candidate pool coverage across the farm: `6`
7. unique selected candidates across the farm: `2`

## Dry-Run Template Smoke

Template-driven dry-run artifact:

1. [generate_template_dryrun_202505.json](/home/user/mcs/candidate_registry/farms/generate_template_dryrun_202505.json)

What it demonstrates:

1. one template expanded into `4` planned scenarios
2. `cases` linked `selection_label` with `selection_start_utc`
3. `matrix` swept `generations = 1, 2`
4. all planned scenarios shared the same default rolling/generator envelope

## Resume Mode

Long farms can now be resumed with:

```bash
.venv/bin/python scripts/run_candidate_farm.py \
  --manifest-path /tmp/candidate_farm_manifest.json \
  --resume-completed
```

If a scenario already has its full downstream chain:

`rolling -> lifecycle -> portfolio -> baselines`

the farm runner reuses those artifacts instead of rerunning the scenario.

## Live Feed Mode

The runner can now also refresh the farm dashboard feed while the farm is
running:

```bash
.venv/bin/python scripts/run_candidate_farm.py \
  --manifest-path /tmp/candidate_farm_manifest.json \
  --dashboard-sync-path operator_ui/public/data/farm-dashboard-feed.json
```

Relevant flags:

1. `--dashboard-feed-name`
2. `--dashboard-sync-path`
3. `--dashboard-max-scenarios`
4. `--dashboard-max-broom-lines`
5. `--heartbeat-interval-seconds`

This is the bridge from a long backend batch to a live-refreshing farm UI.

`--heartbeat-interval-seconds` is especially important for long `rolling`
stages. The runner now emits periodic `heartbeat` events while a scenario is
still inside the same stage, so the farm UI can distinguish:

1. a live long-running step
2. a stale or hung process

## Why This Layer Matters

Before this layer, the project could run one rolling conveyor at a time.
Now it can compare families of rolling scenarios under one stable contract.

That is the first usable version of `candidate-farm expansion`:

`farm manifest -> rolling scenarios -> lifecycle -> ledger -> baselines -> gate summary`

## Next Follow-Up

Immediate follow-ups after this layer:

1. push real generator throughput higher under these same gates
2. surface farm reports in the operator dashboard
3. add stagnation and throughput diagnostics for long-running search batches
