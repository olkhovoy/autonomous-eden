# Candidate Registry

Last updated: 2026-03-10

## Purpose

The candidate registry is the first durable contract between:

1. candidate generation
2. candidate evaluation
3. manual operator review
4. future automatic selection
5. future UI

It exists so the project can stop passing candidate state around as scattered
checkpoints, logs, and ad hoc script outputs.

## Current Implementation

Core modules:

1. [schema.py](/home/user/mcs/umc_nn/candidates/schema.py)
2. [registry.py](/home/user/mcs/umc_nn/candidates/registry.py)
3. [selection.py](/home/user/mcs/umc_nn/candidates/selection.py)

Current registry root:

1. [candidate_registry](/home/user/mcs/candidate_registry)

## Core Records

### Candidate Record

Each candidate stores:

1. stable `candidate_id`
2. display name
3. engine name and role
4. status
5. tags
6. experiment manifest
7. period stats
8. baseline stats
9. selection flags
10. optional notes and metadata

### Experiment Manifest

This captures:

1. source script
2. representation name
3. data path
4. checkpoint path
5. log path
6. train and OOS windows
7. search config
8. economics config

### Period Stats

Current period-level contract includes:

1. steps
2. final balance
3. PnL
4. drawdown
5. trades
6. win rate
7. action counts
8. baseline comparison flags

### Rule Set

Rules are saved as JSON and can be shared between:

1. CLI tools
2. future UI filters
3. future automatic selection passes

### Diversification And Shortlist Reports

The registry also stores cross-candidate artifacts:

1. diversification reports on explicit common windows
2. common-window normalized curve snapshots for monitoring UI
3. cluster reports built from downside/co-crash similarity
4. operator override sets with audit trail
3. shortlist reports with explainable score components
4. selected subset pair compatibility breakdowns
5. allocator workbench reports with scenario sizing and common-window bootstrap
6. exhaustive combination search reports over shortlist pools
7. dashboard snapshot feeds for UI consumption
8. continuous cycle reports
9. tradeforward handoff plans
10. tradeforward evaluation reports
11. rolling conveyor reports
12. lifecycle state-machine reports
13. portfolio ledger and rebalance reports
14. portfolio baseline and gate reports
15. candidate-farm batch reports

## Current CLI

Import walk-forward probe results:

```bash
.venv/bin/python scripts/register_walkforward_candidates.py \
  --summary-path checkpoints/monolith_walkforward_probe/summary.json \
  --registry-root candidate_registry \
  --tag walkforward --tag probe --tag fused32
```

List candidates:

```bash
.venv/bin/python scripts/list_candidates.py --registry-root candidate_registry
```

Apply rules:

```bash
.venv/bin/python scripts/apply_candidate_rules.py \
  --registry-root candidate_registry \
  --rule selection_flags.oos_positive:eq:true \
  --rule periods.oos.trades:gte:10
```

Update status manually:

```bash
.venv/bin/python scripts/update_candidate_status.py \
  --registry-root candidate_registry \
  --candidate-id cand_xxx \
  --status approved \
  --note "manual review"
```

Export trades:

```bash
.venv/bin/python scripts/export_candidate_trades.py --registry-root candidate_registry
```

Run bootstrap resampling:

```bash
.venv/bin/python scripts/run_candidate_resampling.py \
  --registry-root candidate_registry \
  --period train \
  --iterations 500 \
  --fraction 0.5 \
  --fraction 1.0
```

Run common-window diversification:

```bash
.venv/bin/python scripts/run_candidate_diversification.py \
  --registry-root candidate_registry \
  --start-utc '2025-05-08 00:00:00' \
  --end-utc '2025-05-15 00:00:00' \
  --report-name common_20250508_20250515_v2
```

Build candidate clusters:

```bash
.venv/bin/python scripts/run_candidate_clustering.py \
  --registry-root candidate_registry \
  --diversification-report common_20250508_20250515_v2 \
  --report-name clusters_20250508_20250515 \
  --similarity-threshold 0.40
```

Build shortlist:

```bash
.venv/bin/python scripts/run_candidate_shortlist.py \
  --registry-root candidate_registry \
  --diversification-report common_20250508_20250515_v2 \
  --report-name shortlist_20250508_20250515_f1_00 \
  --resampling-name train_bootstrap_f1.00 \
  --max-candidates 3
```

Update operator overrides:

```bash
.venv/bin/python scripts/update_operator_overrides.py \
  --registry-root candidate_registry \
  --override-name ops_20250508 \
  --actor operator \
  --source-cluster-report clusters_20250508_20250515 \
  --candidate-id cand_d8959d02a5e6 \
  --pin true \
  --note "keep oos-positive candidate in reviewed subsets"
```

Build allocator workbench:

```bash
.venv/bin/python scripts/run_allocator_workbench.py \
  --registry-root candidate_registry \
  --shortlist-report shortlist_20250508_20250515_f1_00 \
  --report-name allocator_20250508_20250515_clustered \
  --cluster-report clusters_20250508_20250515 \
  --override-set ops_20250508 \
  --default-cluster-cap 0.50 \
  --risk-fraction 0.25 \
  --risk-fraction 0.50 \
  --risk-fraction 0.75
```

Run combination search:

```bash
.venv/bin/python scripts/run_shortlist_combinations.py \
  --registry-root candidate_registry \
  --shortlist-report shortlist_20250508_20250515_f1_00 \
  --report-name combinations_20250508_20250515_clustered \
  --cluster-report clusters_20250508_20250515 \
  --override-set ops_20250508 \
  --default-cluster-cap 0.50 \
  --max-pool-size 4 \
  --min-subset-size 1 \
  --max-subset-size 3
```

Build dashboard feed:

```bash
.venv/bin/python scripts/build_operator_dashboard_feed.py \
  --registry-root candidate_registry \
  --feed-name dashboard_20250508_20250515 \
  --shortlist-report shortlist_20250508_20250515_f1_00 \
  --diversification-report common_20250508_20250515_v2 \
  --cluster-report clusters_20250508_20250515 \
  --override-set ops_20250508 \
  --allocator-report allocator_20250508_20250515_clustered \
  --combination-report combinations_20250508_20250515_clustered
```

Run one continuous cycle on an existing candidate pool:

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

Run a rolling historical conveyor simulation:

```bash
.venv/bin/python scripts/run_rolling_conveyor_simulator.py \
  --registry-root candidate_registry \
  --report-name reuse_rolling_20250501 \
  --mode reuse \
  --selection-start-utc '2025-05-01 00:00:00' \
  --selection-days 7 \
  --forward-days 7 \
  --num-cycles 2 \
  --status research \
  --tag walkforward \
  --tag probe \
  --tag fused32
```

Run the lifecycle supervisor on a rolling report:

```bash
.venv/bin/python scripts/run_lifecycle_state_machine.py \
  --registry-root candidate_registry \
  --rolling-report reuse_rolling_20250501 \
  --report-name reuse_rolling_20250501_lifecycle
```

Build a portfolio ledger on top of rolling and lifecycle artifacts:

```bash
.venv/bin/python scripts/run_portfolio_ledger.py \
  --registry-root candidate_registry \
  --rolling-report reuse_rolling_20250501 \
  --lifecycle-report reuse_rolling_20250501_lifecycle \
  --report-name reuse_rolling_20250501_portfolio
```

Build portfolio-level baselines and gates:

```bash
.venv/bin/python scripts/run_portfolio_baselines.py \
  --registry-root candidate_registry \
  --portfolio-ledger-report reuse_rolling_20250501_portfolio \
  --report-name reuse_rolling_20250501_baselines
```

Run a manifest-driven candidate farm:

```bash
.venv/bin/python scripts/run_candidate_farm.py \
  --manifest-path /tmp/candidate_farm_manifest.json \
  --resume-completed
```

## How The Current Candidate Engine Plugs In

The current single-policy engine is still producing:

1. checkpoint `.npy`
2. log file
3. period metrics from walk-forward evaluation

The registry layer converts that into one canonical candidate record.

This means the current engine can already be used as:

1. one candidate source

without pretending that it is:

1. the final strategy architecture

## UI Direction

The future UI should read directly from the registry and expose:

1. candidate list
2. saved filters
3. period metrics
4. baseline comparisons
5. status transitions
6. operator notes
7. common-window curve fans
8. shortlist score explanations
9. highlighted exceptions near rule boundaries
10. allocator scenarios and global risk-dial states
11. subset-comparison views across the shortlist pool
12. cluster labels and concentration warnings
13. operator overrides and audit history
14. one snapshot feed per dashboard state without UI-side joins
15. cycle-level and tradeforward-level operational artifacts
16. rolling conveyor history and cycle-by-cycle portfolio outcomes
17. lifecycle transitions and current service-state summaries
18. rebalance turnover and portfolio service history
19. portfolio-level baseline comparisons and gate verdicts
20. farm-level batch comparisons and gate-pass summaries

The UI should not invent a separate candidate state model.

## Immediate Next Extensions

The next useful additions on top of this contract are:

1. operator dashboard / monitoring UI
2. live lifecycle state and disable audit trail
3. continuous replacement-loop orchestration
4. larger-pool smart portfolio search
5. allocation promotion workflow from research to paper/active
6. dashboard views for farm batches and search stagnation
