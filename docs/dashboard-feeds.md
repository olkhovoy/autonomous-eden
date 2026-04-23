# Dashboard Feeds

Last updated: 2026-03-10

## Purpose

The operator UI should not join half the registry client-side.
It should consume one read-optimized snapshot feed assembled on the backend.

This keeps:

1. UI code thin
2. research logic centralized
3. operator screens reproducible from named artifacts

## Current Implementation

Core module:

1. [dashboard.py](/home/user/mcs/umc_nn/candidates/dashboard.py)

CLI:

1. [build_operator_dashboard_feed.py](/home/user/mcs/scripts/build_operator_dashboard_feed.py)

Registry artifact directory:

1. [candidate_registry/dashboard](/home/user/mcs/candidate_registry/dashboard)

Farm counterpart:

1. [farm_dashboard.py](/home/user/mcs/umc_nn/candidates/farm_dashboard.py)
2. [build_farm_dashboard_feed.py](/home/user/mcs/scripts/build_farm_dashboard_feed.py)
3. [candidate_registry/farm_dashboard](/home/user/mcs/candidate_registry/farm_dashboard)

## Feed Contents

The current feed includes:

1. flattened candidate rows for the main grid
2. monitoring summary counts
3. recent candidate timeline
4. interesting exceptions list
5. broom-ready curve payload
6. cluster summaries
7. operator overrides and recent audit entries
8. allocator panel payload
9. combination panel payload

## Current Real Feed

Current snapshot:

1. [dashboard_20250508_20250515.json](/home/user/mcs/candidate_registry/dashboard/dashboard_20250508_20250515.json)

Current farm snapshot:

1. [reuse_farm_smoke_light_202505_feed.json](/home/user/mcs/candidate_registry/farm_dashboard/reuse_farm_smoke_light_202505_feed.json)

Current observed summary:

1. `6` total candidates
2. `2` selected shortlist candidates
3. `3` exception-tagged candidates
4. `1` pinned candidate
5. `4` clusters

## Current Usage

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

## Limits

Still pending:

1. live event stream instead of snapshot-only monitoring
2. explicit stagnation metrics
3. server push or polling contract
4. UI-side filtering presets persisted back to registry

## Farm Feed Notes

The farm feed is intentionally separate from the candidate feed.
It optimizes for:

1. scenario-level progress and gate monitoring
2. cross-scenario broom rendering
3. long-run throughput and stagnation diagnostics
4. later live-refresh without changing the UI contract

Farm monitoring payload now also includes:

1. heartbeat state and seconds since last event
2. throughput metrics such as events/hour and completions/hour
3. stagnation metrics such as time since last completion and last gate pass
4. recent event trail for scenario-stage transitions
