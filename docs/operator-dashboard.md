# Operator Dashboard

Last updated: 2026-03-10

## Purpose

The first dashboard is not a passive report viewer.
It is the main operator workbench for a long-running search process.

Assumption carried from legacy practice:

1. search is continuous
2. meaningful decisions remain operator-supervised until the conveyor proves
   itself in production
3. the UI should keep the operator useful, not idle

## Core Product Goal

The dashboard should make it easy to:

1. monitor the flow of new candidates in real time
2. inspect only the candidates that are near the current ideal
3. catch search stagnation early
4. notice interesting exceptions that strict rules would miss
5. approve, reject, paper, or promote candidates with an audit trail

## Required Primary Views

### 1. Real-Time Filtered Candidate List

Must support:

1. saved filter presets
2. sort by any score component or rule
3. live updates as new candidates arrive
4. direct status transitions
5. operator notes

### 2. Candidate Detail View

Must show:

1. train / valid / OOS / total stats
2. baselines
3. trade list
4. resampling summaries
5. pairwise diversification links
6. shortlist score components
7. exception flags

### 3. Search Monitoring View

Must surface:

1. candidate generation rate
2. best-so-far improvement over time
3. time since last shortlist-quality candidate
4. distribution drift of candidate quality
5. diversity collapse in the candidate pool

This is where the operator notices stagnation.

### 4. Broom View

This should be a signature visualization.

Definition:

1. plot normalized equity curves from one shared origin on the same time axis
2. use the current filter/sort result as the visible population
3. modulate brightness by the current ranking or rule score
4. keep the full fan visible so clusters, outliers, and regime split become
   obvious at a glance

Recommended encoding:

1. x-axis: common-window step or timestamp
2. y-axis: normalized balance multiple
3. opacity or brightness: shortlist score or chosen sort key
4. color: status, representation, engine family, or cluster label

Data source:

1. diversification report curve snapshots
2. shortlist report brightness hints
3. registry candidate metadata

## Exception Surfacing

The UI should not only show rule-passing systems.
It should also show "interesting exceptions".

First useful exception categories:

1. strong OOS result but fragile train bootstrap
2. strong diversifier despite mediocre standalone score
3. near-miss quality candidate just below one hard threshold
4. candidate with unusual curve shape versus its cluster

The operator should be able to pin these candidates for later review.

## Interaction Principles

1. the UI should never force the operator into one automatic score
2. every shortlist decision should be explainable by stored components
3. every manual override should be saved
4. monitoring should prioritize pattern recognition over dense tables alone

## What The Current Data Layer Already Supports

Already available:

1. candidate registry
2. saved rules
3. trade artifacts
4. resampling artifacts
5. common-window diversification reports
6. broom-ready normalized curve snapshots
7. explainable shortlist reports
8. cluster labels and cluster summaries
9. operator overrides with audit trail
10. allocator and combination reports with shared objective metrics

Still missing for the full dashboard:

1. live search event stream
2. search stagnation metrics
3. search-progress notifications and anomaly surfacing

## Current Backend Feed

The first UI should not read half a dozen registry artifacts directly.
It should read one dashboard snapshot feed built from them.

Current builder:

1. [dashboard.py](/home/user/mcs/umc_nn/candidates/dashboard.py)
2. [build_operator_dashboard_feed.py](/home/user/mcs/scripts/build_operator_dashboard_feed.py)
3. stack options:
   [ui-stack-options.md](/home/user/mcs/docs/ui-stack-options.md)

Current real feed:

1. [dashboard_20250508_20250515.json](/home/user/mcs/candidate_registry/dashboard/dashboard_20250508_20250515.json)
2. phase-1 UI screenshot:
   [operator_ui_dashboard.png](/home/user/mcs/checkpoints/operator_ui_dashboard.png)

This feed already contains:

1. candidate list rows flattened for table rendering
2. broom lines with brightness and cluster labels
3. cluster summaries
4. operator overrides and recent audit entries
5. allocator panel payload
6. combination panel payload
7. basic monitoring payload such as recent candidates and interesting exceptions

## Phase 1 UI

Current frontend:

1. [operator_ui](/home/user/mcs/operator_ui)
2. stack note:
   [ui-stack-options.md](/home/user/mcs/docs/ui-stack-options.md)
3. launch instructions:
   [README.md](/home/user/mcs/operator_ui/README.md)

Current phase-1 screens:

1. candidate table
2. broom view
3. candidate detail
4. allocator panel
5. combination panel
6. cluster and override panels

## Farm View

The same UI now also has a farm-oriented view.
This avoids building a separate throwaway frontend for long-running search.

Current farm backend:

1. [farm_dashboard.py](/home/user/mcs/umc_nn/candidates/farm_dashboard.py)
2. [build_farm_dashboard_feed.py](/home/user/mcs/scripts/build_farm_dashboard_feed.py)
3. real feed:
   [reuse_farm_smoke_light_202505_feed.json](/home/user/mcs/candidate_registry/farm_dashboard/reuse_farm_smoke_light_202505_feed.json)

Current farm UI panels:

1. scenario summary strip
2. farm broom view across scenarios
3. scenario table with gate and progress state
4. scenario detail panel
5. farm monitoring panel

Current live-monitoring behavior:

1. farm view polls its feed every `5s` by default
2. query parameter `?view=farm&refresh=2` changes polling to `2s`
3. the monitoring panel now shows heartbeat, throughput, stagnation, and recent events
4. `run_candidate_farm.py --dashboard-sync-path ...` can update the served JSON in place
5. `run_candidate_farm.py --heartbeat-interval-seconds ...` emits in-stage heartbeats during long-running steps

This is the first place where the operator can see a "broom" for the
candidate farm itself, not only for one shortlist window.
