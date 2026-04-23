# Allocator Workbench

Last updated: 2026-03-10

## Purpose

The allocator workbench is the first portfolio sizing layer above shortlist.

It answers:

1. how much capital should be put under the control of the selected systems
2. how that capital should be split between them
3. how the portfolio behaves under common-window bootstrap scenarios

This is also where the global risk dial lives:

1. the total fraction of account equity placed at the disposal of all systems

## Current Implementation

Core module:

1. [allocator.py](/home/user/mcs/umc_nn/candidates/allocator.py)

CLI:

1. [run_allocator_workbench.py](/home/user/mcs/scripts/run_allocator_workbench.py)

Registry artifact directory:

1. [candidate_registry/allocations](/home/user/mcs/candidate_registry/allocations)

## Current Resampling Model

This layer uses common-window block bootstrap over synchronized step returns:

1. shortlisted candidates are re-evaluated on the same shared window
2. per-step returns are aligned on one time axis
3. bootstrap samples contiguous blocks of that synchronized matrix
4. all candidates in the sampled block move together exactly as they did in
   history

Why this matters:

1. simultaneous loss episodes are preserved inside blocks
2. overlap between systems is preserved in the resampled scenarios
3. allocator decisions are based on portfolio behavior, not only single-system
   summaries

This is not yet the full legacy trade-overlap replay engine.
It is the first portfolio-safe approximation suitable for allocator work.

## Current Sizing Logic

For each shortlisted candidate:

1. take the selected shortlist score
2. convert scores to relative shares
3. apply a per-system capital cap
4. apply an optional per-cluster cap
5. apply operator pin/exclude/force/max-cap overrides
6. scale by the requested portfolio risk fraction

Terminology:

1. `requested_risk_fraction` = the operator's global risk dial
2. `allocated_risk_fraction` = what was actually allocated after per-system caps
3. `reserve_fraction` = capital left idle
4. `cap_reason` = why an individual weight was clipped

## Current Objective

Each risk-dial scenario is scored by:

1. pessimistic net profit efficiency
2. median net profit
3. penalty for exceeding the configured drawdown budget
4. ruin penalty
5. loss-rate penalty

This is not intended as the final allocator.
It is the first operator-facing sizing workbench.

## Current CLI Usage

Example:

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
  --risk-fraction 0.75 \
  --per-system-cap 0.50
```

## Current Real Observation

Current real report:

1. [allocator_20250508_20250515_clustered.json](/home/user/mcs/candidate_registry/allocations/allocator_20250508_20250515_clustered.json)
2. source cluster report:
   [clusters_20250508_20250515.json](/home/user/mcs/candidate_registry/clusters/clusters_20250508_20250515.json)
3. source override set:
   [ops_20250508.json](/home/user/mcs/candidate_registry/overrides/ops_20250508.json)

On the current two-candidate shortlist:

1. all tested risk scenarios were negative on the shared window
2. the objective therefore chose the lowest risk dial, `0.25`
3. higher requested risk increased both pessimistic loss and drawdown almost
   monotonically
4. the pinned OOS-positive candidate remained in scope under the override set
5. the configured cluster cap did not bind on this two-cluster shortlist, so
   no reserve capital was created by concentration controls yet

Practical meaning:

1. shortlist alone is not enough
2. allocator workbench already adds a useful veto layer
3. the global risk dial is behaving as intended
4. cluster/override plumbing is now on the same metric path as sizing

## What This Enables Next

This workbench now makes it possible to add:

1. promotion from reviewed subset into approved portfolio state
2. turnover-aware rebalance proposals
3. multi-subset comparison via [combination-search.md](/home/user/mcs/docs/combination-search.md)
4. later `optimal f` and similar fraction scans on portfolio subsets
