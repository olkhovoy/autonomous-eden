# Cluster Controls

Last updated: 2026-03-10

## Purpose

Cluster controls exist to stop the allocator from silently concentrating capital
into systems that look different in rank order but fail together.

This layer is the first durable approximation of the legacy operator workflow
where similarity and concentration were judged visually and enforced manually.

## Current Implementation

Core module:

1. [clustering.py](/home/user/mcs/umc_nn/candidates/clustering.py)

CLI:

1. [run_candidate_clustering.py](/home/user/mcs/scripts/run_candidate_clustering.py)

Registry artifact directory:

1. [candidate_registry/clusters](/home/user/mcs/candidate_registry/clusters)

## Current Similarity Signal

Cluster edges are built from a weighted similarity score over pair metrics from
the common-window diversification report:

1. downside correlation
2. simultaneous loss rate
3. positive return correlation
4. action agreement rate

Candidates above the configured threshold are connected into one cluster via
connected components.

## Current Real Report

Current report:

1. [clusters_20250508_20250515.json](/home/user/mcs/candidate_registry/clusters/clusters_20250508_20250515.json)

Observed structure on the current pool:

1. `cluster_002` contains three tightly related systems:
   `wf_01_20240101_run01`, `wf_03_20250501_run01`, `wf_03_20250501_run02`
2. the remaining candidates are singleton clusters
3. this is already enough to express concentration rules at portfolio level

## Current Usage

Example:

```bash
.venv/bin/python scripts/run_candidate_clustering.py \
  --registry-root candidate_registry \
  --diversification-report common_20250508_20250515_v2 \
  --report-name clusters_20250508_20250515 \
  --similarity-threshold 0.40
```

The resulting cluster ids can be consumed by:

1. allocator workbench
2. combination search
3. future dashboard views

## Limits

This is not yet the final concentration model.

Still pending:

1. hierarchical clustering or graph community alternatives
2. time-varying cluster stability
3. operator-visible cluster explanation panels
4. cluster-level live exposure monitoring
