# Operator Overrides

Last updated: 2026-03-10

## Purpose

The first modern conveyor should remain operator-assist.
Override sets are the explicit contract for that human-in-the-loop layer.

They let the operator:

1. pin a candidate into reviewed subsets
2. force-include or exclude a candidate
3. cap a single candidate's capital fraction
4. cap a whole cluster's capital fraction
5. leave an auditable reason for each decision

## Current Implementation

Core module:

1. [overrides.py](/home/user/mcs/umc_nn/candidates/overrides.py)

CLI:

1. [update_operator_overrides.py](/home/user/mcs/scripts/update_operator_overrides.py)
2. [show_operator_overrides.py](/home/user/mcs/scripts/show_operator_overrides.py)

Registry artifact directory:

1. [candidate_registry/overrides](/home/user/mcs/candidate_registry/overrides)

## Current Real Override Set

Current override set:

1. [ops_20250508.json](/home/user/mcs/candidate_registry/overrides/ops_20250508.json)

Current real actions inside it:

1. pin `wf_03_20250501_run02` for reviewed subsets
2. cap `cluster_002` at `0.50` gross capital
3. preserve both actions in the audit log with actor and note

## Current Usage

Pin a candidate:

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

Cap a cluster:

```bash
.venv/bin/python scripts/update_operator_overrides.py \
  --registry-root candidate_registry \
  --override-name ops_20250508 \
  --actor operator \
  --cluster-id cluster_002 \
  --max-cap 0.50 \
  --note "cap dense co-crash cluster at 50%"
```

Show the full override set:

```bash
.venv/bin/python scripts/show_operator_overrides.py \
  --registry-root candidate_registry \
  --override-name ops_20250508
```

## Current Behavior

The same override set now flows through:

1. allocator workbench
2. exhaustive combination search
3. future operator dashboard

This matters because manual judgment is no longer outside the reproducible
research path.

## Limits

Still pending:

1. UI editing of overrides
2. approval/sign-off workflow around proposed allocations
3. live-state aware overrides such as `draining` and `retired`
4. richer override scopes such as engine family or representation family
