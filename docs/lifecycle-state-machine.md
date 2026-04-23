# Lifecycle State Machine

Last updated: 2026-03-10

## Purpose

The lifecycle state machine is the first explicit supervisor layer above the
rolling conveyor.

It decides how candidates move through simulated service states:

1. `research`
2. `approved`
3. `paper`
4. `active`
5. `draining`
6. `retired`

The current rebuild keeps this layer transparent and operator-friendly:
the state machine emits an audit artifact first, and only applies status
updates back into the registry when explicitly requested.

## Current Implementation

Core module:

1. [lifecycle.py](/home/user/mcs/umc_nn/candidates/lifecycle.py)

CLI:

1. [run_lifecycle_state_machine.py](/home/user/mcs/scripts/run_lifecycle_state_machine.py)

Registry artifact directory:

1. [candidate_registry/lifecycle](/home/user/mcs/candidate_registry/lifecycle)

## Current Real Smoke Run

Current report:

1. [reuse_rolling_20250501_lifecycle.json](/home/user/mcs/candidate_registry/lifecycle/reuse_rolling_20250501_lifecycle.json)

Observed result:

1. final status counts: `research=4`, `approved=1`, `paper=1`
2. `wf_01_20240101_run01` moved `research -> paper` after `2` successful forward cycles
3. `wf_03_20250501_run02` moved `research -> approved` after `1` successful forward cycle

## Default Decision Rules

The current supervisor is intentionally simple and conservative:

1. selected candidate with positive forward PnL, enough trades, and acceptable DD is a success
2. `research -> approved` on first selected success
3. `approved -> paper` on next selected success
4. `paper -> active` after configured consecutive selected successes
5. `paper/active -> draining` on failed selected cycle or service idle threshold
6. `draining -> retired` on repeated failure or repeated idle omission
7. `draining -> paper` on configured recovery success

The thresholds are CLI-configurable.

## Usage

```bash
.venv/bin/python scripts/run_lifecycle_state_machine.py \
  --registry-root candidate_registry \
  --rolling-report reuse_rolling_20250501 \
  --report-name reuse_rolling_20250501_lifecycle
```

To also apply the resulting statuses back into the registry:

```bash
.venv/bin/python scripts/run_lifecycle_state_machine.py \
  --registry-root candidate_registry \
  --rolling-report reuse_rolling_20250501 \
  --report-name reuse_rolling_20250501_lifecycle_applied \
  --apply-status-updates
```

## Next Step

The immediate next development target is:

1. portfolio-level baselines and gates on top of lifecycle-aware portfolio history

The current portfolio ledger layer already consumes this lifecycle output:

1. [reuse_rolling_20250501_portfolio.json](/home/user/mcs/candidate_registry/portfolio_ledger/reuse_rolling_20250501_portfolio.json)
