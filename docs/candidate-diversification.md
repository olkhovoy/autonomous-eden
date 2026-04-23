# Candidate Diversification

Last updated: 2026-03-10

## Purpose

Diversification metrics answer a portfolio-level question:

1. which candidate pairs are less likely to fail together on the same window

This is not a single-candidate property.
So diversification is stored as a separate registry report over a common
evaluation window.

## Current Implementation

Core module:

1. [diversification.py](/home/user/mcs/umc_nn/candidates/diversification.py)

CLI:

1. [run_candidate_diversification.py](/home/user/mcs/scripts/run_candidate_diversification.py)

Registry artifact directory:

1. [candidate_registry/diversification](/home/user/mcs/candidate_registry/diversification)

## Why Common Windows Matter

Pairwise diversification must be evaluated on one explicit shared slice.

It is invalid to compare:

1. one candidate on its own train window
2. another candidate on a different train window

because that does not measure simultaneous failure risk.

The current implementation therefore evaluates all selected candidates on one
shared UTC window.

## Current Metrics

Each pair now stores:

1. return correlation
2. downside correlation
3. simultaneous loss rate
4. simultaneous drawdown rate
5. action agreement rate
6. same-side non-flat overlap
7. opposite-side non-flat overlap
8. equal-weight combined final balance
9. equal-weight combined net profit
10. equal-weight combined max drawdown
11. average individual max drawdown
12. drawdown improvement in percentage points

Each report now also stores downsampled common-window candidate curves:

1. normalized balance history
2. sample indices on the shared step axis
3. final balance
4. max drawdown

This makes the diversification report directly usable by a future monitoring UI.

The most actionable fields for manual review are usually:

1. downside correlation
2. simultaneous loss rate
3. equal-weight max drawdown
4. drawdown improvement

## Current CLI Usage

Example:

```bash
.venv/bin/python scripts/run_candidate_diversification.py \
  --registry-root candidate_registry \
  --start-utc '2025-05-08 00:00:00' \
  --end-utc '2025-05-15 00:00:00' \
  --report-name common_20250508_20250515
```

## Current Observation On The Probe Pool

Current report:

1. [common_20250508_20250515.json](/home/user/mcs/candidate_registry/diversification/common_20250508_20250515.json)

On that shared week:

1. some pairs show negative or near-zero return correlation
2. some pairs still have positive downside correlation despite weak headline
   correlation
3. several pairs improve equal-weight drawdown materially
4. the best pair by drawdown improvement is not automatically the best pair for
   profit or robustness

Practical meaning:

1. raw candidate quality is not enough
2. pair compatibility is already measurable
3. future allocation should use both:
   candidate robustness,
   and pairwise diversification
4. the same report can now feed a "broom" visualization of normalized curves
   on one shared window

## Immediate Use In Operator Workflow

This layer already supports a manual review loop:

1. shortlist candidates by OOS and resampling
2. run a common-window diversification report on the shortlist
3. reject highly correlated or co-crashing pairs
4. prefer combinations that reduce equal-weight drawdown
5. inspect common-window normalized curves instead of relying on pair tables alone

## Current Limitation

This is still pairwise only.

It does not yet solve:

1. portfolio subset search
2. cluster detection
3. overlap-aware multi-system bootstrap
4. capital allocation

Those are the next layers above this one.
