# Combination Search

Last updated: 2026-03-10

## Purpose

Combination search is the first layer that explicitly compares portfolio
subsets against each other, not only individual candidates.

It answers:

1. which subset of the shortlist pool behaves best under the current objective
2. whether the currently selected shortlist is actually the best local subset
3. which near-miss candidates improve the portfolio despite weaker standalone
   scores

## Current Implementation

Core module:

1. [portfolio_search.py](/home/user/mcs/umc_nn/candidates/portfolio_search.py)

CLI:

1. [run_shortlist_combinations.py](/home/user/mcs/scripts/run_shortlist_combinations.py)

Registry artifact directory:

1. [candidate_registry/combinations](/home/user/mcs/candidate_registry/combinations)

## Search Method

The current implementation is intentionally exhaustive on a bounded pool:

1. build a pool from the shortlist report
2. always keep selected candidates in the pool
3. optionally keep exception-flag candidates in the pool
4. fill the remaining pool slots by shortlist score
5. evaluate every subset within the configured size range
6. for each subset, evaluate every configured risk-dial scenario
7. rank all subset scenarios by the allocator objective

This is the correct baseline because:

1. it is transparent
2. it is easy to test
3. it gives a reference point before introducing smarter search engines

## Current Evaluation Path

Each subset scenario reuses the same logic as the allocator workbench:

1. common-window synchronized step returns
2. per-subset capital split
3. block bootstrap portfolio resampling
4. same risk-dial objective

This keeps subset search and allocation on one metric path.

## Why This Matters

The greedy shortlist is only a proposal.

Combination search can overturn it when:

1. one selected candidate is only locally attractive
2. an exception candidate is a better diversifier
3. the best pair or triplet is not obvious from standalone scores

## Testing Principle

This layer needs stronger testing than earlier filters because it is easy to
introduce subtle regressions.

The current tests check at least:

1. pool construction keeps selected and exception candidates as intended
2. exhaustive search can prefer a diversified pair over a greedily selected
   alternative
3. pin/exclude overrides are honored during pool and subset construction
4. cluster caps propagate through the same weight builder used by allocator
3. registry save/load preserves combination reports

## Current Real Observation

Current real report:

1. [combinations_20250508_20250515_clustered.json](/home/user/mcs/candidate_registry/combinations/combinations_20250508_20250515_clustered.json)
2. source cluster report:
   [clusters_20250508_20250515.json](/home/user/mcs/candidate_registry/clusters/clusters_20250508_20250515.json)
3. source override set:
   [ops_20250508.json](/home/user/mcs/candidate_registry/overrides/ops_20250508.json)

Observed result on the current pool:

1. the pinned `wf_03_20250501_run02` candidate stayed in every admissible best
   subset
2. the best scenario shifted to a two-system subset:
   `wf_03_20250501_run02 + wf_01_20240101_run01`
3. both systems belong to the same dense cluster, so the `0.50` cluster cap
   bound the pair and marked weights with `cap_reason=cluster_cap`
4. the clustered pair still slightly beat the pinned single-candidate scenario
   on the current objective, which is exactly the kind of tradeoff the operator
   should be able to inspect in UI

## Next Extensions

Natural next steps:

1. beam or evolutionary search for larger pools
2. cluster-aware similarity collapse warnings in the UI
3. promotion from subset search result directly into allocator workbench UI
4. later live-state aware exclusions such as `draining` or `retired`
