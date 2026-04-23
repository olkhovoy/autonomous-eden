# Candidate Shortlist

Last updated: 2026-03-10

## Purpose

The shortlist layer is the bridge between:

1. single-candidate robustness
2. pairwise diversification
3. operator review
4. later allocator logic

It does not allocate capital yet.
It proposes a compatible subset and explains why.

## Current Implementation

Core module:

1. [combination.py](/home/user/mcs/umc_nn/candidates/combination.py)

CLI:

1. [run_candidate_shortlist.py](/home/user/mcs/scripts/run_candidate_shortlist.py)

Registry artifact directory:

1. [candidate_registry/shortlists](/home/user/mcs/candidate_registry/shortlists)

## Current Selection Logic

The first version is a greedy diversified shortlist:

1. compute one standalone base score per candidate
2. compute one compatibility score per pair
3. select the first candidate by standalone quality
4. add later candidates by marginal score:
   `base score + average compatibility with already selected candidates`
5. stop when the next marginal score is too weak or the subset is full

This is intentionally simple.
The goal is to produce an explainable operator-facing shortlist before moving
to heavier combination search.

## Current Base Score Inputs

Base score uses:

1. adjacent OOS return
2. OOS beats-flat flag
3. OOS activity
4. train resampling pessimistic return
5. train resampling profitable rate
6. train resampling pessimistic drawdown
7. ruin penalty
8. diversifier potential from the best pairwise compatibilities
9. flat-collapse penalty

## Current Pair Compatibility Inputs

Compatibility rewards:

1. drawdown improvement
2. negative return correlation
3. opposite-side overlap

Compatibility penalizes:

1. downside correlation
2. simultaneous loss rate
3. excessive action agreement
4. excessive same-side overlap

This is not yet a portfolio optimizer.
It is a first shortlist heuristic for operator review.

## Explainability For UI

Each shortlist report stores:

1. selected candidate ids
2. candidate-level score components
3. candidate brightness hints for curve visualization
4. exception flags
5. selected-pair compatibility breakdowns

That means the future UI can show both:

1. why a candidate was selected
2. why a surprising near-miss is still worth operator attention

## Current Real Example

Current real shortlist:

1. [shortlist_20250508_20250515_f1_00.json](/home/user/mcs/candidate_registry/shortlists/shortlist_20250508_20250515_f1_00.json)

Current outcome on the six-candidate probe pool:

1. `wf_03_20250501_run02` selected first by standalone quality
2. `wf_02_20240601_run02` selected second by positive compatibility
3. shortlist stopped at two candidates because the next marginal score was not
   attractive enough

This is the correct behavior for now:

1. propose a manageable subset
2. leave room for operator judgment
3. avoid pretending the allocator is already solved
