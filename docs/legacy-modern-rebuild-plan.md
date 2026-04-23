# Legacy Modern Rebuild Plan

Last updated: 2026-03-10

## Purpose

This document defines the modern rebuild target for the legacy trading
conveyor discovered in `core.7z`.

It exists to keep the project aimed at the correct system:

1. not one monolithic strategy
2. not one best checkpoint
3. but a rotating portfolio of many replaceable systems
4. selected, sized, and disabled by explicit robustness logic

Important correction:

The legacy system was not fully automated at the portfolio meta-level.
The upper layer was human-in-the-loop:

1. operators interpreted resampling and stats views
2. operators chose portfolio composition and concentration limits
3. operators rebalanced and replaced systems roughly twice per month
4. the platform assisted with statistics and automatic drawdown-based shutdown

## Canonical Legacy Reference

Primary notes are in [legacy-conveyor-notes.md](./legacy-conveyor-notes.md).

The strongest legacy signals found so far are:

1. explicit train / valid / OOS / total period accounting
2. explicit trade-level resampling with overlap-aware equity replay
3. explicit rule-based candidate filtering
4. explicit combinations / allocator layer
5. explicit enable / disable / leader lifecycle
6. explicit session control outside alpha logic

## Modern Target Architecture

The rebuild should be organized as these layers.

The first target should be an operator-assist conveyor.
Full automation is a later layer, not the initial success criterion.

Current implementation scope note:

1. today the working path is still effectively one instrument and one venue
2. this is only an execution simplification, not an architectural decision
3. all new layers should be designed so the project can later carry many
   instruments, many venues, and multiple account scopes without a rewrite
4. multi-market guardrails are captured in
   [multi-market-architecture.md](./multi-market-architecture.md)

### 1. Feature Factory

Inputs:

1. minute market data
2. exchange metadata
3. optional cross-market or derived features

Outputs:

1. causal feature arrays
2. neurobars representations
3. future derived representations

Current state:

1. partially implemented
2. multi-scale neurobars experiments exist
3. still needs leakage-safe training promoted from experiment path into main path

### 2. Candidate Generator

Inputs:

1. one feature representation
2. one search spec
3. one train window
4. one random seed

Outputs:

1. many disposable candidate systems
2. their configs, checkpoints, and raw signal traces

Notes:

1. `GGGP` should search candidate hyperparameters and later candidate
   combinations, not only one policy vector.
2. Overfit local systems are acceptable if downstream filtering is strong.

### 3. Period-Split Evaluator

Each candidate must produce one immutable evaluation record containing:

1. train stats
2. valid stats
3. OOS stats
4. total stats
5. trade list
6. equity curve
7. action counts
8. config snapshot

This must use one shared evaluation module so evolution, visualizations,
baselines, portfolio logic, and reports cannot silently diverge.

### 4. Trade Resampling Engine

This is the first truly legacy-critical component to port well.

Required behavior:

1. resample trades, not bars
2. preserve overlapping open trades
3. rebuild equity curves under resampled sequences
4. report percentiles, mean, stddev, pessimistic net profit, pessimistic DD
5. support fixed and fraction-based sizing assumptions

This engine is the basis for candidate approval and capital allocation.

### 5. Rule Engine

Candidates should be filtered by declarative rules over named metrics, e.g.:

1. minimum trade count
2. max drawdown
3. minimum pessimistic resampled return
4. max degradation from train to valid or OOS
5. max correlation to already approved systems
6. session eligibility

This should be a data-driven layer, not hardcoded `if` statements spread across
scripts.

In the first rebuild phase this rule engine can support:

1. hard automatic gates
2. operator-defined saved filters
3. explainable approval and rejection reports

### 6. Diversification / Combination Search

This is the portfolio construction layer.

Inputs:

1. approved candidate pool
2. candidate trade streams
3. candidate robustness records
4. diversification constraints

Outputs:

1. selected subset or weighted subset
2. combination-level stats
3. replacement candidates

`GGGP` is a plausible search engine here because the search space is naturally
combinatorial.

Current state:

1. pairwise common-window diversification exists
2. a first greedy diversified shortlist layer exists
3. exhaustive subset search over shortlist pools now exists
4. cluster reports and override-aware subset search now exist
5. larger-pool smart search still does not exist

### 7. Capital Allocator

This layer converts candidate robustness into capital fractions.

Initial target:

1. use pessimistic trade-resampling metrics
2. cap per-system and per-cluster concentration
3. support `disabled`, `paper`, `active`, `draining`
4. separate gross weight from broker-executable position sizing

This is likely where the old platform earned much of its practical edge.

Important correction:

In the legacy system the final allocation decision was made by the operator.
Therefore the first modern version should be an allocator workbench with:

1. candidate robustness summaries
2. proposed default weights
3. editable operator overrides
4. audit trail of accepted and rejected allocations

Current state:

1. first allocator workbench exists
2. it includes a global risk dial over gross deployed capital
3. it uses common-window block-bootstrap portfolio resampling
4. cluster caps and operator overrides now flow through the same sizing path
5. operator override audit trail now exists
6. turnover-aware rebalance logic is still pending

### 8. Supervisor / Lifecycle Manager

Each active system should have explicit state:

1. `research`
2. `approved`
3. `paper`
4. `active`
5. `draining`
6. `retired`

Disable logic should compare live rolling stats against the training envelope.

In the first rebuild this should be hybrid:

1. automatic hard-disable on configured drawdown or statistical breakage
2. operator review for softer disable decisions

### 9. Replacement Loop

The system should continuously:

1. generate new candidates
2. evaluate them on fixed split windows
3. resample them
4. approve or reject them
5. rebalance active capital
6. retire degraded systems

This is the operating model. The trading edge is in the loop, not in a single
model artifact.

Legacy cadence note:

1. rebalance and replacement were approximately semi-monthly
2. cadence was constrained by operator and staff capacity

### 10. Operator Dashboard / Search Monitoring

The first production UI should be treated as part of the trading system, not
as an optional convenience layer.

Required operator-facing capabilities:

1. real-time filtered candidate list
2. saved filters and sortable score components
3. shortlist explanations
4. diversification review
5. search stagnation monitoring
6. "broom" visualization of normalized common-window equity curves
7. manual approval and override workflow with audit trail

## What The Current Repo Already Has

Useful pieces already present:

1. explicit simplified futures environment with exchange fee presets
2. shared trading evaluation module
3. baseline runner
4. simulator invariant tests
5. multi-scale neurobars experiment harness
6. phase-1 neurobar visualization
7. candidate registry
8. trade export and bootstrap resampling
9. pairwise diversification on common windows
10. greedy diversified shortlist with explainable score components
11. allocator workbench with scenario-based risk dial
12. exhaustive combination search baseline over shortlist pools
13. cluster similarity reports and concentration controls
14. operator overrides with audit trail wired into allocation and subset search
15. dashboard snapshot feed builder for UI consumption
16. phase-1 operator dashboard UI over the snapshot feed
17. first continuous-cycle launcher and tradeforward handoff artifacts

These are valid building blocks, but they currently support single-policy
evaluation better than portfolio-of-systems operation.

## What Is Missing

High-priority missing pieces:

1. live supervisor and disable audit trail
2. continuous candidate generation orchestration
3. larger-pool smart search beyond exhaustive baseline
4. lifecycle supervisor
5. continuous replacement orchestrator
6. richer real-time dashboard features beyond the current snapshot UI

Current implementation note:

1. first cycle launcher now exists for `reuse` and `generate` modes
2. it already produces cycle reports, refreshed portfolio artifacts, dashboard
   feed, and a tradeforward handoff plan
3. first manifest-driven candidate-farm runner now exists above the rolling
   conveyor layer
4. `generate` mode still uses the current walk-forward probe as the candidate
   engine, so farm throughput has improved at the orchestration level before it
   improves at the engine level

## Proposed Build Order

### Phase 0. Lock Core Contracts

Deliverables:

1. candidate record schema
2. trade record schema
3. period-split stats schema
4. experiment manifest schema

Do not build higher layers before these contracts are stable.

### Phase 0.5. Operator Workflow Capture

Deliverables:

1. saved filter schema
2. candidate review views
3. similarity and concentration diagnostics
4. approval and allocation audit schema

Success condition:

1. the modern stack can support the same human review loop the legacy platform
   relied on

### Phase 1. Candidate Registry

Deliverables:

1. on-disk registry for candidate metadata, metrics, and artifacts
2. immutable run manifests
3. candidate ids reproducible from config + seed + data window

Success condition:

1. many short evolutions can write comparable candidates into one registry

Current implementation note:

1. base schemas, on-disk registry, rule filtering, and walk-forward candidate
   import now exist in [candidate-registry.md](./candidate-registry.md)

### Phase 2. Trade Resampling Port

Deliverables:

1. overlap-aware trade bootstrap
2. pessimistic percentile metrics
3. unit tests against synthetic trade cases
4. CLI to evaluate one candidate and one candidate set

Success condition:

1. candidates can be ranked by robustness, not just raw train PnL

Current implementation note:

1. candidate-level trade export and single-stream bootstrap are now available in
   [trade-resampling.md](./trade-resampling.md)
2. overlap-aware multi-system resampling is still pending

### Phase 3. Rule Engine

Deliverables:

1. declarative selection rules
2. approved / rejected candidate views
3. reproducible filtering reports

Success condition:

1. candidate pool shrinks by rules without ad hoc script edits

### Phase 4. Diversification And Combination Search

Deliverables:

1. candidate similarity metrics
2. candidate subset search
3. combination-level backtest and robustness report

Success condition:

1. a portfolio beats the median single candidate more often than not

Current implementation note:

1. pairwise common-window diversification reports now exist in
   [candidate-diversification.md](./candidate-diversification.md)
2. shortlist search, exhaustive subset search, cluster reports, and
   override-aware combination search now exist
3. larger-pool smart search is still pending

### Phase 5. Allocator

Deliverables:

1. capital fractions from pessimistic robustness metrics
2. exposure caps
3. cluster caps
4. turnover-aware rebalance logic
5. operator override and sign-off path

Success condition:

1. portfolio equity is smoother than equal-weight selection on the same pool

Current implementation note:

1. allocator workbench, risk dial, cluster caps, and operator overrides now
   exist
2. portfolio ledger and turnover-aware rebalance transitions now exist
3. allocator output is still a research workbench until portfolio baselines are in place

### Phase 6. Supervisor / Lifecycle

Deliverables:

1. rolling live-vs-training drift checks
2. disable thresholds
3. state machine transitions
4. audit trail for activation and retirement

Success condition:

1. degraded systems can be removed without manual intervention

### Phase 7. Live Adapter Integration

Deliverables:

1. paper execution
2. live shadow mode
3. controlled activation of approved systems

This should come after the research conveyor is credible, not before.

## Probability Of Success

Short answer:

1. moderate, if we rebuild the conveyor
2. lower, if we insist on end-to-end full automation immediately
3. low, if we keep chasing one monolith

Rough estimate for the conveyor rebuild producing a credible research and paper
trading stack:

1. `70-80%` chance of reaching a technically sound operator-assist replacement
   for the old research workflow
2. `40-55%` chance of reaching a paper-trading portfolio that behaves
   plausibly enough to justify live shadow deployment
3. `15-25%` chance that a partially automated allocator can be trusted early
   without a long operator shadow period
4. `10-20%` chance that the first live-capital version will reproduce the
   original legacy economics closely enough without several further iterations

These are not probabilities of profit. They are probabilities of rebuilding the
machinery without losing the core logic.

## Main Reasons The Probability Is Not Higher

1. The legacy edge likely came from the full conveyor, not any single strategy.
2. The current repo still lacks funding, liquidation, lot rounding, and some
   execution realism.
3. Candidate diversification quality is still unmeasured.
4. The resampling engine has not yet been ported.
5. The allocator was probably a major source of practical robustness.
6. Live disable thresholds and lifecycle logic are not yet reconstructed.
7. Part of the legacy edge lived in operator judgment that has not yet been
   formalized.

## Questions Still To Resolve

1. What exact trade-resampling percentile or pessimistic score drove sizing?
2. How were systems disabled in live mode:
   by net profit drift, win-rate drift, DD burst, or a composite rule?
3. How were correlation or similarity limits enforced between systems?
4. How often was reallocation run?
5. Was capital allocated per strategy, per family, per market regime, or all
   three?
6. Which parts of the old execution model matter enough to reproduce exactly?
7. Which operator judgments can be formalized safely, and which should remain
   manual at first?

## Recommended Immediate Next Phase

Current agreed roadmap after the first continuous-cycle launcher is:

1. `tradeforward evaluator` implemented
2. rolling conveyor simulation across history implemented
3. lifecycle state transitions for systems moving through simulated service implemented
4. portfolio ledger and rebalance/turnover accounting implemented
5. portfolio-level baselines and gates implemented
6. only then increase candidate-farm throughput and meta-search complexity
7. keep real-time adapters and paper/live integration out of scope until the
   historical conveyor is strong enough

Detailed persistent roadmap:

1. [next-development-roadmap.md](./next-development-roadmap.md)
2. [continuous-search-cycle.md](./continuous-search-cycle.md)
3. [tradeforward.md](./tradeforward.md)

## Working Principle

The modern system should be judged by this standard:

1. Can it continuously produce, evaluate, review, allocate, and replace many
   systems with operator leverage similar to legacy?

Not by this standard:

1. Can one checkpoint look good on one equity curve?
