# Current Conveyor Audit

Date: 2026-03-17

Related diagram:
- `docs/current-conveyor-map.svg`

## Executive Summary

The current project is no longer a loose set of scripts. It is a real offline
research conveyor with:

1. data preparation and representation training
2. candidate generation
3. immutable candidate registry artifacts
4. trade export
5. candidate-level resampling
6. diversification, clustering, shortlist, allocator, and subset search
7. tradeforward, rolling simulation, lifecycle, portfolio ledger, and
   portfolio-level baselines
8. dashboard feeds and operator UI
9. a first multi-stream extension via `CoinGlass slow-context`

However, several important parts are still materially simpler than the target
architecture described in the rebuild plan:

1. the candidate engine is still one fixed monolith architecture with weight-only search
2. candidate records still carry only `train` and `oos`, not `train/valid/oos/total`
3. trade resampling is not overlap-aware equity replay yet
4. multi-stream ingestion is still a minimal `CoinGlass` branch, not the full
   architecture
5. the operator UI is a strong monitoring console, but it is still mostly
   snapshot/read-only, not a full operator workbench with write-back actions
6. multi-market support is architecturally anticipated, but not actually
   implemented

In short:

- the conveyor is real
- most of the meta-layer exists
- the weakest current layer is still the candidate engine and feature richness
- the most important implementation gap versus legacy is the simplified
  resampling/equity replay logic

## High-Level Runtime Flow

Current working sequence:

`market parquet` -> `optional slow-context merge` -> `prepare cache` ->
`neurobar training` -> `neurobar export (.npz)` -> `walk-forward probe /
candidate generation` -> `candidate registry` -> `trade export` ->
`candidate resampling` -> `diversification / clustering / shortlist` ->
`allocator / combination search / overrides` -> `tradeforward` ->
`rolling conveyor` -> `lifecycle` -> `portfolio ledger` ->
`portfolio baselines and gates` -> `candidate farm` -> `dashboard feeds` ->
`operator UI`

Main code paths:

1. trading environment and shared evaluator
   - `umc_nn/pure_env.py`
   - `umc_nn/trading_eval.py`
2. representation path
   - `experiments/neurobars_autoresearch/prepare.py`
   - `experiments/neurobars_autoresearch/train.py`
   - `experiments/neurobars_autoresearch/export_neurobars.py`
3. candidate engine
   - `scripts/evolve_monolith.py`
   - `scripts/probe_monolith_walkforward.py`
4. candidate meta-layer
   - `umc_nn/candidates/*.py`
   - `scripts/run_continuous_search_cycle.py`
   - `scripts/run_candidate_farm.py`
5. multi-stream extension
   - `umc_nn/multistream/*.py`
   - `scripts/build_coinglass_slow_context_v1.py`
   - `scripts/run_coinglass_slow_v1_ab.py`

## Inputs In The Current System

### 1. Anchor Market Inputs

The current honest v1 anchor frame is extracted by
`umc_nn/multistream/join_asof.py::extract_market_only_frame`.

Actual anchor columns:

1. `timestamp`
2. `open`
3. `high`
4. `low`
5. `close`
6. `volume`
7. `turnover`

This means the current clean baseline branch is:

1. price / volume only
2. single instrument
3. single venue history
4. no direct cross-asset context

### 2. CoinGlass Slow-Context Inputs

Current live `CoinGlass` branch adds only three slow-context families:

1. derivatives regime
   - open interest level and delta
   - funding last and rolling mean
   - long liquidation USD
   - short liquidation USD
   - liquidation imbalance
   - long/short ratio
2. sentiment / regime
   - fear & greed level and delta
   - stablecoin market cap level and delta
3. ETF regime
   - BTC ETF daily net flow
   - BTC ETF 5-day rolling flow

Each family also gets:

1. `cg_<family>_age_minutes`
2. `cg_<family>_is_stale`

Important implementation fact:

- joins are correctly backward `as-of available_at`
- there is no interpolation
- this is real code, not a mock branch
- but availability modeling is still simplified into fixed lag minutes per family

### 3. Representation Artifacts

Current downstream contract after export is still intentionally simple:

1. `timestamps`
2. `close_prices`
3. `neurobars`

This is what the candidate engine actually consumes.

### 4. Candidate-Level Inputs

Every candidate generation run is currently defined by:

1. one `.npz` neurobar dataset
2. one train window
3. one random run index
4. one economic configuration
5. one fixed monolith architecture

### 5. Operator / Meta Inputs

The operator/meta-layer currently accepts:

1. rule sets
2. resampling configs
3. clustering thresholds
4. shortlist limits
5. allocator risk fractions and caps
6. cluster overrides
7. candidate overrides
8. farm manifests

## Degrees Of Freedom That Exist Today

### A. Representation / Feature Degrees Of Freedom

Currently real:

1. branch choice
   - `baseline_market_only_v1`
   - `coinglass_slow_v1`
2. train time budget
3. dataset path
4. cache path
5. checkpoint path
6. export representation mode
   - `fused`
   - `base_multiscale`

But mostly fixed in code:

1. encoder family
   - `base_plus_multiscale_horizon_branches`
2. latent size
   - `32`
3. lookbacks
   - fast / mid / slow are fixed in `train.py`
4. loss family
   - weighted next-bar reconstruction + close delta + direction accuracy

Practical conclusion:

- representation freedom exists at branch / data / budget level
- architecture search does not yet exist

### B. Candidate Engine Degrees Of Freedom

Currently real:

1. window starts
2. `train_days`
3. `oos_days`
4. `runs_per_window`
5. `generations`
6. `population_size`
7. exchange / fee / slippage inputs
8. sizing mode / leverage / notional fraction
9. data path / representation choice

Currently missing:

1. architecture search
2. hidden size search
3. action space search
4. search over fitness families
5. search over session filters
6. search over trade management logic
7. search over multi-model ensembles inside one candidate

Important implementation fact:

The candidate engine is still one `UMCTradingCell`:

1. `input -> hidden(64) -> 3-way action head`
2. recurrent hidden state
3. action chosen by hard `argmax`
4. search space is flat parameter vector only

This is the biggest current bottleneck in candidate freedom.

### C. Selection / Portfolio Degrees Of Freedom

Currently real:

1. resampling fraction grid
2. resampling iterations and seed
3. diversification window
4. clustering threshold
5. shortlist size
6. shortlist pair hard limits
7. allocator risk dial
8. per-system cap
9. per-cluster cap
10. combination subset size bounds
11. override sets
12. rolling window geometry
13. farm manifest matrices and cases

This part of the system is already relatively rich.

### D. Operator Degrees Of Freedom

Currently real in backend artifacts:

1. candidate tagging and status changes
2. saved rules
3. override sets
4. shortlist selection artifacts
5. combination reports
6. allocator scenarios
7. farm manifests

Currently limited in UI:

1. monitoring is strong
2. direct write-back actions from UI are still limited / not the main path
3. many operator actions still happen through scripts and JSON artifacts

## What Is Implemented Versus Planned

### Implemented Well Enough To Be Considered Real

1. realistic trading environment with exchange presets and explicit sizing
2. shared evaluation path producing trades, equity, and metrics
3. neurobar autoresearch training and export path
4. walk-forward candidate generation
5. candidate registry and artifact persistence
6. trade export
7. candidate-level bootstrap resampling
8. diversification reports
9. cluster reports
10. shortlist
11. allocator workbench
12. combination search over bounded pools
13. tradeforward plan and evaluation
14. rolling conveyor simulator
15. lifecycle state machine
16. portfolio ledger
17. portfolio baselines and gates
18. manifest-driven candidate farm
19. dashboard feeds and farm dashboard feeds
20. operator UI

These are not mocks. They produce real on-disk artifacts under
`candidate_registry`, `checkpoints`, and the dashboard feed folders.

### Implemented But Materially Simplified

1. `PureTradingEnv`
   - real fees and explicit sizing
   - but still a simplified futures model
   - no funding
   - no liquidation model
   - no lot rounding
   - no exchange-specific contract semantics
2. candidate engine
   - real evolution
   - but only one fixed monolith topology
3. multi-stream
   - real `CoinGlass` live integration
   - but only one provider and one slow-context branch
4. dashboard
   - real monitoring UI
   - but mostly snapshot/read-only
5. rolling simulator
   - real historical conveyor replay
   - but still single-symbol, single-venue in practice

### Planned But Not Yet Actually Implemented

1. full period accounting in candidate record
   - planned: `train/valid/oos/total`
   - actual: mostly `train/oos`
2. overlap-aware trade resampling equity replay
   - planned
   - not yet implemented
3. smarter large-pool combination search
   - current search is exhaustive over small shortlist pools
4. live/source freshness observability across many feeds
   - only partial via farm heartbeat and `CoinGlass` stale markers
5. full multi-stream architecture from the design document
   - only `CoinGlass slow-context v1` exists
6. real multi-market execution and account scope support
7. live paper trading / exchange adapters
8. learned or optimized portfolio meta-search using `GGGP`

## Where The Code Still Contains Important Simplifications

### 1. Candidate Fitness Is Still Hardcoded

In `scripts/evolve_monolith.py`, candidate fitness is still a custom hardcoded
formula inside `evaluate_cell`.

This means:

1. evolution fitness is not derived from the shared evaluation module
2. candidate generation and later selection are still not perfectly aligned
3. changes to evaluation semantics can still diverge from generation semantics

This is one of the most important architectural mismatches still present.

### 2. Resampling Is Not The Legacy-Grade Engine Yet

`umc_nn/candidates/resampling.py` currently:

1. converts each trade into a scalar return on equity
2. resamples those returns independently
3. compounds them into a new path

What it does not do:

1. preserve overlapping open-trade structure
2. rebuild portfolio equity under actual overlapping trade sequences
3. model multi-system overlap at trade level

This is the clearest legacy-critical gap.

### 3. The Candidate Engine Is Weight Search Only

`scripts/evolve_monolith.py` plus `umc_nn/umc_cell.py` give a narrow search
space:

1. one fixed cell
2. one hidden width
3. one action space
4. one recurrent dynamic
5. one hardcoded fitness family

So the search is broad only in parameter count, not in model family.

### 4. Candidate Records Are Missing Some Planned Periods

The rebuild plan wanted one immutable record with:

1. train
2. valid
3. OOS
4. total

Actual candidate import from `register_walkforward_candidates.py` writes:

1. train
2. OOS
3. baseline maps
4. selection flags

This is a real gap, not just a documentation mismatch.

### 5. Multi-Stream Contracts Are Still Minimal

The architecture document defined many record types.
Actual code in `umc_nn/multistream/contracts.py` currently has only:

1. `MarketScope`
2. `NormalizedTimeSeriesRecord`

This is enough for the first branch, but it is still only the start of the
planned source architecture.

### 6. Operator UI Is Stronger As A Monitor Than As A Workbench

The UI is real and useful, but in its current form it is mainly:

1. feed visualization
2. monitoring
3. scenario inspection

It is not yet the full operator console where all approvals, edits, and
workflow changes happen in-place.

## Mocks, Fallbacks, And Dangerous Hidden Shortcuts

### Core Runtime Path

I did not find mock/stub code in the main offline conveyor path:

1. candidate generation is real
2. registry artifacts are real
3. resampling/diversification/allocator are real
4. rolling/lifecycle/ledger/baselines are real
5. dashboard feeds are real

### Test-Only Fakes

There are fake sessions and small synthetic payloads in tests, which is normal.

### Real Runtime Fallbacks To Be Aware Of

The most important runtime fallback is in `umc_nn/pure_env.py`:

1. if data loading fails, it falls back to synthetic data

This is not used in the normal research path when files are present, but it is
still a dangerous fallback and should probably be hardened later into a hard
failure for production research commands.

### Dry-Run Modes

`dry-run` exists in orchestration scripts such as:

1. `run_continuous_search_cycle.py`
2. `run_candidate_farm.py`

These are explicit orchestration modes, not hidden mocks.

## Where Data Is Still Missing

### 1. Market Coverage

Current practical coverage is still:

1. one main instrument: `BTCUSDT`
2. one historical anchor market frame
3. one main execution venue assumption at a time

Missing:

1. multi-symbol training
2. multi-venue synchronized anchor frames
3. futures basis / perp vs spot cross-venue context
4. FX / rates / index / commodity proxy context in the live feature path

### 2. Exogenous Context

Only the first exogenous branch is live:

1. `CoinGlass slow-context v1`

Missing:

1. FRED features in the actual joined frame
2. Bybit exchange-native context integration
3. Deribit options context
4. Polygon cross-asset proxies
5. event/news structured pipeline
6. revision-aware macro storage

### 3. Availability Modeling

Current `CoinGlass` branch uses:

1. fixed lag minutes
2. family-level stale flags

Missing:

1. richer per-endpoint availability rules
2. revision-aware replays
3. full source freshness observability across all sources

## Where Search Freedom Is Still Missing

### 1. Representation Freedom

Missing:

1. more than one encoder family
2. broader latent sizes
3. horizon-aware supervision families beyond the current setup
4. direct optimization toward downstream trading utility

### 2. Candidate Engine Freedom

Missing:

1. structure search
2. regime-conditioned candidates
3. session-aware candidates
4. explicit exit logic search
5. candidate ensembles
6. alternative policies besides the current monolith

This is the single strongest candidate for the next big improvement.

### 3. Meta-Layer Freedom

The meta-layer has more freedom than the engine, but still misses:

1. large-pool intelligent search
2. optimization of rule thresholds by outer search
3. allocator search using richer objectives
4. multi-market capital allocation

## What The Current CoinGlass Result Actually Means

The latest A/B result is informative:

1. representation metrics improved a lot
2. median OOS PnL improved
3. active OOS behavior got worse

This suggests:

1. more context is likely useful
2. the current candidate engine cannot yet convert that context into healthy
   active behavior
3. the next bottleneck is not only data
4. it is also the narrow candidate engine and current objective design

## Bottom-Line Audit

### What Is Solid

1. the meta-conveyor is real
2. the artifact discipline is good
3. the portfolio machine can already be evaluated honestly
4. the UI and dashboard layers are useful in practice

### What Is Not Yet Legacy-Equivalent

1. resampling fidelity
2. period accounting completeness
3. candidate engine breadth
4. operator workflow completeness in the UI
5. multi-market and multi-stream depth

### Where The Biggest Missing Edge Probably Is

Most likely missing edge is a combination of:

1. richer context data
2. broader candidate search families
3. more faithful resampling / approval logic
4. better alignment between generation fitness and downstream evaluation

If only one thing should be called out as the most important remaining
technical mismatch with the intended system, it is this:

`the meta-layer is already richer than the candidate engine and richer than the resampling engine beneath it`

That imbalance is probably the main reason the conveyor is technically
impressive but still not yet robustly profitable.
