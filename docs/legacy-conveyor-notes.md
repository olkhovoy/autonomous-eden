# Legacy Conveyor Notes

Last updated: 2026-03-10

Source inspected:

1. `core.7z`
2. extracted subset:
   `core/src/COMMON`
   `core/src/GPBASE`

Important:

The legacy platform is not a single-strategy monolith.
It is a plugin-based multi-system conveyor with explicit:

1. strategy contracts
2. trade and signal state tracking
3. period-split statistics
4. multiple fitness functions
5. rule-based candidate filtering
6. trade resampling and capital sizing
7. session management
8. candidate enable/disable lifecycle

Critical operator clarification:

The highest-level portfolio decisions in the legacy system were performed
manually by a trained human operator, not by a fully automated meta-layer.

Confirmed manual operator responsibilities:

1. choosing the effective trade-resampling score for capital allocation
2. disabling systems based on UI statistics plus automatic DD-based shutdown
3. judging similarity and concentration using filters, charts, and views
4. running rebalance and replacement roughly twice per month as staff capacity
   allowed

Related plan:

1. [legacy-modern-rebuild-plan.md](./legacy-modern-rebuild-plan.md)

## Confirmed Legacy Building Blocks

### 1. Strategy / Execution Contract

Key files:

1. `core/src/COMMON/uPluginIntf.pas`
2. `core/src/COMMON/uPluginSignal.pas`
3. `core/src/COMMON/uPluginTrade.pas`

Observed:

1. The system exposes a large plugin contract around `TamInstance`,
   `TamSignalManager`, `TamTrade`, and `TamOrderManager`.
2. Signal management, order management, and statistics are separate concepts.
3. Trading settings are explicit:
   capital, slippage, fill type, commission model, margin, position size,
   single-entry constraints, time range filters.
4. The signal layer maintains per-signal state and open/closed signal positions.
5. The execution layer supports multiple order managers:
   `sim`, `neo`, `mbt`, `perfect`.

Implication:

The modern rebuild should keep the same separation:

1. candidate alpha logic
2. signal state
3. execution simulator / broker adapter
4. portfolio allocator / supervisor

### 2. Period-Split Statistics Engine

Key file:

1. `core/src/COMMON/uStats.pas`

Observed:

1. The legacy code has explicit period types:
   `ptTrain`, `ptValid`, `ptOutofsample`, `ptInsample`,
   `ptValidAndOutofsample`, `ptTotal`.
2. Trade statistics are accumulated per period, not only globally.
3. Period boundaries are encoded once and reused by all stats calculations.
4. There is built-in trade resampling support inside the stats layer.

Implication:

The new system should not treat OOS as an afterthought.
Every candidate must carry a period-split stats object as a first-class record.

### 3. Fitness Is Pluggable

Key file:

1. `core/src/COMMON/uFitness.pas`

Observed:

1. The legacy platform supports many named fitness functions.
2. Fitness is evaluated over train, valid, insample, out-of-sample, and total.
3. Fitness is separate from the strategy contract and separate from stats.

Implication:

The current Python monolith evolution is too narrow.
We need a pluggable fitness layer, not one hardcoded objective.

### 4. Trade Resampling and Capital Sizing

Key file:

1. `core/src/COMMON/frResampling.pas`

Observed:

1. The legacy platform resamples trades directly, not bars.
2. It preserves trade overlap using an `open_trades` structure.
3. It simulates equity curves from resampled trade sequences.
4. It records original net profit and drawdown, mean/stdev of resampled
   net profit and drawdown, and percentile distributions.
5. Position sizing is explicit inside the resampling engine:
   `fixed`, `fixed fraction`, and aggregate-linked sizing.
6. Allocation is explicit and separate from signal generation.
7. `GGGP` is used to search combinations of systems, not only parameters
   inside a single strategy.

Implication:

This is the strongest confirmation of the intended rebuild direction.
The modern project should evolve:

1. candidate systems
2. system combinations
3. allocation hyperparameters

not just one monolithic policy network.

Important clarification:

The legacy code exposed the resampling machinery, but the final sizing choice
was made by the operator. So the first faithful rebuild target is not a fully
automatic allocator. It is an allocator workbench that exposes the same
robustness information to either:

1. a human operator
2. a later automation layer

### 5. Rule-Based Candidate Filtering

Key files:

1. `core/src/COMMON/frSelectionRules.pas`
2. `core/src/COMMON/frCombinations.pas`

Observed:

1. Candidate selection is rule-driven.
2. Rules are composable boolean/numeric expressions over named statistics.
3. Combination and candidate views use these rules to decide visibility,
   rejection, and selection.

Implication:

The modern rebuild needs a candidate rule engine:

1. minimum trades
2. max drawdown
3. resampling percentile thresholds
4. OOS degradation thresholds
5. diversification constraints

### 6. Candidate Lifecycle Exists Explicitly

Key file:

1. `core/src/COMMON/frCombinations.pas`

Observed:

Candidate UI state distinguishes:

1. disabled candidate
2. enabled candidate
3. switching on leader
4. active leader
5. switching off leader

Implication:

The old platform was not just a research tool.
It had explicit runtime lifecycle for systems entering and leaving service.

Important clarification:

Lifecycle decisions were not purely automatic. The operator reviewed the
statistics UI and the platform also had automatic shutdown on statistical or
drawdown deterioration.

The new architecture therefore needs:

1. candidate registry
2. state machine:
   `research -> approved -> paper/live shadow -> active -> draining -> retired`
3. hot-swap logic for portfolio membership

### 7. Session Management Is Separate

Key file:

1. `core/src/COMMON/frSessionsManager.pas`

Observed:

1. Sessions and session groups are configured separately from strategies.
2. Timezone-aware include/exclude windows exist as first-class config.
3. End-of-day behavior is explicit.

Implication:

Trading eligibility by session should remain outside candidate alpha logic.

## Legacy Was Human-In-The-Loop

The old platform should be treated as a semi-automatic trading operating
system.

The automation boundary in the legacy workflow was approximately:

1. automatic candidate generation
2. automatic simulation and statistics
3. automatic trade resampling
4. automatic drawdown-based disable conditions
5. manual score interpretation
6. manual portfolio composition
7. manual concentration control
8. manual rebalance cadence

Implication:

The first successful modern rebuild does not need to automate every decision.
It needs to restore the operator's leverage:

1. generate many candidates quickly
2. summarize them correctly
3. show robustness and similarity clearly
4. make enable / disable / allocation decisions tractable

## What This Means For The New Architecture

The rebuild target should be a modern version of the following conveyor:

1. feature factory
   neurobars and other representations
2. candidate generator
   many short evolutions / searches producing disposable systems
3. period-split evaluator
   train / valid / OOS stats for each candidate
4. resampling robustness engine
   trade-level Monte Carlo / bootstrap with overlap-aware equity replay
5. rule engine
   explicit thresholds over stats and robustness
6. diversification / combination search
   choose subsets of candidates
7. allocator
   assign capital fractions
8. supervisor
   live monitoring, drift checks, disable and replacement
9. replacement loop
   continuously generate new systems

## Immediate Porting Priorities

1. Rebuild candidate registry and stats schema first.
2. Rebuild trade-resampling engine second.
3. Rebuild rule engine third.
4. Rebuild combination search and allocator after that.
5. Only then revisit live execution / hot swap.

## Confirmed Modern Rebuild Principle

The correct target is not:

1. one highly general monolith

The correct target is:

1. a farm of weak-to-medium, replaceable, explicitly monitored systems
2. aggregated by a robustness-aware allocator

## Still To Inspect Later

Likely relevant files for a deeper pass:

1. `uPluginHOST.pas`
2. `uPluginThread.pas`
3. `uPlugin.pas`
4. `uPluginStats.pas`
5. `frSignals.pas`

Those should clarify runtime orchestration and live enable/disable behavior in
more detail.
