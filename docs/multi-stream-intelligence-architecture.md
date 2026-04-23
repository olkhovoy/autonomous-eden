 # Multi-Stream Intelligence Architecture

Last updated: 2026-03-15

This document defines the target architecture for the next evolution of the
research stack. It extends [multi-market-architecture.md](/home/user/mcs/docs/multi-market-architecture.md)
and keeps the current BTC-first conveyor as the first consumer, while removing
the hidden assumption that the system should learn from one price stream only.

The design stance is:

1. research-first
2. live-compatible
3. accessible-first vendor posture
4. assistive LLM usage only

`LLM` outputs are not allowed to sit in the hot execution path. They may help
with extraction, labeling, summarization, deduplication, and event enrichment.

## Purpose And Design Principles

### 1. Multiple Stream Families

The system should learn from several classes of information at once:

1. market-native flow
2. cross-asset state
3. macro and policy context
4. logistics and commodity stress
5. on-chain and offshore-dollar context
6. structured news and event streams

### 2. Strict Anti-Lookahead Timing Model

The governing timestamp for training joins and live feature use is
`available_at`.

The architecture must distinguish:

1. `observed_at`: when the underlying phenomenon happened
2. `published_at`: when the source published the value or document
3. `available_at`: when the system could first have consumed it

If a source is revised, corrected, or delayed, the historical training state
must reflect the as-of `available_at` view, not the final cleaned value.

### 3. Source Freshness And Graceful Degradation

Every stream must carry explicit freshness state. Missing or stale external
context must degrade the model into a known lower-information state rather than
silently leaking stale values forward or halting inference.

### 4. Separate Market-Native Signals From Exogenous Context

The system should not mix minute-by-minute price/volume flow with slow
exogenous context too early. Fast market structure and slow macro/event context
should be encoded separately and fused later.

### 5. Keep The Existing Conveyor

The current conveyor remains structurally valid:

`feature factory -> representation -> candidate engine -> registry ->
selection -> allocator -> lifecycle -> dashboard`

The multi-stream architecture changes the feature and representation layers,
not the top-level operator-assist conveyor.

## Stream Families

### 1. Market Data

Primary role: market-native and cross-asset state.

Examples:

1. crypto spot and perpetual microstructure
2. crypto options proxies and implied-volatility surfaces
3. ETF and futures proxies for session-based risk assets
4. volatility proxies
5. FX, rates, and index proxies

For v1, this family should be split into:

1. local traded market observations
2. cross-asset regime proxies

### 2. Macro Data

Primary role: slow context and regime change.

Examples:

1. policy-rate and yield-curve series
2. inflation proxies
3. official release calendars
4. central-bank event calendars
5. tariff, sanction, and trade-policy signals

### 3. Logistics And Commodity Stress

Primary role: supply stress and transport-risk context.

Examples:

1. container freight indices such as `FBX`
2. tanker or shipping-route stress proxies
3. energy curves and spreads
4. commodity transport bottleneck indicators

These are context/regime signals, not minute-level execution triggers.

### 4. On-Chain And Offshore-Dollar Context

Primary role: crypto-specific regime, liquidity, and transfer context.

Examples:

1. stablecoin supply and issuance
2. exchange inflow and outflow measures
3. chain-level transfer activity
4. derivatives-related on-chain settlement context

### 5. News And Event Streams

Primary role: structured event context.

Examples:

1. machine-readable headlines
2. policy statements and release notices
3. conflict escalation or logistics disruption events
4. company or venue-specific market structure events

Raw text should first be normalized into structured `EventRecord` objects.

### 6. Optional Premium Streams

These should be designed as pluggable replacements, not hard requirements:

1. `LSEG Machine Readable News`
2. `RavenPack`
3. `Kpler` or similar shipping intelligence
4. sovereign or corporate `CDS`-class data
5. Bloomberg/LSEG-class enterprise datasets

## Canonical Data Contracts

These contracts are architecture-level requirements. They do not force an
immediate implementation format, but all later code should conform to them.

Where a field does not naturally apply, keep it present and set it to `null`
rather than changing the shape per source.

### 1. `MarketScope`

Purpose: explicit identity for the market context of every record.

Required fields:

1. `market_scope_id`
2. `venue_id`
3. `symbol`
4. `instrument_type`
5. `contract_type`
6. `quote_currency`
7. `account_scope`
8. `timeframe_family`
9. `dataset_id`
10. `revision`

### 2. `SourceRecord`

Purpose: describes one upstream source.

Required fields:

1. `source_id`
2. `source_name`
3. `source_family`
4. `vendor_tier`
5. `observed_at`
6. `published_at`
7. `available_at`
8. `market_scope`
9. `revision`
10. `license_scope`
11. `latency_class`

### 3. `MarketObservation`

Purpose: one normalized market datapoint or bar-like observation.

Required fields:

1. `observation_id`
2. `source_id`
3. `market_scope`
4. `observed_at`
5. `published_at`
6. `available_at`
7. `revision`
8. `fields`
9. `quality_flags`

### 4. `MacroObservation`

Purpose: one normalized macro/policy/logistics observation.

Required fields:

1. `observation_id`
2. `source_id`
3. `market_scope`
4. `series_key`
5. `observed_at`
6. `published_at`
7. `available_at`
8. `revision`
9. `value`
10. `units`
11. `release_window`
12. `quality_flags`

### 5. `NewsDocument`

Purpose: raw or lightly normalized source document.

Required fields:

1. `document_id`
2. `source_id`
3. `market_scope`
4. `observed_at`
5. `published_at`
6. `available_at`
7. `revision`
8. `headline`
9. `body_ref`
10. `language`
11. `publisher`
12. `dedup_key`

### 6. `EventRecord`

Purpose: structured event extracted from one or more documents.

Required fields:

1. `event_id`
2. `source_id`
3. `market_scope`
4. `observed_at`
5. `published_at`
6. `available_at`
7. `revision`
8. `event_type`
9. `entities`
10. `regions`
11. `severity`
12. `confidence`
13. `supporting_documents`
14. `dedup_group`

### 7. `AvailabilityTimestamp`

Purpose: explicit timing metadata for replay and debugging.

Required fields:

1. `source_id`
2. `market_scope`
3. `observed_at`
4. `published_at`
5. `available_at`
6. `ingested_at`
7. `revision`
8. `availability_reason`

### 8. `RevisionVersion`

Purpose: track revisions and prevent rewritten history.

Required fields:

1. `entity_id`
2. `entity_type`
3. `source_id`
4. `market_scope`
5. `revision`
6. `observed_at`
7. `published_at`
8. `available_at`
9. `supersedes_revision`
10. `change_reason`

### 9. `SourceFreshnessStatus`

Purpose: operator-facing and model-facing freshness state.

Required fields:

1. `source_id`
2. `market_scope`
3. `observed_at`
4. `published_at`
5. `available_at`
6. `revision`
7. `freshness_state`
8. `expected_interval`
9. `stale_after`
10. `last_successful_ingest_at`
11. `degradation_mode`

### 10. `FeatureSlice`

Purpose: model-consumable slice built `as-of available_at`.

Required fields:

1. `feature_slice_id`
2. `source_id`
3. `market_scope`
4. `observed_at`
5. `published_at`
6. `available_at`
7. `revision`
8. `feature_family`
9. `feature_tensor_ref`
10. `coverage_summary`
11. `freshness_summary`
12. `join_policy_version`

## Time And Join Model

### 1. Governing Join Rule

All joins are `as-of available_at`.

The system must never join on:

1. final revised value time
2. document discovery time in the operator UI
3. bar close time if the value was only published later

### 2. 24/7 Crypto Versus Session Assets

Crypto trades continuously. Many useful context assets do not.

Therefore:

1. crypto observations remain continuous
2. session assets must carry exchange/session semantics
3. weekends and holidays must preserve explicit stale or frozen state
4. Monday reopen values must never leak into weekend BTC features

### 3. Revision-Aware Macro Storage

Macro and official series can be revised. Storage must keep:

1. initial release
2. subsequent revisions
3. the exact as-of view used in each training or replay run

Historical backfills must not overwrite prior research state.

### 4. Post-Release And Post-Close Discipline

Examples of prohibited leakage:

1. using a macro release value before its actual release timestamp
2. using official close-derived features during still-open sessions
3. using post-market ETF data as if it were available during the cash session

### 5. Stale-Value Policies

Every source must declare a stale policy:

1. `frozen`: last valid value is held with explicit stale marker
2. `drop`: feature family is masked out
3. `carry_with_penalty`: carry forward only with freshness penalty feature
4. `market_closed`: session-closed state, not outage

The chosen policy must be visible in both model features and operator
monitoring.

## Modeling Architecture

### 1. Separate Encoders

The target system should use separate encoders for:

1. crypto microstructure / local price-volume flow
2. cross-asset market-state
3. macro, logistics, and on-chain slow context
4. event/news context from structured event features

These encoders should not be collapsed into one raw mixed input stack at the
first layer.

### 2. Late Fusion

The preferred fusion order is:

`per-stream encoder -> stream family state -> regime/context layer ->
candidate-facing fused tensor`

This is consistent with the current repo's evolution away from a single
price-only representation.

### 3. Explicit Regime Layer

The candidate engine should receive both:

1. fused representation features
2. explicit regime state or regime probabilities

This regime layer may later be implemented with probabilistic clustering,
state-space models, HMM-like components, or learned regime heads. The
interface matters more than the first implementation.

### 4. `Neurobars v2`

The current neurobars path should be preserved as one representation family.

`Neurobars v2` should become:

1. multi-stream
2. late-fusion
3. availability-aware
4. regime-aware

The immediate purpose is better regime/context representation for the existing
candidate conveyor, not direct text-conditioned trading.

### 5. LLM Role

Allowed uses:

1. document summarization
2. event extraction
3. entity and region tagging
4. dedup support
5. weak labeling and research annotations

Disallowed use:

1. raw free-text LLM output directly triggering trades

## Integration With Existing Conveyor

### 1. Feature Factory Sits Before Representation Training

The new layer should sit before current representation training. The conveyor
becomes:

`raw streams -> normalized records -> as-of feature factory ->
representation training -> candidate generation -> registry -> selection ->
allocator -> lifecycle -> dashboard`

### 2. Candidate Generation Consumes Fused Feature Tensors

Candidate engines should consume fused, timestamp-disciplined tensors rather
than raw headlines or vendor-specific payloads.

### 3. Registry Artifacts Need Source Metadata

Candidate, cycle, tradeforward, and forward-eval artifacts should later carry:

1. `market_scope`
2. source coverage summary
3. freshness summary
4. feature-family availability summary

### 4. Dashboard Needs Source Observability

The dashboard should later expose:

1. source freshness
2. event load
3. current regime state
4. missing-stream degradation state

### 5. Allocator And Lifecycle Stay Above Source Logic

Allocator, lifecycle, and operator overrides remain portfolio-level layers.
They should not need to know source-specific schemas beyond coverage and
freshness summaries.

## Source Tiering

The system should explicitly separate accessible baseline inputs from optional
premium replacements.

### 1. Accessible Baseline

Good first-stack candidates:

1. crypto venue-native market data from major exchanges
2. `Polygon` for broad market proxies where timing and session semantics are
   acceptable
3. `Alpaca` where its market scope matches the use case
4. `Deribit` for crypto options and volatility context
5. official macro and policy sources such as release calendars and public data
   portals
6. `Freightos FBX` or similar accessible logistics series
7. `Truflation` or related accessible inflation proxies
8. `Glassnode` and `CryptoQuant`-class on-chain APIs
9. structured-but-accessible news/event sources

`NewsAPI` may exist only as prototype or fallback enrichment. It should not be
treated as the primary event backbone.

### 2. Premium Optional

Optional replacements:

1. `LSEG Machine Readable News`
2. `RavenPack`
3. `Kpler` and comparable shipping intelligence
4. enterprise-grade CDS, fixed-income, and cross-asset datasets
5. Bloomberg/LSEG-class premium data layers

These improve coverage and latency, but the core architecture must not depend
on them for v1.

### 3. Tiering Notes

Per source family, selection should consider:

1. latency and availability discipline
2. cost and licensing
3. replayability for offline research
4. revision behavior
5. usefulness for regime context versus execution timing

## Locked Decisions

The following decisions are intentionally fixed now:

1. `available_at` governs training joins and live feature use
2. all streams are normalized into explicit `market_scope`
3. `data_path` is never identity
4. news and LLM outputs enter only as structured events or features
5. premium feeds are pluggable replacements, not baseline requirements
6. macro, logistics, and on-chain streams are primarily context/regime inputs
7. weekends and market-closed periods use explicit stale or frozen semantics
8. allocator and lifecycle remain above source-specific ingestion details

## Implementation Roadmap

### Phase A: Reviewed Source Matrix And Availability Model

Deliverables:

1. per-stream source inventory
2. timing and freshness semantics
3. licensing and replay constraints
4. market-scope mapping rules

Reference artifact:

1. [multi-stream-phase-a-source-matrix.md](/home/user/mcs/docs/multi-stream-phase-a-source-matrix.md)

### Phase B: Raw Ingestion And Normalized Event Schema

Deliverables:

1. normalized record writers
2. `NewsDocument` and `EventRecord` schema implementation
3. dedup and revision tracking
4. source freshness tracking

### Phase C: As-Of Feature Store And Offline Replay

Deliverables:

1. `available_at` join engine
2. revision-aware replay
3. stale and degradation policies
4. train/research reproducibility on historical windows

### Phase D: Multi-Stream Representation Training

Deliverables:

1. stream-specific encoders
2. late-fusion representation
3. regime/context heads
4. `neurobars v2` research loop

### Phase E: Candidate-Engine Integration

Deliverables:

1. fused feature tensors into current candidate engines
2. registry metadata for source coverage and freshness
3. walk-forward evaluation against current gates

### Phase F: Dashboard And Source-Freshness Observability

Deliverables:

1. source freshness panels
2. event load and regime state panels
3. degradation and outage visibility
4. operator-facing explanations of missing or stale context

## Future Validation Cases

The architecture must explicitly pass these cases when implemented:

1. Friday traditional-market close must not leak Monday reopen values into
   weekend BTC features.
2. Revised macro releases must not rewrite prior as-of training state.
3. Missing news feeds must degrade to structured market and macro context
   rather than halt inference.
4. Stale source detection must surface in operator monitoring.
5. Event deduplication must handle repeated headlines and later updates from
   the same source.
6. A premium feed replacement must be able to slot into an existing stream
   family without redesigning allocator, lifecycle, or dashboard layers.
