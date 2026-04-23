# Multi-Stream Phase A Source Matrix

Last updated: 2026-03-15

This document turns `Phase A` from [multi-stream-intelligence-architecture.md](/home/user/mcs/docs/multi-stream-intelligence-architecture.md)
into a practical acquisition plan.

The goal of Phase A is not to ingest everything immediately.
The goal is to:

1. choose the first useful stream set for the current BTC-first research stack
2. define `available_at` and freshness semantics per source family
3. produce a concrete list of access actions
4. avoid premium-data assumptions in v1

## Phase A Deliverables

By the end of Phase A, the repo should have:

1. a reviewed source matrix
2. one normalized source inventory with access status
3. one availability model per source family
4. one access-action backlog with owner and status
5. one approved v1 baseline stack

Current inventory artifact:

1. [multi-stream-phase-a-source-inventory.json](/home/user/mcs/docs/multi-stream-phase-a-source-inventory.json)

## Current Known Access Status

As of `2026-03-15`, the project already has access or subscription to:

1. `FRED`
2. `Bybit`
3. `CoinGlass`

This changes the immediate priority of Phase A:

1. `FRED`, `Bybit`, and `CoinGlass` move from procurement into inventory and contract-definition work
2. the next access decisions shift toward `Binance`, `Deribit`, `Polygon`, and one slow exogenous context source

## Access Model Categories

Each source should be classified into one of these access models:

1. `public_no_key`
2. `self_service_key`
3. `account_plus_key`
4. `paid_self_service`
5. `sales_contact`
6. `premium_enterprise`

This classification matters because it determines how quickly a stream can move
from architecture to actual ingestion.

## Recommended V1 Baseline Stack

The first useful stack should be intentionally conservative:

1. crypto venue-native market data
2. one cross-asset proxy feed
3. one official macro and revision-aware data path
4. one on-chain context provider
5. one logistics or inflation proxy
6. one structured event/news layer

Recommended v1 baseline:

1. `Binance` market data
2. `Bybit` market data
3. `Deribit` options and volatility context
4. `Polygon` for ETF/index/session proxies
5. `FRED/ALFRED` for official macro and revisions
6. `CoinGlass` for aggregated crypto derivatives, ETF, and on-chain context
7. `Glassnode` or `CryptoQuant` only if `CoinGlass` coverage proves insufficient for the target hypotheses
8. `Freightos FBX` and/or `Truflation` for slow exogenous stress context
9. `Benzinga`-class or other structured event feed if available
10. `NewsAPI` only as fallback or prototyping enrichment, never as primary event backbone

## Source Matrix

### 1. Crypto Venue-Native Market Data

#### `Binance`

- Tier: `accessible baseline`
- Access model: `public_no_key` for public market data, `account_plus_key` for private/account paths
- Primary role: crypto spot/perpetual reference flow, microstructure, liquidity regime
- Why it belongs in v1: already close to the current research path
- Availability notes:
  - public market streams are near-real-time
  - session-closure logic is irrelevant because the market is `24/7`
  - still needs explicit outage and stale-feed handling
- Access actions:
  1. create or confirm exchange account if private data is later required
  2. verify region-specific API availability and permissions
  3. create market-data key only if private endpoints become necessary
  4. document websocket and historical replay path

#### `Bybit`

- Tier: `accessible baseline`
- Access model: `public_no_key` for public market data, `account_plus_key` for private/account paths
- Primary role: second crypto venue, cross-venue sanity and venue-divergence context
- Why it belongs in v1: reduces single-venue bias and improves regime/context coverage
- Availability notes:
  - `V5` public market endpoints and websocket streams are current as of `2026-03-15`
  - hostname and jurisdiction differences matter for some user groups
- Access actions:
  1. confirm public market data endpoints to use in research
  2. confirm whether a private API key is needed for later execution research
  3. record regional hostname constraints if applicable

#### `Deribit`

- Tier: `accessible baseline`
- Access model: `public_no_key` for market data, `account_plus_key` for private trading/account endpoints
- Primary role: options surface, implied vol, skew, term structure
- Why it belongs in v1: best accessible route to crypto derivatives context
- Availability notes:
  - options data is highly valuable as regime context
  - availability must be modeled on observed update cadence, not bar resampling convenience
- Access actions:
  1. verify required public endpoints for options surface history and live snapshots
  2. define which metrics are derived internally versus sourced directly

### 2. Cross-Asset Proxy Market Data

#### `Polygon`

- Tier: `accessible baseline`
- Access model: `paid_self_service`
- Primary role: ETFs, indices, risk proxies, session-based cross-asset context
- Why it belongs in v1: broad accessible coverage with usable developer ergonomics
- Availability notes:
  - session timing matters
  - flat files and delayed availability rules must be reflected in `available_at`
  - weekend BTC joins must preserve explicit stale or closed-market semantics
- Access actions:
  1. choose research plan tier with historical and websocket coverage required for v1
  2. confirm replay rights and historical granularity needed for offline studies
  3. record exact availability assumptions for each market family used

#### `Alpaca`

- Tier: `optional accessible`
- Access model: `account_plus_key`
- Primary role: alternative accessible equities or crypto path when Polygon scope is not enough
- Why it is not first-choice core: narrower fit for the current research target than Polygon plus venue-native crypto
- Availability notes:
  - market scope and jurisdiction coverage must be checked before treating it as a general market feed
- Access actions:
  1. create paper/live account only if required by selected stream scope
  2. document exact symbols and history coverage to avoid false assumptions

### 3. Macro And Revision-Aware Sources

#### `FRED / ALFRED`

- Tier: `accessible baseline`
- Access model: `self_service_key`
- Primary role: official macro series plus revision-aware historical state
- Why it belongs in v1: it solves the revision problem cheaply and correctly
- Availability notes:
  - `FRED` is not enough on its own if revised series matter
  - `ALFRED` or equivalent revision-aware handling is required for as-of replay
- Access actions:
  1. create `FRED` account
  2. request API key
  3. define which series require revision-aware storage
  4. record release calendars and release-time semantics separately from series values

#### Official Release Calendars And Policy Pages

- Tier: `accessible baseline`
- Access model: `public_no_key`
- Primary role: publication-time truth for macro and policy events
- Why they belong in v1: needed to avoid fake timing assumptions from secondary aggregators
- Availability notes:
  - the release schedule itself is a stream
  - revisions to schedules or unscheduled statements need their own event handling
- Access actions:
  1. enumerate target calendars for the first macro set
  2. define a normalized calendar-event schema before ingesting values

### 4. On-Chain Context

#### `CoinGlass`

- Tier: `accessible baseline`
- Access model: `paid_self_service`
- Primary role: aggregated crypto derivatives, ETF, spot, and on-chain context
- Why it belongs in v1: already available to the project and broad enough to seed the first multi-stream crypto context layer
- Availability notes:
  - official docs describe `V4` as a unified API across derivatives, options, spot, ETF, and on-chain markets
  - endpoint-level update cadence must still be recorded per metric family
  - aggregated vendor data should not erase exchange-level `observed_at` and `available_at` discipline
- Access actions:
  1. inventory the subscribed plan and endpoint families actually available
  2. choose the first metric shortlist instead of ingesting the entire product surface
  3. classify each selected metric as `microstructure`, `derivatives context`, `ETF flow`, or `on-chain slow context`

#### `Glassnode`

- Tier: `accessible baseline`
- Access model: `paid_self_service`
- Primary role: exchange flows, supply, transfer activity, stablecoin context
- Why it belongs in v1: strong structured on-chain coverage with mature API surface
- Availability notes:
  - metric-specific delays and update cadences must be recorded
  - different metrics can have different `available_at` and freshness rules
- Access actions:
  1. choose plan with API access matching intended metric set
  2. define initial metric shortlist instead of buying broad coverage blindly
  3. record per-metric update cadence and revision behavior

#### `CryptoQuant`

- Tier: `accessible baseline`
- Access model: `paid_self_service`
- Primary role: alternative or complementary on-chain context
- Why it belongs in v1: gives a second structured on-chain perspective
- Availability notes:
  - authentication is required
  - metric naming and release cadence need explicit mapping
- Access actions:
  1. create member account
  2. obtain API credentials
  3. compare metric overlap with Glassnode before committing to both

### 5. Logistics And Inflation Proxies

#### `Freightos FBX`

- Tier: `accessible baseline`
- Access model: `paid_self_service`
- Primary role: container-shipping stress and logistics regime context
- Why it belongs in v1: directly relevant to fragmentation and supply-chain regime hypotheses
- Availability notes:
  - not a high-frequency feed
  - the architecture should treat it as slow context with explicit carry policy
- Access actions:
  1. confirm exact subscription path and export rights
  2. verify historical depth and update schedule needed for research
  3. define carry and stale semantics for non-daily periods

#### `Truflation`

- Tier: `accessible baseline`
- Access model: `paid_self_service`
- Primary role: faster-moving inflation proxy and macro context
- Why it belongs in v1: complements official macro releases with more frequent alternative inflation context
- Availability notes:
  - must not be treated as a substitute for official CPI release timing
- Access actions:
  1. create account and verify API scope
  2. define exactly which series matter for crypto regime work

### 6. News And Event Sources

#### Structured Event Feed

- Tier: `accessible baseline if available`, otherwise `optional`
- Access model: usually `paid_self_service`
- Primary role: machine-readable event backbone
- Why it belongs in v1: raw text alone is too noisy for hot-path usage
- Availability notes:
  - event dedup and update threading are first-class requirements
  - `published_at` and `available_at` matter more than headline text
- Access actions:
  1. pick one first structured event source
  2. define `NewsDocument` and `EventRecord` normalization rules
  3. define dedup policy for updates and repeats

#### `NewsAPI`

- Tier: `prototype fallback`
- Access model: `paid_self_service`
- Primary role: broad web-news recall, prototyping, enrichment
- Why it is not the primary backbone: insufficient structure and finance-specific reliability for production event modeling
- Availability notes:
  - generic news retrieval must not be confused with machine-readable event intelligence
- Access actions:
  1. use only if needed for prototype recall expansion
  2. keep it outside the core required-source list

### 7. Premium Optional Sources

#### `LSEG Machine Readable News`

- Tier: `premium optional`
- Access model: `sales_contact`
- Role: institutional-grade structured news and event stream
- Why optional: high value, but should not block v1
- Access actions:
  1. open enterprise discussion only after baseline stack is working
  2. request sample payloads and replay/licensing terms

#### `RavenPack`

- Tier: `premium optional`
- Access model: `sales_contact`
- Role: structured news analytics and sentiment/event layers
- Why optional: strong enrichment path, but not needed to start
- Access actions:
  1. request enterprise information package
  2. evaluate only after internal event schema is stable

#### `Kpler` And `CDS`-Class Data

- Tier: `premium optional`
- Access model: `sales_contact` or `premium_enterprise`
- Role: shipping intelligence and sovereign-risk context
- Why optional: valuable, but expensive and likely overkill for first implementation
- Access actions:
  1. treat as stretch goal
  2. only open procurement path if accessible stack proves value first

## Availability Model Checklist

For every candidate source, Phase A must record:

1. whether the source is `24/7` or session-bound
2. expected update cadence
3. whether values are revised later
4. whether there is a separate release calendar
5. whether replay rights exist for historical research
6. whether weekend and holiday semantics are explicit
7. whether stale values should be `frozen`, `dropped`, or `carry_with_penalty`

## Ordered Access Backlog

This is the practical sequence I recommend.

### Stage 1: Immediate Self-Service Access

1. confirm `FRED` key scope and inventory the actual series needed
2. confirm `Bybit` public and private data contracts already available
3. inventory `CoinGlass` endpoint families included in the active subscription
4. confirm `Binance` and `Deribit` public market-data research paths
5. choose `Polygon` plan for cross-asset proxy coverage
6. evaluate `Freightos FBX` and `Truflation` pricing and historical depth

### Stage 2: First Event Layer

1. choose one structured event source if budget allows
2. if not, temporarily use a prototype stack with explicit downgrade status
3. keep `NewsAPI` fallback-only

### Stage 3: Second-Pass Coverage Decisions

1. decide whether both `Glassnode` and `CryptoQuant` are justified
2. decide whether `CoinGlass` removes the need for one of them
3. decide whether `Alpaca` adds anything beyond `Polygon`
4. decide whether one logistics source is enough for v1

### Stage 4: Premium Exploration

1. `LSEG MRN`
2. `RavenPack`
3. `Kpler`
4. `CDS`-class data vendors

Only do this after the accessible baseline stack is functioning and producing
useful regime/context features.

## Concrete Actions To Assign

These are the operational tasks that should exist in the project tracker.

1. Create a `source inventory` artifact with status fields:
   `candidate`, `approved`, `access requested`, `access granted`, `rejected`, `deferred`.
2. Record `FRED`, `Bybit`, and `CoinGlass` as `access granted`.
3. Confirm public market-data ingestion contracts for `Binance` and `Deribit`.
4. Select one cross-asset proxy provider and purchase the smallest tier that still supports offline research.
5. Inventory `CoinGlass` metrics and decide whether a second on-chain provider is still needed.
6. Decide whether `Freightos FBX`, `Truflation`, or both enter v1.
7. Select the first structured event source or explicitly record that event ingestion starts in degraded prototype mode.
8. Record replay rights, retention limits, and redistribution limits for every paid source.
9. Record `available_at` assumptions and stale policies per source before any feature engineering begins.
10. Add secret-management and credential-rotation requirements before private or paid APIs are wired into code.

## Recommended First Procurement Decision

If the project wants to move quickly without overcommitting budget, the first
purchase/access wave from this point should be:

1. `Polygon`
2. `Binance` contract-definition and access confirmation if private paths will matter
3. `Deribit`
4. one of `Freightos FBX` or `Truflation`
5. one of `Glassnode` or `CryptoQuant` only if `CoinGlass` proves insufficient

Keep:

1. `FRED`, `Bybit`, and `CoinGlass` in the immediate active baseline
2. `NewsAPI` out of the required core
3. `LSEG`, `RavenPack`, `Kpler`, and `CDS` providers in a deferred premium lane

## Official Access References

Access status and official access paths should be re-checked before purchase.
Access date for the links below: `2026-03-15`.

1. `FRED` API key docs:
   https://fred.stlouisfed.org/docs/api/api_key.html
2. `Binance` Spot API docs:
   https://developers.binance.com/docs/binance-spot-api-docs
3. `Bybit V5` API docs:
   https://bybit-exchange.github.io/docs/v5/intro
4. `Deribit` API docs:
   https://docs.deribit.com/
5. `Polygon` docs:
   https://polygon.io/docs
6. `Polygon` flat files:
   https://polygon.io/flat-files
7. `Alpaca` docs:
   https://docs.alpaca.markets/
8. `Glassnode` API docs:
    https://docs.glassnode.com/basic-api/api
9. `CryptoQuant` API auth docs:
    https://userguide.cryptoquant.com/api/authentication
10. `CoinGlass` API docs:
    https://docs.coinglass.com/v4.0/reference
11. `CoinGlass` API authentication:
    https://docs.coinglass.com/v4.0/reference/authentication
12. `Freightos FBX` access page:
    https://www.freightos.com/freightos-baltic-index/
13. `Truflation` API help:
    https://help.truflation.com/technical-resources/4Ubt733J7EEfNPz6XbX5X3/what-data-can-i-retrieve-using-the-api/4UCqW7hUyoZh2MjT3F8C8y
14. `NewsAPI` docs:
    https://newsapi.org/docs
