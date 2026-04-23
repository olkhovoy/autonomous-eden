# Multi-Market Architecture

Last updated: 2026-03-10

## Purpose

The current implementation scope is still close to:

1. one instrument
2. one venue
3. one account view

This is acceptable as a temporary simplification.
It must not become a hidden architectural assumption.

The goal of this note is to prevent another legacy-style overload of manual
work when the system eventually expands to many instruments and venues.

## Non-Negotiable Future Dimensions

Every serious market object should eventually be scoped by some combination of:

1. `venue_id`
2. `symbol`
3. `instrument_type`
4. `contract_type`
5. `quote_currency`
6. `account_scope`
7. `dataset_id` or data version
8. `timeframe_family`

Even if current code carries only one `data_path`, new architectural work should
assume these dimensions exist.

## Guardrails For New Development

### 1. Do Not Treat `data_path` As Identity

`data_path` is an implementation detail.
Candidate identity, cycle identity, and forward plans should conceptually be
about market scope, not about one filesystem path.

### 2. Keep Candidate Scope Explicit

A candidate should always be thought of as:

1. one engine family
2. one representation family
3. one market scope
4. one training window
5. one seed/run identity

When schemas are extended later, market scope should become explicit rather than
being inferred from file naming.

### 3. Keep Portfolio Logic Above Market Logic

Allocator and lifecycle layers should operate on systems with explicit market
labels, not on anonymous checkpoints.

That will allow:

1. within-market diversification
2. cross-market diversification
3. venue concentration limits
4. account-level caps

### 4. Separate Per-Market Search From Cross-Market Allocation

The clean future shape is:

1. feature factory per market
2. candidate generation per market or market family
3. research filtering per market
4. portfolio allocator above all markets

This avoids mixing raw search mechanics with account-level portfolio control.

### 5. Keep UI Filters Ready For Market Scope

The operator dashboard should eventually filter and group by:

1. venue
2. symbol
3. market family
4. account scope
5. candidate family

UI state and registry artifacts should be designed so these dimensions can be
added without redesigning the whole dashboard.

### 6. Avoid Hardcoded Single-Market Names In New APIs

New scripts, modules, and docs should avoid baking `BTCUSDT` or one exchange
into their conceptual contracts, even if examples still use them.

### 7. Tradeforward Must Remain General

`tradeforward` should be thought of as:

1. one selected subset
2. with explicit market scopes
3. and portfolio weights
4. for one next forward window

not as:

1. one BTC system on one venue

## Deferred Implementation

This note does not require immediate code changes for:

1. multi-venue data loading
2. cross-market feature factories
3. multi-account routing
4. real-time venue adapters

It only fixes the architectural direction now, while the codebase is still
small enough to steer cleanly.
