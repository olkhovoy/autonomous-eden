# Neurobar Compression Roadmap

Last updated: 2026-03-09

## Current Direction

The active path is **multi-scale neurobars**.

Reason:

1. It is the smallest architectural step beyond single-scale compression.
2. It preserves fast information instead of smoothing it away.
3. It gives the downstream policy explicit short, medium, and slow context.
4. It does not require committing yet to more exotic frequency or recurrent designs.

## Active Variant

### Multi-Scale v1

Current design target:

1. baseline full-context branch over the full minute window
2. `fast` branch: last `32` minute bars at full resolution
3. `mid` branch: last `64` minute bars at full resolution
4. `slow` branch: last `128` minute bars at full resolution
5. multi-scale branches act as residual context instead of replacing the base path

Principle:

Do not denoise by deletion. Separate information by horizon and let the policy
learn what matters.

Note:

An earlier pooled-view variant was tested first and degraded the fixed
validation metric. It remains a parked sub-variant, but the active version now
avoids explicit smoothing before fusion and keeps the original full-context
path alive as the anchor representation.

## Initial Comparison

Fixed 5-minute runs on the neurobar autoresearch harness:

1. `single-scale baseline`: `val_score = 0.665862`
2. `pooled timeframe branches`: `val_score = 0.789640`
3. `full-resolution horizon branches only`: `val_score = 0.779698`
4. `base + multiscale residual branches`: `val_score = 0.510630`

Current conclusion:

The winning starting point is not a pure replacement of the baseline encoder.
It is a **baseline anchor plus multi-scale residual context**.

## Parked Next Options

These are intentionally preserved as the next candidates if multi-scale v1
fails to improve downstream trading behavior.

### 1. Context + Surprise

Split the representation into:

1. smooth expected context
2. residual surprise / anomaly channel

Use when the encoder starts overfitting to smooth reconstruction and misses
the rare events that matter for trading.

### 2. Multi-Horizon Supervision

Keep the encoder architecture but supervise on multiple targets:

1. next-bar features
2. future close delta
3. future direction
4. future realized range / volatility

Use when the latent is structurally clean but not obviously trade-relevant.

### 3. Wavelet / Band Decomposition

Decompose the minute stream into frequency bands and encode bands separately.

Use when multi-scale views still mix high-frequency and low-frequency behavior
too much.

### 4. Fast / Slow Recurrent State

Maintain two causal states:

1. fast state for microstructure and local reaction
2. slow state for regime and drift

Use when we need streaming inference efficiency or explicit memory separation.

### 5. Contextual Sparsity / Event Tokens

Represent only meaningful latent changes or event bursts instead of every bar
equally.

Use when most bars are visually and informationally redundant.

### Parked Sub-Variant: Pooled Timeframe Views

Approximate `1m / 4m / 8m` style branches with causal pooling before branch
encoding.

Reason parked:

The first implementation lost too much short-horizon information and worsened
the fixed validation metric.

## Evaluation Rule

No compression variant is considered successful unless it improves both:

1. the fixed neurobar validation metric
2. downstream trading baselines against the same `flat/long/short` slices
