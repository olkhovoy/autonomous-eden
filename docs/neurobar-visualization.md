# Neurobar Visualization

## Goal

Classical OHLC bars compress a short market interval into a compact visual
glyph: body, wick, direction, volatility. Neurobars need an equivalent visual
language for a latent vector over time.

The representation should answer three questions quickly:

1. What regime is the encoder seeing right now?
2. How is the latent state evolving through time?
3. Which latent changes line up with price moves, volatility, and trade actions?

## Recommended Canonical View

The most useful default view is a **latent ribbon**:

1. X-axis: time
2. Y-axis: latent dimension index
3. Color: signed z-scored latent activation
4. Optional alpha: latent magnitude confidence or reconstruction error

This is the latent analogue of candles. A candle compresses four prices into a
glyph. A latent ribbon compresses the full neurobar vector into a vertical
color stripe per timestamp.

## Companion Panels

Use the ribbon as the center panel, with three companion panels:

1. Price panel: candles or line price over the same time axis
2. Regime panel: rolling volatility, volume burst, realized direction
3. Policy panel: long/flat/short actions and equity curve

Together this becomes a trading-native latent dashboard instead of a generic
embedding plot.

## GPU-Friendly Views

### 1. WebGL Heatmap

Best first implementation.

- Render time x latent-dim as a texture
- One neurobar becomes one vertical slice
- Use a diverging color map centered at zero
- Overlay price and actions in synchronized panels

This scales well because the latent matrix is naturally image-like.

### 2. Manifold Flow

Project neurobars to 2D using PCA first, UMAP second if needed:

- point position: projected latent
- point color: future return, volatility, or regime
- trail: temporal order
- marker size: latent norm or reconstruction error

Useful for seeing clustering and regime recurrence, but it is secondary.
It should not replace the time-axis view.

### 3. Neuro-Candles

A custom glyph per bar:

- hue: direction of dominant latent component
- saturation: latent norm
- top wick: positive component spread
- bottom wick: negative component spread
- body thickness: reconstruction confidence or entropy

This can become visually expressive, but it is a derived visualization. It is
good after the ribbon, not before it.

## Practical Encoding Choices

To make the ribbon interpretable:

1. Reorder latent dimensions by correlation or hierarchical clustering, not raw index
2. Offer a toggle between raw latent values and rolling z-scores
3. Add event markers for major returns, volatility shocks, and policy flips
4. Allow brushing a time region and seeing its latent trajectory in 2D

## Suggested Implementation Path

### Phase 1

Build a synchronized offline viewer:

1. top: classic price candles
2. middle: latent ribbon heatmap
3. bottom: actions and equity

Matplotlib is enough for a first offline exporter.

### Phase 2

Move to a GPU/browser viewer:

1. WebGL heatmap for latent ribbon
2. WebGL line layer for price
3. WebGL scatter layer for actions/events
4. linked hover across panels

Good stacks:

1. Plotly WebGL for quick prototyping if installed
2. deck.gl or regl for a serious browser-native viewer
3. Datashader plus RAPIDS later if the dataset becomes too large

## Recommendation

The canonical neurobar visualization should be:

`price candles + latent ribbon + action/equity overlay`

That is the closest modern equivalent to OHLC bars for a latent market state.
It preserves time, scales to millions of points, and stays useful for trading
analysis instead of drifting into abstract embedding art.
