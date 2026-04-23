# UI Stack Options

Last updated: 2026-03-10

## Practical Goal

The first UI is an operator console for a long-running research process.
It should be good at:

1. dense sortable candidate tables
2. broom-style curve fans
3. cluster/concentration overlays
4. fast iteration over snapshot feeds
5. later promotion to near-real-time monitoring

## Recommended Path

### 1. First implementation: `React + TypeScript + Vite`

Why:

1. fastest path from the new dashboard feed to a working UI
2. no forced server architecture
3. ideal for a thin client over snapshot JSON feeds
4. easy to keep the rendering layer separate from research logic

Add for visualization:

1. `deck.gl` for GPU-heavy broom/ribbon style views
2. a grid component for the operator candidate table

This is the best first step.

### 2. Packaging path: `Tauri 2` around the same frontend

Why:

1. desktop operator console fits the use case better than a public web app
2. same Vite/React frontend can be wrapped without rewriting UI
3. easier future access to local files, notifications, and operator workstation

This should be added after the browser version proves the workflow.

### 3. Web deployment option: `Next.js`

Use this only if we need:

1. multi-user authenticated deployment
2. server-rendered routes
3. integrated backend endpoints and hosting conventions

For the first operator dashboard this is probably heavier than needed.

### 4. Python-first prototype option: `Dash`

Use this only if the priority is:

1. extremely fast Python-only prototyping
2. minimal frontend coding

Tradeoff:

1. weaker fit for a custom, GPU-heavy, pattern-recognition-oriented operator UI
2. likely to become constraining once broom and cluster views get ambitious

## Recommendation

Recommended build order:

1. ship the first dashboard as `Vite + React + TypeScript`
2. use `deck.gl` for broom and other dense GPU views
3. if the operator workflow proves out, wrap the same app with `Tauri 2`

This gives the cleanest path from the current snapshot-feed backend to a serious
operator console without overcommitting to infrastructure too early.
