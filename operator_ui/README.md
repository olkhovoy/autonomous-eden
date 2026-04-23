# Operator UI

Phase 1 operator dashboard over the snapshot feeds in
[candidate_registry/dashboard](/home/user/mcs/candidate_registry/dashboard) and
[candidate_registry/farm_dashboard](/home/user/mcs/candidate_registry/farm_dashboard).

## Run

```bash
cd operator_ui
npm install
npm run sync-feed
npm run dev
```

The app loads:

1. `public/data/dashboard-feed.json`
2. `public/data/farm-dashboard-feed.json`

Refresh the feed after rebuilding backend artifacts:

```bash
npm run sync-feed
```

## Farm View

Sync the farm feed:

```bash
npm run sync-feed -- --farm
```

Then open:

1. candidate view: `http://localhost:5173/`
2. farm view: `http://localhost:5173/?view=farm`

The farm view polls its feed every `5s` by default.
To override that:

1. `http://localhost:5173/?view=farm&refresh=2`
2. `http://localhost:5173/?view=farm&refresh=10`

For long farm runs, point the runner directly at the served farm feed path:

```bash
.venv/bin/python scripts/run_candidate_farm.py \
  --manifest-path /tmp/candidate_farm_manifest.json \
  --dashboard-sync-path operator_ui/public/data/farm-dashboard-feed.json
```

For long-running `rolling` or other stage-heavy scenarios, also use:

```bash
--heartbeat-interval-seconds 20
```

This writes periodic heartbeat snapshots so the farm view can tell a healthy
long step from a stale process.
