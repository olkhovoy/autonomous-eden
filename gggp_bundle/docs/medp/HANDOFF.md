# MEDP — Cross-IDE handoff (Cursor → Windsurf)

**Updated (UTC):** 2026-04-23T07:44:43Z

## Last known state

- **A1.1**: success (NC4-only path); tag `gggp_bundle-medp/A1.1-success`.
- **A2**: plan drafted in `branches/A2/plan.md`; **Q1/Q2/Q3 locked** (see `Locked Decisions` there); **execution not started** (first unit: **S1a**).
- `log.jsonl` contains `event: answers_locked` for branch `A2` (tail after this file was written).

## Read order for a new session (repo-only)

1. `docs/medp/ROOT.md` — protocol, gates, F0.d commit/tag rules.
2. `docs/medp/log.jsonl` — read last ~5 lines (`tail -n 5 docs/medp/log.jsonl`) for recent events.
3. `docs/medp/branches/A1.1/postmortem.md` — what A1.1 closed.
4. `docs/medp/branches/A2/plan.md` — locked decisions, unit table, mermaid graph, CTRL semantics, critical path.
5. **Local (not in git):** `.windsurf/rules/PLAN-DECOMPOSITION-AND-AGENT-ROUTING.md` — unit format and agent routing (mirror of Cursor rule; `.windsurf/` is gitignored at parent repo root; sync machine-local between IDEs).

## First executable unit

- **S1a** — Rust: `CTRL` / `SCALE_BY_CODE` / `ADD_CODE` in `VectorOp` + parser + execution (see `plan.md` §Key files).
- Recommended agent profile from decomposition: **`opus-4.6`**, `thinking=yes`, `effort=high` (see unit table in `plan.md`).

## Do not commit (artifacts)

Outputs under `demos/semiotic_hypercube/` matching `*.npy`, `*.json`, `*.jsonl` are **gitignored** — keep them local; cite paths in reports only.

## MEDP commits and tags (F0.d)

Per `ROOT.md` §6.6 / §9: while `gggp_bundle/` lives inside the parent repo, use commit prefix **`medp(<branch>): ...`** and tags like **`gggp_bundle-medp/<branch>-<tag>`** (e.g. `medp(A2): start`, `gggp_bundle-medp/A2-start`).

## Post-handoff checklist (Windsurf)

- [ ] `git status` — review any staged research files (e.g. Tóth prior art) plus new `docs/medp/*` handoff edits.
- [ ] Decide: one commit (research + A2 lock) or split (research vs A2-lock).
- [ ] Run **S1a** (Rust CTRL ops) under the chosen agent profile.

## Continuity note

The Cursor-local plan file under `~/.cursor/plans/` was **not** edited; all durable state for Windsurf is in this tree (`HANDOFF.md`, `branches/A2/plan.md`, `log.jsonl`).
