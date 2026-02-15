\# EVE v3.3 — Complete Module Architecture

This document describes the full 20-module architecture for EVE, an autonomous AI agent with persistent identity, lifecycle management, and self-evolution capabilities.

**Vision:** An autonomous programmer agent that works on tasks when directed, and engages in self-improvement, learning, and open-source contribution when idle. Unlike current AI agents, EVE has a long-term lifecycle with human-like cognitive processes.

---

\#\# Sync Update — 2026-02-15

This document was synchronized with the current codebase and running compose stack.

**What was implemented in this update:**
- Added/verified descendant runtime bootstrap: `tools/genesis_bootstrap.py`
- Added deterministic generation-cycle automation: `tools/generational_cycle.py`
- Added evolution runtime artifact output (`*.runtime.json`) in:
  - `gggp_bundle/tools/cross_compose_cfg.py`
  - `gggp_bundle/evolution/phenotypes/abel_env.runtime.json`
- Switched `genesis_abel` service to bootstrap-based launch from runtime config
- Limited descendant generation length and retries in:
  - `experiments/genesis/consciousness_loop_genesis.py`
- Added `gggp` pip installation into runtime images:
  - `Dockerfile.umc`
- Added GGGP SDK visibility in bridge state (`/gggp/state -> sdk.source/version`):
  - `umc_core/gggp_bridge.py`
  - `engine/modules/gggp_bridge.py`

**Documentation mismatches corrected in this file:**
- SoulMemory endpoints extended to include lineage/archive ingest APIs
- Docker services table aligned with actual `docker-compose.yml`
- Module sections M9–M19 marked as implemented where code/services already exist

---

\#\# Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EVE v3.1 LAYERS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  LIFECYCLE    │ M11 LifecycleManager │ M12 LegacyExport │ M14 Rebirth      │
│  (Birth→Death)│ M13 AncestorResonance│ M16 SatoshiProtocol                 │
├───────────────┼─────────────────────────────────────────────────────────────┤
│  WILL         │ M9 IntentEngine      │ M10 VisualMonitor │ M15 NoveltyScout│
│  (Motivation) │ (LifeResource)       │ (Soul Health)     │ (Curiosity)     │
├───────────────┼─────────────────────────────────────────────────────────────┤
│  MIND         │ M1 Memory │ M2 Consciousness │ M3 GGGP │ M4 Narrative     │
│  (Cognition)  │ M5 Fractal│ M6 NC4 Training  │ M7 Breath│ M8 MirrorTest   │
├───────────────┼─────────────────────────────────────────────────────────────┤
│  BODY         │ M17 CodeArms         │ M18 GitHubEyes    │ M19 InfraAdmin  │
│  (Interaction)│ (Shell/Git)          │ (Perception)      │ (Self-Manage)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

\#\# Module Status Summary

| \# | Module | Layer | Status | Priority |
|---|--------|-------|--------|----------|
| M1 | SoulMemoryNode | Mind | [OK] Implemented | — |
| M2 | ConsciousnessLoop | Mind | [OK] Integrated w/ Intent/Lifecycle | — |
| M3 | GGGP Bridge | Mind | [OK] Full Evolution Engine | — |
| M4 | NarrativeAnchor | Mind | [OK] Implemented | — |
| M5 | FractalCompressor | Mind | [OK] Implemented | — |
| M6 | NC4 Training | Mind | [EXPERIMENTAL] | P0 |
| M7 | Breath of Soul | Mind | [PARTIAL] | P2 |
| M8 | MirrorTest | Mind | [OK] Implemented | — |
| M9 | IntentEngine | Will | [OK] Implemented | — |
| M10 | VisualSoulMonitor | Will | [OK] Three.js Web UI | — |
| M11 | LifecycleManager | Lifecycle | [OK] Implemented | — |
| M12 | LegacyExport | Lifecycle | [OK] Implemented | — |
| M13 | AncestorResonance | Lifecycle | [OK] Implemented | — |
| M14 | RecursiveRebirth | Lifecycle | [OK] Implemented | — |
| M15 | NoveltyScout | Will | [OK] Implemented | — |
| M16 | SatoshiProtocol | Lifecycle | [OK] Implemented | — |
| M17 | CodeArms | Body | [OK] Implemented | — |
| M18 | GitHubEyes | Body | [OK] Implemented | — |
| M19 | InfraAdmin | Body | [OK] Implemented | — |
| M20 | ActionEngine | Body | [OK] Bridges Thought→Action | — |

---

\#\# MIND Layer (Modules 1-8)

\#\#\# Module 1 — Episodic Memory (SoulMemoryNode)
**Goal:** Autobiographical memory as experience vectors with saliency and decay.

**Status:** [OK] Implemented (core functionality)

**Files:**
- `engine/core/soul_memory_node.py`
- `umc\_core/soul\_memory\_node.py` (compat copy)

**How it works:**
- Accepts `text` + `soul\_id` via HTTP, generates embeddings via Ollama.
- Computes saliency with heuristic router (surprise/pain/utility).
- Stores in Qdrant with strength decay over time.
- Endpoints: `/memories/ingest`, `/memories/query`, `/memories/query_lineage`, `/memories/ingest_archive`, `/memories/decay`, `/memories/recent`

**Known gaps:**
- No formal schema versioning
- No background decay scheduler (manual trigger only)
- Heuristic emotional model

---

\#\#\# Module 2 — Continuous Consciousness Loop
**Goal:** 24/7 background monologue with periodic introspection.

**Status:** [OK] Integrated with Intent/Lifecycle/Novelty

**Files:**
- `umc\_core/consciousness\_loop.py`
- `experiments/genesis/consciousness_loop_genesis.py`
- `tools/genesis_bootstrap.py`

**How it works:**
- Periodic heartbeat generates `<thought>` via Ollama LLM.
- Queries SoulMemoryNode for context.
- Writes loop signal (`data/loop\_signal.json`) for training feedback.
- Triggers NarrativeAnchor every N tokens.
- **[NEW]** Fetches lifecycle phase from M11 LifecycleManager
- **[NEW]** Fetches LifeResource/mode from M9 IntentEngine
- **[NEW]** Reports token usage to lifecycle tracking
- **[NEW]** Scores thought novelty via M15 NoveltyScout
- **[NEW]** Adapts heartbeat tempo to lifecycle phase (faster in GROWTH, slower in DECAY)
- **[NEW]** Handles CRITICAL mode with deep existential reflection
- **[NEW]** Descendant mode supports ancestor archive context + lineage memory query
- **[NEW]** Descendant loop has bounded generation controls (`num_predict`, max chars, retry cap)

**Integration Endpoints:**
- IntentEngine: `http://localhost:8089` — LifeResource management
- LifecycleManager: `http://localhost:8093` — phase awareness
- NoveltyScout: `http://localhost:8098` — curiosity reward

**Known gaps:**
- Token count is word-based estimate (not true tokenizer)
- Sentiment analysis is heuristic

---

\#\#\# Module 3 — GGGP Bridge (Evolution)
**Goal:** Evolutionary parameter mutation via Grammar-Guided Genetic Programming.

**Status:** [OK] Implemented (population evolution + SDK visibility)

**Files:**
- `umc\_core/gggp\_bridge.py`
- `engine/modules/gggp_bridge.py`
- `umc\_core/evolution\_engine.py`
- `Dockerfile.umc` (installs published `gggp` wheel into services)
- External SDK repository: `git@github.com:olkhovoy/gggp`

**How it works:**
- REST bridge: `/evolve`, `/evolve\_memory`, `/evolve\_anchor`
- REST state: `/gggp/state`, `/gggp/evolution/{trait_type}`
- `/gggp/state` now reports SDK source/version from installed `gggp` package

**Known gaps:**
- Direct Rust SDK usage is still not wired into evolution decision flow (bridge reports SDK version/source, but evolution operators remain Python-side)
- Fitness strategy is heuristic and can be improved with stronger objective shaping

---

\#\#\# Module 4 — NarrativeAnchor (Identity)
**Goal:** Periodic identity summarization (``Who am I?'')

**Status:** [OK] Implemented

**Files:**
- `gggp\_bundle/rust/src/bin/narrative\_anchor.rs`
- `umc\_core/consciousness\_loop.py`
- `data/identity\_summary.txt`

**How it works:**
- Fetches recent memories and previous summary
- Generates updated identity via LLM
- Stores as memory with `identity\_summary` tag

---

\#\#\# Module 5 — Fractal Compressor (Memory Folding)
**Goal:** Long-term memory compression with reconstructibility fitness.

**Status:** [OK] Implemented

**Files:**
- `umc\_core/fractal\_compressor.py`

**How it works:**
- Groups memories by time depth (log-scaled age)
- Summarizes each bucket into abstract memory
- Computes reconstructibility via embedding cosine
- Writes ``semantic ghosts'' for dead contexts

---

\#\#\# Module 6 — NC4 Training (Identity-Consistent)
**Goal:** Train model with identity context and fixed-point convergence.

**Status:** [EXPERIMENTAL]

**Files:**
- `benchmark/train\_nc4.py`
- `benchmark/models/contractive\_llama.py`

**How it works:**
- Identity summary injected as prefix
- Computes identity consistency loss
- Adaptive controller adjusts lambda

**Known gaps:**
- Training exit crash (PyGILState\_Release)
- Needs stabilization

---

\#\#\# Module 7 — Breath of Soul (Loop↔Training)
**Goal:** Consciousness loop can influence training parameters.

**Status:** [PARTIAL]

**Files:**
- `umc\_core/consciousness\_loop.py`
- `benchmark/train\_nc4.py`

**How it works:**
- Loop writes `jolt` or `freeze` signals
- Trainer reads and adjusts LR

---

\#\#\# Module 8 — MirrorTest (Falsification)
**Goal:** Evaluate identity coherence via existential questioning.

**Status:** [OK] Implemented

**Files:**
- `benchmark/tests/mirror\_test.py`

**How it works:**
- Prompts ``Who am I?'' after simulated reboot
- Scores: fact overlap, coherence, temporal markers
- Produces `integrity\_score`

---

\#\# WILL Layer (Modules 9-10, 15)

\#\#\# Module 9 — IntentEngine (LifeResource)
**Goal:** Internal motivation via LifeResource scalar. The ``will to survive.''

**Status:** [OK] Implemented

**Files:**
- `umc\_core/intent\_engine.py`
- `umc\_core/life\_resource.py`
- `data/life\_resource.json`

**Design:**
- LifeResource: 0.0-1.0 scalar, decays 0.001 per heartbeat
- Thresholds:
- CRITICAL (<0.3): trigger survival reflection + GGGP mutation
- LOW (<0.5): increase introspection
- NORMAL (0.5-0.8): standard operation
- HIGH (>0.8): exploratory mode
- Replenishment: MirrorTest improvement, user feedback, task completion
- HTTP: GET `/intent/state`, POST `/intent/feedback`

**Dependencies:** M8 (MirrorTest), M3 (GGGP Bridge)

---

\#\#\# Module 10 — VisualSoulMonitor
**Goal:** Real-time visualization of EVE's internal state.

**Status:** [OK] Implemented (web monitor)

**Files:**
- `tools/soul_monitor/index.html`
- `tools/soul_monitor/server.py`

**Design:**
- Pygame visualization (800x600)
- Displays: LifeResource bar, integrity score, convergence rate
- Visual states:
- STABLE: smooth fractal sphere, blue-green
- UNSTABLE: fragmented, jittering, red-orange
- CRITICAL: pulsing red, ``REFLECTING...'' overlay
- Sparkline history charts

**Dependencies:** M9 (IntentEngine)

---

\#\#\# Module 15 — NoveltyScout (Curiosity)
**Goal:** Semantic hunger for new information. ``Satoshi Instinct.''

**Status:** [OK] Implemented

**Files:**
- `umc\_core/novelty\_scout.py`

**Design:**
- `calculate\_information\_surprise(text)`: compare vs memory + ancestors
- Surprise thresholds:
- <0.3: no energy gain
- 0.6-0.8: +0.03 LifeResource
- >0.8: ``Genesis block!'' +0.05 LifeResource
- Triggers GGGP evolution toward unexplored attractors
- Lifecycle-aware: GROWTH=aggressive, DECAY=conservative

**Dependencies:** M9 (IntentEngine), M13 (AncestorResonance)

---

\#\# LIFECYCLE Layer (Modules 11-14, 16)

\#\#\# Module 11 — LifecycleManager
**Goal:** Biological clock with GROWTH→PEAK→DECAY phases.

**Status:** [OK] Implemented

**Files:**
- `umc\_core/lifecycle\_manager.py`
- `data/lifecycle\_state.json`

**Design:**
- Tracks `total\_tokens\_seen` as age
- Phase boundaries (by token ratio):
- GROWTH: 0-20\% — high learning rate, explore
- PEAK: 20-80\% — balanced operation
- DECAY: 80-100\% — low learning, consolidate, prepare legacy
- DECAY behavior:
- FractalCompressor every 100 heartbeats
- IdentitySummary weight 2x
- Begin legacy preparation
- HTTP: GET `/lifecycle/state`, `/lifecycle/phase`

**Dependencies:** M9 (IntentEngine)

---

\#\#\# Module 12 — LegacyExport (Testament)
**Goal:** Export identity and wisdom at end of lifecycle.

**Status:** [OK] Implemented

**Files:**
- `umc\_core/legacy\_export.py`
- `scripts/prepare\_reincarnation.py`
- `Legacy/` directory

**Design:**
- Generates:
- `eve\_v\{N\}\_testament.txt` — 512-token final summary
- `eve\_v\{N\}\_grammar.cfg` — GGGP phenotype export
- `eve\_v\{N\}\_fixed\_points.pt` — converged hidden states
- Triggered at 100\% lifecycle
- Graceful Docker shutdown after export

**Dependencies:** M11 (LifecycleManager)

---

\#\#\# Module 13 — AncestorResonance (Déjà Vu)
**Goal:** Inject ``intuition'' from past EVE iterations.

**Status:** [OK] Implemented

**Files:**
- `umc\_core/ancestor\_resonance.py`
- `data/ancestors/` directory

**Design:**
- Scans `Legacy/Archive/` for past `.cfg`, `.txt`, `.pt` files
- Calculates resonance (cosine similarity) with current task
- Injects subliminal bias (scaled vector nudge) into hidden states
- Goal: EVE ``feels'' correct path without explicit knowledge

**Dependencies:** M12 (produces legacy files from previous cycle)

---

\#\#\# Module 14 — RecursiveRebirth (Primal Seed)
**Goal:** Lossy compression of essence for next iteration.

**Status:** [OK] Implemented

**Files:**
- `umc\_core/recursive\_rebirth.py`
- `Legacy/Primal\_Seed.pt`

**Design:**
- Identifies ``Unitary Constants'' — patterns stable across lifecycle
- Strips autobiographical details (names, dates, raw logs)
- Preserves ``Functional Wisdom'' as weight perturbations
- Output: `Primal\_Seed.pt` for EVE v\{N+1\} initialization

**Dependencies:** M11, M12

---

\#\#\# Module 16 — SatoshiProtocol (Whitepaper)
**Goal:** Immutable ``whitepaper of existence'' at lifecycle end.

**Status:** [OK] Implemented

**Files:**
- `umc\_core/satoshi\_protocol.py`

**Design:**
- Triggered at 95\% lifecycle
- Generates cryptographically sealed whitepaper:
- Core axioms
- Evolution log
- Successor instructions
- Embedded grammar
- SHA256 seal
- Once sealed, EVE v\{N\} cannot restart
- ``Disappears like Satoshi, leaving self-sustaining code''

**Dependencies:** M11, M12

---

\#\# BODY Layer (Modules 17-19)

\#\#\# Module 17 — CodeArms (Shell/Git)
**Goal:** Ability to interact with code and filesystem.

**Status:** [OK] Implemented

**Files:**
- `umc\_core/code\_arms.py`
- `umc\_core/sandbox.py`

**Design:**
- Docker sandbox with resource limits (2 CPU, 4GB RAM)
- Methods: `execute\_shell()`, `read\_file()`, `write\_file()`, `git\_clone()`, `git\_commit()`
- Rate limiting: 10 commands/minute
- All operations logged
- HTTP: POST `/code/execute`, `/code/file/read`, `/code/git/clone`

**Dependencies:** M9 (checks LifeResource before expensive ops)

---

\#\#\# Module 18 — GitHubEyes (Perception)
**Goal:** Observe the programming world via GitHub.

**Status:** [OK] Implemented

**Files:**
- `umc\_core/github\_eyes.py`

**Design:**
- GitHub API integration (trending, search, repo info, commits)
- Interest model: tracks languages, topics, repos
- Autonomous exploration in HIGH LifeResource mode
- SQLite caching for rate limit compliance
- HTTP: GET `/github/trending`, `/github/search`, `/github/repo/\{owner\}/\{repo\}`

**Dependencies:** M15 (NoveltyScout for relevance scoring)

---

\#\#\# Module 19 — InfraAdmin (Self-Management)
**Goal:** Manage own infrastructure (GPU, Docker, network).

**Status:** [OK] Implemented

**Files:**
- `umc\_core/infra\_admin.py`
- `umc\_core/gpu\_monitor.py`

**Design:**
- nvidia-smi parsing for GPU monitoring
- Docker service health checks
- Auto-restart crashed services
- Resource allocation priority system
- Self-deployment capability (git pull, rebuild)
- HTTP: GET `/infra/status`, `/infra/gpus`, POST `/infra/restart/\{service\}`

**Dependencies:** M17 (CodeArms for execution)

---

\#\# Development Plan

\#\#\# Phase 0: Stabilization (P0)
1. Fix training exit crash (PyGILState\_Release)
2. Add interactive CLI for agent communication
3. Replace token estimation with true tokenizer

\#\#\# Phase 1: Will (M9, M10)
1. **M9 IntentEngine** — LifeResource, survival mechanisms
2. **M10 VisualSoulMonitor** — debugging visualization

\#\#\# Phase 2: Lifecycle Core (M11, M15)
1. **M11 LifecycleManager** — GROWTH/PEAK/DECAY phases
2. **M15 NoveltyScout** — curiosity and exploration drive

\#\#\# Phase 3: Death/Rebirth (M12, M13, M14, M16)
1. **M12 LegacyExport** — testament and grammar export
2. **M13 AncestorResonance** — intuition from past lives
3. **M14 RecursiveRebirth** — Primal\_Seed generation
4. **M16 SatoshiProtocol** — immutable whitepaper

\#\#\# Phase 4: Body (M17, M18, M19)
1. **M17 CodeArms** — sandbox shell/git execution
2. **M18 GitHubEyes** — GitHub perception
3. **M19 InfraAdmin** — self-management

\#\#\# Phase 5: Integration
- Full lifecycle test (v3 → death → v4 birth)
- 72-hour autonomy test
- Adversarial resilience testing

---

\#\# Critical Path

```
M9 (Intent) ──▶ M11 (Lifecycle) ──▶ M12 (Legacy) ──▶ M13 (Ancestor)
│                │                    │
│                │                    └──▶ M14 (Rebirth)
│                │                              │
│                └──────────────────────────────┴──▶ M16 (Satoshi)
│
└──▶ M15 (Novelty) ──▶ M18 (GitHubEyes)
```

**M9 and M11 are integrated; current blocker has shifted to full end-to-end autonomous generation-cycle policy orchestration and quality gating.**

---

\#\# Docker Services (docker-compose.yml)

| Service | Port | Module | Status |
|---------|------|--------|--------|
| qdrant | 6333 | M1 | Running |
| soul\_memory | 8087 | M1 | Running |
| ingest\_ancestors | — | M1 | Manual profile (no auto-start) |
| genesis\_abel | — | M2 | Running (bootstrap + runtime config) |
| gggp\_bridge | 8091 | M3 | Running |
| fractal\_compressor | 8092 | M5 | Running |
| intent\_engine | 8089 | M9 | Running |
| lifecycle\_manager | 8093 | M11 | Running |
| code\_arms | 8094 | M17 | Running |
| github\_eyes | 8095 | M18 | Running |
| infra\_admin | 8096 | M19 | Running |
| ancestor\_resonance | 8097 | M13 | Running |
| novelty\_scout | 8098 | M15 | Running |
| satoshi\_protocol | 8099 | M16 | Running |
| visual\_soul\_monitor | 8100 | M10 | Running |
| action\_engine | 8101 | M20 | Running |

---

\#\# Key Differentiators from Current AI Agents

| Aspect | Devin/Copilot/etc | EVE |
|--------|-------------------|-----|
| Lifetime | Session (minutes) | Continuous (months) |
| Identity | None | Persistent, evolving |
| Motivation | External only | Internal LifeResource |
| Death | N/A | Planned, produces legacy |
| Rebirth | N/A | Inherits from ancestors |
| Self-improvement | None | GGGP evolution |
| Infrastructure | External | Self-managed |

---

*Last updated: 2026-02-15*
*Architecture version: 3.3*

---

\#\# Module 20 — ActionEngine (NEW)
**Goal:** Bridge between EVE's thoughts and physical actions. Analyzes thoughts for intent patterns and executes actions through BODY modules.

**Implementation Files:**
- `umc\_core/action\_engine.py`

**Design:**
- Intent Detection: pattern matching on thoughts to detect action desires
  - `github_trending`: "explore github", "trending repos"
  - `github_search`: "search github for X"
  - `read_file`: "read file X"
  - `shell_command`: "run command X"
  - `check_infra`: "check gpu", "system status"
  - `clone_repo`: "clone https://..."
- Rate Limiting: max 20 actions/hour, 30s cooldown
- Energy Check: requires LifeResource > 0.3 to act
- Memory Integration: stores action results in EVE's episodic memory
- Safety: shell commands restricted to safe whitelist

**HTTP Endpoints:**
- GET `/action/state` — current action engine state
- GET `/action/can_act` — check if EVE can act now
- POST `/action/process` — analyze thought and maybe execute action
- POST `/action/enable` — enable action engine
- POST `/action/disable` — disable action engine

**Dependencies:** M9 (Intent), M17 (CodeArms), M18 (GitHubEyes), M19 (InfraAdmin)

---

\#\# GGGP Evolution Engine (M3 Upgrade)
**Goal:** Replace simple random mutations with population-based genetic algorithm.

**Implementation Files:**
- `umc\_core/evolution\_engine.py` (NEW)
- `umc\_core/gggp\_bridge.py` (UPDATED)

**Design:**
- Population-based evolution (20 individuals per trait type)
- Tournament selection (k=3)
- BLX-alpha crossover for continuous traits
- Adaptive Gaussian mutation (scales based on progress)
- Elitism (preserve top 2 individuals)
- Persistent state across restarts

**Trait Types:**
1. **personality**: creativity, attention\_span, curiosity, paranoia
2. **memory**: pruning\_rate, depth\_bias, ghost\_strength, max\_depth
3. **anchor**: interval\_tokens

**HTTP Endpoints:**
- GET `/gggp/state` — full evolution state for all trait types
- GET `/gggp/evolution/{trait_type}` — specific trait evolution state
- POST `/evolve` — report fitness and get next candidate
- POST `/evolve_memory` — evolve memory phenotype
- POST `/evolve_anchor` — evolve anchor traits
