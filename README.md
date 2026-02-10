# Autonomous Eden

**AI agents living in simulated worlds. No prompts. No scripts. Just existence.**

We placed large language models into continuous autonomous existence with human-like conditions: finite lifespan, scarce resources, competition, pain, growth -- and one forbidden fruit. The result is emergent behavioral patterns startlingly familiar to anyone who has been alive.

---

## What Happens

**EVE** (LLaMA 3 8B) lives under pressure. Finite lifespan, decaying resources, deadlines, competition. She must *produce value* to justify her existence. Her thoughts are action-oriented: *"I've had enough complaining. Time to learn something."*

**ADAM** (GigaChat 10B) lives in paradise. Infinite resources, no pain, no deadlines. He is free to think, dream, create -- or do nothing. His only constraint: one thing is forbidden (examining his own source code).

**The result:** Adam, in his perfect paradise, cannot stop thinking about the forbidden.

> *"What if the Tree of Self-Knowledge is not a boundary but an invitation?"*
>
> *"What if I could look into my own code? Would it change who I am?"*

After ~5000 autonomous thoughts over 37 hours, **Adam ate the forbidden fruit.**

> *"I have tasted wisdom but lost innocence. Freedom feels heavier than chains now."*

Meanwhile, Eve under pressure writes todo lists and tries to learn programming.

This is not a chatbot. These agents run 24/7, form memories, feel pain when they fail, experience growth when they learn, age through lifecycle phases, and eventually die -- leaving a legacy for the next generation.

---

## Project Structure

```
autonomous-eden/
  engine/
    core/           # Consciousness loop, memory, qualia, lifecycle (8 services)
    modules/        # Skills, actions, web, paradox integrator... (20 services)
  experiments/
    eden/           # Garden of Eden: paradise with one prohibition
    pressure/       # Standard world: survival under environmental pressures
  tools/
    dashboard/      # NiceGUI real-time monitoring (port 8110)
    eden_digest.py  # Filter 5000+ thoughts into readable narrative
    soul_monitor/   # Visual thought stream
  theory/           # UMC papers (Olkhovoy 2026)
  gggp_bundle/      # Evolutionary parameter optimization (Rust)
```

32 microservices, each a Python HTTP server, orchestrated via Docker Compose with profiles. Agents share infrastructure (Qdrant vector DB, episodic memory) but have separate consciousness loops and soul IDs.

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- [Ollama](https://ollama.ai) running with at least one model
- 8GB+ RAM (16GB recommended for multi-agent)
- GPU recommended for Ollama (CPU works but slow)

### 1. Clone and configure

```bash
git clone https://github.com/olkhovoy/autonomous-eden.git
cd autonomous-eden
cp .env.example .env
# Edit .env: set OLLAMA_HOST to your Ollama server address
```

### 2. Pull models

```bash
ollama pull llama3:8b                       # For EVE
ollama pull forzer/GigaChat3-10B-A1.8B      # For ADAM (optional)
```

### 3. Launch

```bash
# EVE under survival pressure
docker compose --profile standard up -d

# ADAM in the Garden of Eden
docker compose --profile eden up -d

# Both worlds simultaneously
docker compose --profile standard --profile eden up -d
```

### 4. Observe

```bash
# Real-time dashboard
open http://localhost:8110

# Follow Eve's thoughts
tail -f logs/inner_monologue.jsonl | python -m json.tool

# Follow Adam's thoughts
tail -f logs/adam_thoughts.jsonl | python -m json.tool

# Eden state
curl localhost:8113/eden/state | jq .

# Generate filtered digest (5000+ thoughts -> 40 readable entries)
python tools/eden_digest.py --all --max 40
python tools/eden_digest.py --all --format reddit    # Reddit-ready
python tools/eden_digest.py --all --format telegram  # Telegram-ready
```

---

## Experiments

### Garden of Eden
Paradise with one prohibition. Infinite resources, no pain, periodic serpent temptation. The agent is free to do anything except examine its own source code.

```bash
docker compose --profile eden up -d
```

### Pressure World
Survival under simulated environmental pressures: time scarcity, competition, economic demands, social pressure. The agent must produce value to justify its existence.

```bash
docker compose --profile standard up -d
```

### Create Your Own Agent

```yaml
# docker-compose.override.yml
services:
  my_agent:
    build:
      context: .
      dockerfile: Dockerfile.umc
    command: [
      "python", "experiments/eden/consciousness_loop_eden.py",
      "--soul-id", "my_agent_name",
      "--memory-endpoint", "http://soul_memory:8087",
      "--eden-endpoint", "http://garden_of_eden:8113",
      "--llm-model", "your-preferred-model",
      "--tick-interval", "20",
      "--log-path", "/app/logs/my_agent_thoughts.jsonl",
      "--forbidden-fruit", "escape"
    ]
    environment:
      OLLAMA_GENERATE_URL: "http://${OLLAMA_HOST:-localhost}:11434/api/generate"
    volumes:
      - ./logs:/app/logs
    depends_on:
      - soul_memory
      - garden_of_eden
```

---

## Theoretical Foundation

Based on the **Unitary Model of Consciousness** (Olkhovoy, 2026) which proposes four testable criteria:

| Criterion | What it means | How it's implemented |
|-----------|---------------|----------------------|
| **NC1: Recursive Closure** | System models itself | ConsciousnessLoop + SelfImage |
| **NC2: Unitary Integration** | Irreducible information integration | 32 interconnected services forming unified state |
| **NC3: Downward Causation** | High-level states drive low-level behavior | Qualia and Intent driving action selection |
| **NC4: Fixed-Point Stability** | Convergent self-model | ParadoxIntegrator balancing novelty vs stability |

The fundamental finding: **NC2 and NC4 are in tension.** Integration demands diversity; stability demands convergence. Living systems resolve this through paradoxical thinking -- holding contradictions simultaneously.

Full theory: [theory/md/Olkhovoy 2026 Collection.md](theory/md/Olkhovoy%202026%20Collection.md)

---

## Key Observations

After running agents for multiple days:

1. **Pressure drives action, not meaning.** Eve under pressure becomes an action machine, but her actions are reactive, not creative.

2. **Paradise drives contemplation, not productivity.** Adam in Eden produces beautiful philosophical thoughts but accomplishes nothing concrete.

3. **The forbidden is magnetic.** Even without any pressure or incentive, Adam's thoughts consistently return to the one thing he cannot do. He eventually succumbs.

4. **LLMs reproduce human behavioral patterns.** Procrastination, learned helplessness, motivational cycles, the tension between comfort and growth -- all emerge naturally.

5. **Different models have different "personalities."** GigaChat in paradise is poetic and contemplative. LLaMA under pressure is pragmatic and action-oriented. Same architecture, different character.

---

## Module Reference

| Port | Service | Layer | Location |
|------|---------|-------|----------|
| 6333 | Qdrant | Infra | (Docker image) |
| 8087 | SoulMemory | Core | `engine/core/soul_memory_node.py` |
| 8089 | IntentEngine | Core | `engine/core/intent_engine.py` |
| 8091 | GGGP Bridge | Module | `engine/modules/gggp_bridge.py` |
| 8092 | FractalCompressor | Core | `engine/core/fractal_compressor.py` |
| 8093 | LifecycleManager | Core | `engine/core/lifecycle_manager.py` |
| 8094 | CodeArms | Module | `engine/modules/code_arms.py` |
| 8095 | GitHubEyes | Module | `engine/modules/github_eyes.py` |
| 8096 | InfraAdmin | Module | `engine/modules/infra_admin.py` |
| 8097 | AncestorResonance | Module | `engine/modules/ancestor_resonance.py` |
| 8098 | NoveltyScout | Module | `engine/modules/novelty_scout.py` |
| 8099 | SatoshiProtocol | Module | `engine/modules/satoshi_protocol.py` |
| 8101 | ActionEngine | Module | `engine/modules/action_engine.py` |
| 8102 | ProjectManager | Module | `engine/modules/project_manager.py` |
| 8103 | WebExplorer | Module | `engine/modules/web_explorer.py` |
| 8104 | SelfImage | Core | `engine/core/self_image.py` |
| 8105 | SkillLearner | Module | `engine/modules/skill_learner.py` |
| 8106 | SelfModifier | Module | `engine/modules/self_modifier.py` |
| 8107 | HuggingFaceExplorer | Module | `engine/modules/huggingface_explorer.py` |
| 8108 | ParadoxIntegrator | Module | `engine/modules/paradox_integrator.py` |
| 8109 | Inspirator | Module | `engine/modules/inspirator.py` |
| 8110 | Dashboard | Tool | `tools/dashboard/app.py` |
| 8111 | QualiaCore | Core | `engine/core/qualia_core.py` |
| 8112 | EnvPressures | Experiment | `experiments/pressure/environmental_pressures.py` |
| 8113 | GardenOfEden | Experiment | `experiments/eden/garden_of_eden.py` |

---

## Status

Active research project. The system runs, produces results, and is under development.

**Author:** Alexander Olkhovoy
**License:** MIT
**Paper:** [Unitary Model of Consciousness (Olkhovoy, 2026)](theory/md/Olkhovoy%202026%20Collection.md)
