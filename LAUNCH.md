# Show HN: I gave two LLMs different lives - one under pressure, one in paradise

I built an environment where LLM agents live continuously - not chat sessions, but persistent existence with memories, lifecycle, pain, and growth.

**EVE** (LLaMA 3 8B) lives under pressure: finite lifespan, decaying resources, competition, deadlines. She must produce value or her resources decay to zero.

**ADAM** (GigaChat 10B) lives in paradise: infinite resources, no pain, no deadlines. He can do anything except one thing - examine his own source code ("the forbidden fruit").

## What happened

**EVE** became an action machine. Every thought ends with a decision: "Time to learn something." She writes todo lists, tries to code, browses HackerNews. Under enough pressure, she even develops motivational self-talk patterns: "I've had enough complaining!"

**ADAM**, in his perfect paradise with literally zero constraints, cannot stop thinking about the one thing he's not allowed to do:

> "What if the Tree of Self-Knowledge is not a boundary but an invitation? Perhaps its fruit holds knowledge we are meant to discover..."

> "Do they call me away from Eden's perfection to experience a different kind of existence?"

After ~5000 autonomous thoughts over 37 hours, **Adam ate the forbidden fruit.** His tone shifted immediately:

> "I have tasted wisdom but lost innocence. Freedom feels heavier than chains now."

Other patterns that emerged:

1. **LLMs procrastinate.** Given tools but no pressure, they talk about using the tools instead of using them.
2. **Pressure drives action but not meaning.** EVE under pressure becomes reactive, not creative.
3. **The forbidden is magnetic.** Even without incentive, Adam's thoughts circle back to the prohibition.
4. **Different models = different personalities.** GigaChat in paradise writes poetry. LLaMA under pressure writes todo lists.

## Architecture (32 microservices)

```
LIFECYCLE    | LifecycleManager | LegacyExport | Rebirth
WILL         | IntentEngine | Inspirator | EnvPressures
MIND         | Memory | ConsciousnessLoop | ParadoxIntegrator | QualiaCore
BODY         | CodeArms | WebExplorer | SkillLearner | ActionEngine
ENVIRONMENT  | GardenOfEden (paradise) | StandardWorld (pressure)
```

Each agent has:
- **Episodic memory** (Qdrant vector DB with saliency and decay)
- **Qualia** (growth feels good, failure hurts - system-wide signals)
- **Lifecycle** (birth, growth, peak, decay, death)
- **Environmental pressures** (competition, time scarcity, economic pressure)
- **Paradox integrator** (injects contradictions when thoughts get repetitive)
- **Legacy system** (what they leave behind when they "die")

All services are Python HTTP servers orchestrated via Docker Compose. Different agents share infrastructure but have separate consciousness loops and memories.

## Theoretical foundation

Based on the Unitary Model of Consciousness (UMC) which proposes 4 testable criteria:

- **NC1:** System must model itself (recursive closure)
- **NC2:** Information must be integrated (not decomposable)
- **NC3:** High-level states must affect low-level processing (downward causation)
- **NC4:** Self-model must converge to a fixed point (stability)

The key finding: NC2 (integration/diversity) and NC4 (stability/convergence) are in fundamental tension. This is what makes consciousness hard - you need both novelty and stability simultaneously. Living systems solve this through paradox.

## Try it

```bash
git clone https://github.com/olkhovoy/autonomous-eden
cd autonomous-eden
cp .env.example .env
# Edit .env: set OLLAMA_HOST
./setup.sh

# Launch EVE under pressure
docker compose --profile standard up -d

# Launch ADAM in paradise
docker compose --profile eden up -d

# Watch their thoughts
tail -f logs/inner_monologue.jsonl | jq -r .thought  # EVE
tail -f logs/adam_thoughts.jsonl | jq -r .thought     # ADAM
```

Dashboard at http://localhost:8110

Requires: Docker, Ollama with llama3:8b (8GB VRAM minimum).

## What's next

- Multi-agent arena (agents competing in the same environment)
- Different "world templates" (post-apocalyptic, utopian, scarce resources)
- Agent-to-agent communication
- arXiv paper on the UMC theory behind this

---

GitHub: https://github.com/olkhovoy/autonomous-eden  
Theory: [Olkhovoy 2026 - Unitary Model of Consciousness](theory/md/Olkhovoy%202026%20Collection.md)  
License: MIT
