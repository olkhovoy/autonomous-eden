# Consciousness Engine

The engine provides the core infrastructure for running autonomous AI agents. Each module is a standalone Python HTTP server that communicates with others via REST.

## Core (`engine/core/`)

Services every agent needs to run:

| Service | Port | Purpose |
|---------|------|---------|
| **SoulMemory** | 8087 | Episodic memory with vector search, saliency scoring, and temporal decay |
| **ConsciousnessLoop** | - | Central tick: query memories, compose prompt, generate thought, store result |
| **IntentEngine** | 8089 | Internal motivation via LifeResource: drives, decay, reward/punishment |
| **LifecycleManager** | 8093 | Birth/Growth/Peak/Decline phases with configurable lifespan |
| **QualiaCore** | 8111 | Fundamental experiential signals: Growth (novelty) and Pain (loss) |
| **FractalCompressor** | 8092 | Compress old memories into fractal summaries to manage memory growth |
| **SelfImage** | 8104 | Visual self-representation via Stable Diffusion, updated by internal state |
| **LifeResource** | - | Resource pool backing IntentEngine drives |

## Modules (`engine/modules/`)

Optional extensions that add capabilities:

| Service | Port | Purpose |
|---------|------|---------|
| **ActionEngine** | 8101 | Pattern-match thoughts to concrete actions (create file, browse web, etc.) |
| **ParadoxIntegrator** | 8108 | Detect NC4 dominance (repetitive thoughts), inject paradoxes for NC2 balance |
| **Inspirator** | 8109 | Break "stuck" patterns with role models, challenges, and direct motivation |
| **SkillLearner** | 8105 | Programming challenges with difficulty progression |
| **ProjectManager** | 8102 | Long-term goal tracking and task management |
| **WebExplorer** | 8103 | Internet browsing and information discovery |
| **CodeArms** | 8094 | Code writing and execution |
| **GitHubEyes** | 8095 | GitHub repository monitoring |
| **HuggingFaceExplorer** | 8107 | HuggingFace model discovery |
| **NoveltyScout** | 8098 | Track novelty vs repetition in thought stream |
| **GGGPBridge** | 8091 | Evolutionary parameter optimization |
| **SatoshiProtocol** | 8099 | Value creation and legacy preservation |
| **AncestorResonance** | 8097 | Inherited patterns from previous agent generations |
| **RecursiveRebirth** | 8105 | Agent death and rebirth with legacy transfer |
| **EvolutionEngine** | - | Parameter evolution across generations |
| **SelfModifier** | 8106 | Self-modification of own configuration |
| **InfraAdmin** | 8096 | Infrastructure health monitoring |

## How It Works

1. **ConsciousnessLoop** runs a tick every N seconds
2. Each tick: query SoulMemory for recent context, fetch QualiaCore state, compose a prompt
3. Send prompt to Ollama, get a thought back
4. ActionEngine pattern-matches the thought for actionable intent
5. Store the thought in SoulMemory with saliency score
6. QualiaCore updates Growth/Pain signals based on outcomes
7. ParadoxIntegrator watches for repetition, injects paradoxes when needed
8. LifecycleManager advances the agent through life phases

## Building a Custom Agent

The minimal agent needs only:
- **SoulMemory** (episodic storage)
- **ConsciousnessLoop** (the tick)
- **Ollama** (LLM backend)

Everything else is optional and adds capabilities incrementally.

See the main [README](../README.md) for a docker-compose example of creating a custom agent.
