### **ROADMAP v2.0: The Unitary "Janus" Agent**
**(Scaling to Production with Foundation Models & Fluid Identity)**

## Objective
Validate the Unitary Hypothesis: agents that predict their own internal state outperform price-only agents.
**New Core Hypothesis:** Survival in a chaotic simulation requires **Fluid Identity** (switching between Bull/Bear personas) and **Contextual Depth** (multi-scale perception via Foundation Models).

---

## Phase 1: The Arena (High-Performance Environment)
**Goal:** A GPU-native simulation capable of 1M steps/minute.

*   **Week 1: Vectorized Environment (GPU)**
    *   Implement `VecEnv` API in PyTorch.
    *   State transitions (Balance, PnL, Exposure) calculated entirely on GPU tensors.
    *   Support for "continuously pouring" data (streaming).

*   **Week 2: Execution Realism**
    *   Add Maker/Taker fees.
    *   Add Volatility-based slippage.
    *   Add Funding Rates (critical for perpetuals).

---

## Phase 2: The Eyes (Perception Upgrade)
**Goal:** Replace raw OHLCV with semantic embeddings from SOTA Time-Series models.

*   **Week 3: Foundation Model Integration**
    *   **Selection:** `amazon/chronos-t5-small` (or `google/timesfm-1.0-200m` depending on VRAM fit).
    *   **Implementation:** Create a `FeatureExtractor` service.
        *   Input: Raw OHLCV window.
        *   Output: A dense embedding vector (The "Gist" of the market).
    *   **Optimization:** Run inference in batch mode or pre-compute embeddings for the entire historical dataset to save training time.

*   **Week 4: Fractal Vision Pipeline**
    *   Modify Data Loader to yield **Three Tensors** per step:
        1.  `Tactical`: Last 128 minutes (Raw + Chronos Embedding).
        2.  `Strategic`: Last 128 *hours* (Resampled + Chronos Embedding).
        3.  `Cyclical`: Last 128 *days* (Resampled + Chronos Embedding).
    *   **Fusion Layer:** A simple MLP to concatenate and compress these three views into a single `World_State`.

---

## Phase 3: The Brain (The Janus Architecture)
**Goal:** Implement the "Fluid Identity" mechanism.

*   **Week 5: The "Twin Souls" (Bull & Bear)**
    *   Design two separate, smaller `PsychoModules` inside the agent:
        *   **The Bull:** Dopamine = Positive PnL on Longs. Cortisol = Price Drop.
        *   **The Bear:** Dopamine = Positive PnL on Shorts. Cortisol = Price Rise.
    *   The model must predict the *future state* of BOTH sub-personalities.

*   **Week 6: The Meta-Observer (Policy Network)**
    *   Input: `World_State` + `Bull_State` + `Bear_State`.
    *   **The Switch:** The policy does not output "Buy/Sell" directly. It outputs **"Identity Weight"**.
        *   `Action = Weight_Bull * Action_Bull + Weight_Bear * Action_Bear`.
    *   **The Insight:** The agent learns to "become the Bear" when the Bull is suffering (high Cortisol), effectively automating the psychological switch you discovered manually.

---

## Phase 4: Training & Evolution
**Goal:** Evolutionary selection of the best psycho-dynamics (GGGP).

*   **Week 7: PPO with Auxiliary Tasks**
    *   **Loss Function:**
        *   `L_Trading`: Maximize PnL (PPO).
        *   `L_Empathy`: Minimize prediction error of Bull/Bear internal states (MSE).
        *   `L_Coherence`: Penalty for high volatility in "Identity Weight" (don't flip-flop every second).

*   **Week 8: The Evolutionary Tournament (GGGP)**
    *   Instead of training one model, spawn a population of 50 agents with different **Hyperparameters**:
        *   `Pain_Sensitivity`, `Dopamine_Decay`, `Time_Horizon_Bias`.
    *   Run a tournament on historical data.
    *   Mutate the "DNA" of the top 10 survivors.

---

## Technical Stack (Updated)

*   **Core:** PyTorch 2.0+ (`torch.compile` enabled).
*   **Foundation Model:** HuggingFace Transformers (`Chronos` or `TimesFM`).
*   **Data:** Polars (for fast resampling of multi-scale windows).
*   **Environment:** Custom Torch-based VecEnv.
*   **Evolution:** DEAP or custom genetic loop.

## Success Metrics

1.  **Survival:** Agent survives 2022 crash and 2024 volatility without liquidation.
2.  **Identity Fluidity:** Agent switches to "Bear Persona" *before* the crash accelerates (predicts its own Bull-pain).
3.  **Profit:** Sharpe Ratio > 2.0 on out-of-sample data.
