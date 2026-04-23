#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np

# Try to import torch, fallback to dummy objects if not found
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. Using dummy objects for network state.")
    class nn:
        class Module: pass
        def Linear(in_f, out_f): return None
    class torch:
        @staticmethod
        def save(obj, f): pass

# Adjust path to import umc_core if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "gggp_bundle"))
import semiotic_hypercube as umc_core

# --- Global State for Logging ---
GLOBAL_STATE = {
    "evals": 0,
    "current_gen": 0,
    "pop_size": 30,
    "best_score": -1e9,
    "best_vector": None,
    "best_policy": None,
}

# --- Scaffolds for Janus Agent & Environment ---

class MultiTimeframeVecEnv:
    def __init__(self, num_envs=4, max_steps=1000):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.current_step = 0
        self.prices = np.ones(num_envs) * 100.0
        self.positions = np.zeros(num_envs) # 0=Flat, 1=Long, -1=Short
        
    def reset(self):
        self.current_step = 0
        self.prices = np.ones(self.num_envs) * 100.0
        self.positions = np.zeros(self.num_envs)
        return np.random.randn(self.num_envs, 10)
        
    def step(self, actions):
        self.current_step += 1
        
        # Simulate price movement
        changes = np.random.randn(self.num_envs) * 0.5
        self.prices += changes
        
        prev_positions = self.positions.copy()
        
        # Determine new positions
        # 0: Flat, 1: Long, 2: Short
        new_positions = np.zeros(self.num_envs)
        new_positions[actions == 1] = 1
        new_positions[actions == 2] = -1
        self.positions = new_positions
        
        # PnL from price movement
        step_pnl = changes * prev_positions
        
        # --- Friction: Transaction Costs ---
        FEE_RATE = 0.0006
        delta_positions = np.abs(self.positions - prev_positions)
        transaction_costs = delta_positions * self.prices * FEE_RATE
        
        step_pnl -= transaction_costs
        
        obs = np.random.randn(self.num_envs, 10)
        rewards = step_pnl # Use real PnL as base reward, will be modified later
        
        dones = np.array([self.current_step >= self.max_steps] * self.num_envs)
        
        infos = []
        for i in range(self.num_envs):
            infos.append({
                "cortisol": np.random.uniform(0, 1),
                "transaction_cost": transaction_costs[i],
                "price": self.prices[i],
                "position": self.positions[i]
            })
            
        return obs, rewards, dones, infos

if HAS_TORCH:
    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)
        def forward(self, x):
            return self.fc(x)
else:
    class PolicyNet:
        def parameters(self): return []
        def state_dict(self): return {"dummy_weights": True}

class PatchTST_Self:
    def __init__(self):
        self.frozen = True
        self.policy = PolicyNet()

class JanusAgent:
    def __init__(self, limbic_system, neuro_params):
        self.limbic_system = limbic_system
        self.stress_penalty = neuro_params[0]
        self.pain_scale = neuro_params[1]
        self.cortisol_decay = neuro_params[2]
        self.time_decay = neuro_params[3]
        self.entropy_coef = neuro_params[4]
        
    def act(self, obs):
        # 0: Flat, 1: Long, 2: Short
        # Mock: randomly choose an action.
        return np.random.randint(0, 3, size=(obs.shape[0],))
        
    def update(self):
        # Mock PPO Update
        if HAS_TORCH:
            # We don't actually do backprop on mock data, just simulate the weight update call
            pass

# --- Evaluation Function ---

def evaluate_neurochemistry(vector: np.ndarray) -> float:
    """
    Fitness function for the neurochemical parameters.
    The vector contains:
    [0] stress_penalty
    [1] pain_scale
    [2] cortisol_decay
    [3] time_decay
    [4] entropy_coef
    """
    if vector.shape[0] != 5:
        return -1000.0

    # 1. Penalties must be positive (Agent cannot enjoy pain or time)
    stress_penalty = max(0.0, vector[0]) 
    
    # Force a minimum time_decay (Hunger) so sitting Flat is always painful
    # It must be higher than the background noise of the market
    time_decay = max(0.0005, vector[3])

    # 2. Decay must be a fraction (0 to 1) to prevent exponential explosion
    cortisol_decay = max(0.0, min(0.999, vector[2]))

    # 3. Scales must be strictly positive
    pain_scale = max(0.1, vector[1])

    # 4. Entropy must be bounded (e.g., 0.01 to 0.5) to force exploration
    entropy_coef = max(0.01, min(0.5, vector[4]))

    clamped_vector = np.array([
        stress_penalty,
        pain_scale,
        cortisol_decay,
        time_decay,
        entropy_coef
    ])

    env = MultiTimeframeVecEnv(num_envs=1, max_steps=1000)
    limbic = PatchTST_Self()
    agent = JanusAgent(limbic, clamped_vector)

    obs = env.reset()
    pnl_sum = 0.0
    
    actions_taken = []
    cortisol_levels = []

    # Run abbreviated PPO training / evaluation loop
    # Simulate 25 updates of 40 steps each = 1000 steps total
    num_updates = 25
    steps_per_update = env.max_steps // num_updates
    
    for _ in range(num_updates):
        for _ in range(steps_per_update):
            actions = agent.act(obs)
            next_obs, rewards, dones, infos = env.step(actions)
            
            # apply neurochemical modifiers to rewards and friction
            cortisol = infos[0]["cortisol"]
            
            # Action Shift Penalty
            action_shift_penalty = 0.005 if (len(actions_taken) > 0 and actions[0] != actions_taken[-1]) else 0.0
            
            # Base reward is the step_pnl (which already has transaction_cost subtracted)
            modified_reward = rewards[0] - (cortisol * agent.stress_penalty) - agent.time_decay - action_shift_penalty
            
            pnl_sum += modified_reward
            actions_taken.append(actions[0])
            cortisol_levels.append(cortisol)
            
            obs = next_obs
            if dones[0]:
                obs = env.reset()
                
        # Simulate PPO Update Phase
        agent.update()

    # Calculate Engagement Quality
    actions_taken = np.array(actions_taken)
    cortisol_levels = np.array(cortisol_levels)
    
    # High cortisol zone defined as cortisol > 0.7
    high_cortisol_mask = cortisol_levels > 0.7
    if np.any(high_cortisol_mask):
        flat_in_danger = np.mean(actions_taken[high_cortisol_mask] == 0)
    else:
        flat_in_danger = 0.5
        
    engagement_quality = flat_in_danger
    
    # Calculate penalty for out-of-bound parameters
    penalty = 0.0
    
    if vector[0] < 0.0:
        penalty += abs(vector[0]) * 100.0
    if vector[3] < 0.0005:
        penalty += (0.0005 - vector[3]) * 100.0
        
    if vector[2] < 0.0:
        penalty += abs(vector[2]) * 100.0
    elif vector[2] > 0.999:
        penalty += (vector[2] - 0.999) * 100.0
        
    if vector[1] < 0.1:
        penalty += (0.1 - vector[1]) * 100.0
        
    if vector[4] < 0.01:
        penalty += (0.01 - vector[4]) * 100.0
    elif vector[4] > 0.5:
        penalty += (vector[4] - 0.5) * 100.0

    # Fitness Score
    score = (pnl_sum * engagement_quality) - penalty
    
    # Global tracking and logging
    GLOBAL_STATE["evals"] += 1
    gen = GLOBAL_STATE["evals"] // GLOBAL_STATE["pop_size"]
    
    if score > GLOBAL_STATE["best_score"]:
        GLOBAL_STATE["best_score"] = score
        GLOBAL_STATE["best_vector"] = clamped_vector.copy()
        if HAS_TORCH:
            GLOBAL_STATE["best_policy"] = agent.limbic_system.policy.state_dict()
        else:
            GLOBAL_STATE["best_policy"] = {"dummy": True}

    if gen > GLOBAL_STATE["current_gen"]:
        # Log generation summary
        print(f"--- Generation {GLOBAL_STATE['current_gen']} Complete --- Best Score: {GLOBAL_STATE['best_score']:.4f}")
        
        # Log to CSV
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "evolution_history.csv")
        
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["generation", "stress_penalty", "pain_scale", "cortisol_decay", "time_decay", "entropy_coef", "score"])
            
            if GLOBAL_STATE["best_vector"] is not None:
                v = GLOBAL_STATE["best_vector"]
                writer.writerow([GLOBAL_STATE["current_gen"], v[0], v[1], v[2], v[3], v[4], GLOBAL_STATE["best_score"]])
        
        # Save PyTorch checkpoints every 10 generations
        # To test checkpoint saving efficiently we do it for gen > 0 and % 10
        if GLOBAL_STATE["current_gen"] > 0 and GLOBAL_STATE["current_gen"] % 10 == 0:
            chk_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
            os.makedirs(chk_dir, exist_ok=True)
            chk_path = os.path.join(chk_dir, f"best_policy_gen_{GLOBAL_STATE['current_gen']}.pt")
            if HAS_TORCH:
                torch.save(GLOBAL_STATE["best_policy"], chk_path)
            else:
                with open(chk_path, "w") as f:
                    f.write("dummy pt file")
            print(f"Saved policy checkpoint to {chk_path}")

        # Reset best score for next generation
        GLOBAL_STATE["best_score"] = -1e9
        GLOBAL_STATE["current_gen"] = gen

    return float(score)

# --- Execution Loop ---

def main():
    print("Initializing SemanticHypercube with neuro_grammar.cfg...")
    grammar_path = os.path.join(os.path.dirname(__file__), "..", "gggp_bundle", "neuro_grammar.cfg")
    if not os.path.exists(grammar_path):
        print(f"Error: Grammar file not found at {grammar_path}")
        sys.exit(1)
        
    hypercube = umc_core.SemanticHypercube(grammar_path)
    
    # Grand Evolutionary Run Parameters
    generations = 100
    population_size = 30
    dim = 5
    
    GLOBAL_STATE["pop_size"] = population_size
    
    print("\n================================================")
    print("      THE GRAND EVOLUTIONARY RUN INITIATED      ")
    print("================================================")
    print(f"Dimensions:      {dim} (Neurochemistry Vector)")
    print(f"Generations:     {generations}")
    print(f"Population Size: {population_size}")
    print("Evaluation:      1000-step PPO loop (25 updates/eval)")
    print("Constraints:     Hunger > 0.0005, Entropy > 0.01")
    print("Logging:         logs/evolution_history.csv")
    print("Checkpoints:     checkpoints/best_policy_gen_*.pt (every 10 gen)")
    print("================================================\n")
    
    # evolve_with_fitness calls Rust which orchestrates the CMA-ES loop and calls our python eval function
    optimized_weights, out_array, final_fitness = hypercube.evolve_with_fitness(
        dim, 
        generations, 
        population_size,
        evaluate_neurochemistry
    )
    
    print("\n--- Evolution Complete ---")
    print(f"Final Fitness: {final_fitness:.4f}")
    
    # Clamp final output for accurate reporting
    final_stress_penalty = max(0.0, out_array[0])
    final_pain_scale = max(0.1, out_array[1])
    final_cortisol_decay = max(0.0, min(0.999, out_array[2]))
    final_time_decay = max(0.0005, out_array[3])
    final_entropy_coef = max(0.01, min(0.5, out_array[4]))
    
    print(f"Optimized Parameters (Raw / Clamped):")
    print(f"  stress_penalty: {out_array[0]:>8.4f} -> {final_stress_penalty:.4f}")
    print(f"  pain_scale:     {out_array[1]:>8.4f} -> {final_pain_scale:.4f}")
    print(f"  cortisol_decay: {out_array[2]:>8.4f} -> {final_cortisol_decay:.4f}")
    print(f"  time_decay:     {out_array[3]:>8.4f} -> {final_time_decay:.4f}")
    print(f"  entropy_coef:   {out_array[4]:>8.4f} -> {final_entropy_coef:.4f}")

if __name__ == "__main__":
    main()
