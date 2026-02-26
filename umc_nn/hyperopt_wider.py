import sys
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from umc_core.evolution_engine import EvolutionEngine, EvolutionConfig
from core import UMC_Cell

from train import UMC_Network, generate_synthetic_data

class UMCHyperoptEngineWider(EvolutionEngine):
    """Subclass to define UMC_Cell specific traits for wider search."""
    TRAIT_BOUNDS = {
        "umc_nn": {
            "hidden_dim": (128, 512),        # int - expanded upwards!
            "num_nodes": (4, 32),            # int - pushing up!
            "k_contractive": (0.85, 0.9999), # float - keep it highly contractive
            "learning_rate": (0.0005, 0.05), # float 
        },
        "personality": {} # Dummy to satisfy super().__init__ fallback
    }

def evaluate_fitness(genome, device='cpu'):
    """
    Trains UMC_Network briefly with the given hyperparameters 
    and returns a fitness score.
    """
    input_dim = 32
    output_dim = 32
    seq_len = 15
    
    hidden_dim = int(genome['hidden_dim'])
    num_nodes = int(genome['num_nodes'])
    k_contractive = genome['k_contractive']
    learning_rate = genome['learning_rate']
    
    model = UMC_Network(input_dim, num_nodes, hidden_dim, output_dim, k_contractive).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Fast evaluation to fit 150 generations in a shorter time
    epochs = 40
    batch_size = 256
    
    # Generate dataset
    X_train, Y_train = generate_synthetic_data(1000, seq_len, input_dim)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            pred, _ = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
    # Evaluation (NC2)
    model.eval()
    with torch.no_grad():
        X_test, Y_test = generate_synthetic_data(200, seq_len, input_dim)
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        pred, _ = model(X_test)
        nc2_loss = criterion(pred, Y_test).item()
        
        # Stability Test (NC4)
        h_umc = model.cell.init_hidden(1, device)
        zero_input = torch.zeros(1, input_dim).to(device)
        
        # Burn-in
        for _ in range(15):
            h_umc, _ = model.cell(zero_input, h_umc)
            
        # Measure oscillation
        macros = []
        for _ in range(15):
            h_umc, macro = model.cell(zero_input, h_umc)
            macros.append(macro.norm().item())
            
        oscillation = max(macros) - min(macros)
        
    # We heavily penalize oscillation to enforce strict fixed point
    total_error = nc2_loss + (oscillation * 5.0) 
    
    import math
    if math.isnan(total_error) or math.isinf(total_error):
        return 0.0001
        
    fitness = 1.0 / (total_error + 1e-6)
    return fitness

def run_evolution():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = EvolutionConfig(
        population_size=15, 
        elite_count=2, 
        tournament_size=3,
        mutation_rate=0.4,
        max_generations=150
    )
    
    engine = UMCHyperoptEngineWider(trait_type="umc_nn", config=config, state_path="./evolution_umc_nn_wider.json")
    
    print("Starting Evolution for UMC Hyperparameters (Wider Capacity)...")
    print(f"Population: {config.population_size}, Max Generations: {config.max_generations}")
    
    for gen in range(config.max_generations):
        print(f"\n--- Generation {engine.generation} ---")
        
        eval_count = 0
        for ind in engine.population:
            if ind.fitness == 0.0:
                print(f"Evaluating genome: hidden={int(ind.genome['hidden_dim'])}, nodes={int(ind.genome['num_nodes'])}, k={ind.genome['k_contractive']:.3f}, lr={ind.genome['learning_rate']:.4f}", flush=True)
                fitness = evaluate_fitness(ind.genome, device)
                ind.fitness = fitness
                engine.report_fitness(ind.genome, fitness)
                eval_count += 1
                
        print(f"Evaluated {eval_count} new individuals.", flush=True)
        best = engine.get_current_best()
        print(f"Best Fitness: {best.fitness:.4f} -> Genome: {best.genome}", flush=True)
        
        # Save every generation
        engine._save_state()
        
        best_ind, new_gen_started = engine.evolve_step()

    print("\nEvolution Complete!")
    best = engine.get_current_best()
    print("Optimal Hyperparameters Found:")
    print(json.dumps(best.genome, indent=2))
    print(f"Fitness Score: {best.fitness:.4f}")

if __name__ == "__main__":
    run_evolution()