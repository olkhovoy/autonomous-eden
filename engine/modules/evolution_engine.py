#!/usr/bin/env python3
"""
EvolutionEngine: Real evolutionary algorithm for EVE's parameter optimization.

Implements:
- Population-based evolution
- Tournament selection
- Crossover (blend + uniform)
- Adaptive mutation
- Elitism
- Fitness history tracking
"""

import json
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy


@dataclass
class Individual:
    """A single individual in the population."""
    genome: Dict[str, float]
    fitness: float = 0.0
    age: int = 0
    created_at: float = field(default_factory=time.time)
    parent_ids: List[str] = field(default_factory=list)
    
    @property
    def id(self) -> str:
        return f"{int(self.created_at * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "genome": self.genome,
            "fitness": self.fitness,
            "age": self.age,
            "created_at": self.created_at,
            "parent_ids": self.parent_ids,
        }


@dataclass
class EvolutionConfig:
    """Evolution hyperparameters."""
    population_size: int = 20
    elite_count: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.3
    mutation_scale: float = 0.1
    crossover_rate: float = 0.7
    max_generations: int = 1000
    fitness_stagnation_limit: int = 50
    
    # Adaptive mutation
    adaptive_mutation: bool = True
    min_mutation_scale: float = 0.01
    max_mutation_scale: float = 0.3


class EvolutionEngine:
    """
    Full evolutionary algorithm for optimizing EVE's parameters.
    
    Supports multiple trait types:
    - personality: creativity, attention_span, curiosity, paranoia
    - memory: pruning_rate, depth_bias, ghost_strength, max_depth
    - anchor: interval_tokens
    """
    
    TRAIT_BOUNDS = {
        "personality": {
            "creativity": (0.0, 1.0),
            "attention_span": (0.0, 1.0),
            "curiosity": (0.0, 1.0),
            "paranoia": (0.0, 1.0),
        },
        "memory": {
            "pruning_rate": (0.0, 1.0),
            "depth_bias": (0.0, 1.0),
            "ghost_strength": (0.0, 1.0),
            "max_depth": (1, 8),
        },
        "anchor": {
            "interval_tokens": (128, 4096),
        }
    }
    
    def __init__(
        self,
        trait_type: str = "personality",
        config: Optional[EvolutionConfig] = None,
        state_path: str = "data/evolution_state.json",
    ):
        self.trait_type = trait_type
        self.config = config or EvolutionConfig()
        self.state_path = state_path
        self.bounds = self.TRAIT_BOUNDS.get(trait_type, self.TRAIT_BOUNDS["personality"])
        
        # State
        self.population: List[Individual] = []
        self.generation: int = 0
        self.best_fitness: float = 0.0
        self.stagnation_count: int = 0
        self.fitness_history: List[Dict[str, float]] = []
        self.current_mutation_scale: float = self.config.mutation_scale
        
        # Load or initialize
        self._load_state()
        if not self.population:
            self._initialize_population()
    
    def _load_state(self):
        """Load evolution state from disk."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                if data.get("trait_type") == self.trait_type:
                    self.generation = data.get("generation", 0)
                    self.best_fitness = data.get("best_fitness", 0.0)
                    self.stagnation_count = data.get("stagnation_count", 0)
                    self.fitness_history = data.get("fitness_history", [])
                    self.current_mutation_scale = data.get("mutation_scale", self.config.mutation_scale)
                    
                    self.population = []
                    for ind_data in data.get("population", []):
                        ind = Individual(
                            genome=ind_data["genome"],
                            fitness=ind_data.get("fitness", 0.0),
                            age=ind_data.get("age", 0),
                            created_at=ind_data.get("created_at", time.time()),
                            parent_ids=ind_data.get("parent_ids", []),
                        )
                        self.population.append(ind)
            except Exception as e:
                print(f"[WARN] Failed to load evolution state: {e}")
    
    def _save_state(self):
        """Save evolution state to disk."""
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        data = {
            "trait_type": self.trait_type,
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "stagnation_count": self.stagnation_count,
            "fitness_history": self.fitness_history[-100:],  # Keep last 100
            "mutation_scale": self.current_mutation_scale,
            "population": [ind.to_dict() for ind in self.population],
        }
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.config.population_size):
            genome = {}
            for trait, (lo, hi) in self.bounds.items():
                if isinstance(lo, int) and isinstance(hi, int):
                    genome[trait] = random.randint(lo, hi)
                else:
                    genome[trait] = random.uniform(lo, hi)
            self.population.append(Individual(genome=genome))
        self._save_state()
    
    def _clamp(self, genome: Dict[str, float]) -> Dict[str, float]:
        """Clamp genome values to valid bounds."""
        clamped = {}
        for trait, value in genome.items():
            lo, hi = self.bounds.get(trait, (0.0, 1.0))
            if isinstance(lo, int) and isinstance(hi, int):
                clamped[trait] = int(max(lo, min(hi, round(value))))
            else:
                clamped[trait] = max(lo, min(hi, value))
        return clamped
    
    def _tournament_select(self) -> Individual:
        """Select individual via tournament."""
        candidates = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Dict[str, float]:
        """Blend crossover between two parents."""
        child_genome = {}
        for trait in self.bounds.keys():
            v1 = parent1.genome.get(trait, 0.5)
            v2 = parent2.genome.get(trait, 0.5)
            
            # BLX-alpha crossover
            alpha = 0.5
            lo = min(v1, v2) - alpha * abs(v1 - v2)
            hi = max(v1, v2) + alpha * abs(v1 - v2)
            
            if isinstance(self.bounds[trait][0], int):
                child_genome[trait] = random.randint(int(lo), int(hi))
            else:
                child_genome[trait] = random.uniform(lo, hi)
        
        return self._clamp(child_genome)
    
    def _mutate(self, genome: Dict[str, float]) -> Dict[str, float]:
        """Apply mutation to genome."""
        mutated = dict(genome)
        for trait in self.bounds.keys():
            if random.random() < self.config.mutation_rate:
                lo, hi = self.bounds[trait]
                current = mutated.get(trait, (lo + hi) / 2)
                
                # Gaussian mutation
                scale = self.current_mutation_scale * (hi - lo)
                delta = random.gauss(0, scale)
                
                if isinstance(lo, int) and isinstance(hi, int):
                    mutated[trait] = int(round(current + delta))
                else:
                    mutated[trait] = current + delta
        
        return self._clamp(mutated)
    
    def _adapt_mutation(self, improved: bool):
        """Adapt mutation scale based on progress."""
        if not self.config.adaptive_mutation:
            return
        
        if improved:
            # Decrease mutation on improvement (fine-tuning)
            self.current_mutation_scale *= 0.95
        else:
            # Increase mutation on stagnation (exploration)
            self.current_mutation_scale *= 1.05
        
        self.current_mutation_scale = max(
            self.config.min_mutation_scale,
            min(self.config.max_mutation_scale, self.current_mutation_scale)
        )
    
    def report_fitness(self, genome: Dict[str, float], fitness: float) -> bool:
        """
        Report fitness score for a genome. Updates population.
        Returns True if this is a new best.
        """
        # Find matching individual
        for ind in self.population:
            if ind.genome == genome:
                ind.fitness = fitness
                break
        else:
            # New individual from external source
            self.population.append(Individual(genome=genome, fitness=fitness))
        
        # Track best
        new_best = fitness > self.best_fitness
        if new_best:
            self.best_fitness = fitness
            self.stagnation_count = 0
            self._adapt_mutation(improved=True)
        else:
            self.stagnation_count += 1
            self._adapt_mutation(improved=False)
        
        self.fitness_history.append({
            "generation": self.generation,
            "fitness": fitness,
            "best": self.best_fitness,
            "timestamp": time.time(),
        })
        
        self._save_state()
        return new_best
    
    def evolve_step(self) -> Tuple[Individual, bool]:
        """
        Perform one evolution step. Returns (best_individual, is_new_generation).
        """
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Check if we need a new generation
        all_evaluated = all(ind.fitness > 0 for ind in self.population)
        
        if all_evaluated:
            # Create new generation
            new_population = []
            
            # Elitism: keep best individuals
            for i in range(min(self.config.elite_count, len(self.population))):
                elite = deepcopy(self.population[i])
                elite.age += 1
                new_population.append(elite)
            
            # Fill rest with offspring
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_select()
                
                if random.random() < self.config.crossover_rate:
                    parent2 = self._tournament_select()
                    child_genome = self._crossover(parent1, parent2)
                    parent_ids = [parent1.id, parent2.id]
                else:
                    child_genome = dict(parent1.genome)
                    parent_ids = [parent1.id]
                
                # Mutate
                child_genome = self._mutate(child_genome)
                
                child = Individual(
                    genome=child_genome,
                    fitness=0.0,  # Will be evaluated
                    parent_ids=parent_ids,
                )
                new_population.append(child)
            
            self.population = new_population
            self.generation += 1
            self._save_state()
            
            return self.population[0], True
        
        return self.population[0], False
    
    def get_current_best(self) -> Individual:
        """Get the current best individual."""
        if not self.population:
            self._initialize_population()
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0]
    
    def get_candidate_for_evaluation(self) -> Individual:
        """Get the next individual that needs fitness evaluation."""
        for ind in self.population:
            if ind.fitness == 0.0:
                return ind
        # All evaluated, return best
        return self.get_current_best()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current evolution state."""
        best = self.get_current_best()
        return {
            "trait_type": self.trait_type,
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_fitness,
            "stagnation_count": self.stagnation_count,
            "mutation_scale": self.current_mutation_scale,
            "best_genome": best.genome,
            "avg_fitness": sum(i.fitness for i in self.population) / len(self.population) if self.population else 0,
            "recent_history": self.fitness_history[-10:],
        }


# Singleton engines per trait type
_engines: Dict[str, EvolutionEngine] = {}


def get_engine(trait_type: str = "personality") -> EvolutionEngine:
    """Get or create evolution engine for trait type."""
    if trait_type not in _engines:
        state_path = f"data/evolution_{trait_type}.json"
        _engines[trait_type] = EvolutionEngine(trait_type=trait_type, state_path=state_path)
    return _engines[trait_type]
