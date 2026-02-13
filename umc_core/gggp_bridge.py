#!/usr/bin/env python3
"""
GGGP Bridge: REST wrapper that connects Python orchestration to evolutionary optimization.

This module manages EVE's parameter evolution using a proper population-based
genetic algorithm with tournament selection, crossover, and adaptive mutation.
"""

import argparse
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from http.server import BaseHTTPRequestHandler, HTTPServer

from umc_core.evolution_engine import EvolutionEngine, get_engine, EvolutionConfig


@dataclass
class PersonalityVector:
    creativity: float
    attention_span: float
    curiosity: float
    paranoia: float

    def clamp(self):
        self.creativity = min(max(self.creativity, 0.0), 1.0)
        self.attention_span = min(max(self.attention_span, 0.0), 1.0)
        self.curiosity = min(max(self.curiosity, 0.0), 1.0)
        self.paranoia = min(max(self.paranoia, 0.0), 1.0)
    
    def to_genome(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_genome(cls, genome: Dict[str, float]) -> "PersonalityVector":
        return cls(**genome)


@dataclass
class MemoryPhenotype:
    pruning_rate: float = 0.2
    depth_bias: float = 0.3
    ghost_strength: float = 0.2
    max_depth: int = 4

    def clamp(self):
        self.pruning_rate = min(max(self.pruning_rate, 0.0), 1.0)
        self.depth_bias = min(max(self.depth_bias, 0.0), 1.0)
        self.ghost_strength = min(max(self.ghost_strength, 0.0), 1.0)
        self.max_depth = max(1, min(int(self.max_depth), 8))
    
    def to_genome(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_genome(cls, genome: Dict[str, float]) -> "MemoryPhenotype":
        return cls(**genome)


@dataclass
class AnchorTraits:
    interval_tokens: int = 1000

    def clamp(self):
        self.interval_tokens = max(128, min(int(self.interval_tokens), 4096))
    
    def to_genome(self) -> Dict[str, float]:
        return {"interval_tokens": float(self.interval_tokens)}
    
    @classmethod
    def from_genome(cls, genome: Dict[str, float]) -> "AnchorTraits":
        return cls(interval_tokens=int(genome.get("interval_tokens", 1000)))


class GGGPBridge:
    """
    Bridge between EVE's systems and the evolutionary optimization engine.
    
    Uses population-based evolution with:
    - Tournament selection
    - BLX-alpha crossover
    - Adaptive Gaussian mutation
    - Elitism
    """
    
    def __init__(self, gggp_bin: str = "", workdir: str = "/home/user/mcs/gggp_bundle/rust"):
        self.gggp_bin = gggp_bin
        self.workdir = workdir
        
        # Initialize evolution engines for each trait type
        self.personality_engine = get_engine("personality")
        self.memory_engine = get_engine("memory")
        self.anchor_engine = get_engine("anchor")
        
        # Track statistics
        self.total_evolves = 0
        self.new_generations = 0

    def evolve(self, traits: PersonalityVector, score: float) -> PersonalityVector:
        """
        Evolve personality traits using genetic algorithm.
        Reports fitness and returns next candidate genome.
        """
        self.total_evolves += 1
        
        # Report fitness for current traits
        genome = traits.to_genome()
        self.personality_engine.report_fitness(genome, score)
        
        # Get next candidate (may trigger new generation)
        best, is_new_gen = self.personality_engine.evolve_step()
        if is_new_gen:
            self.new_generations += 1
        
        # Return candidate for evaluation
        candidate = self.personality_engine.get_candidate_for_evaluation()
        return PersonalityVector.from_genome(candidate.genome)

    def evolve_memory(self, traits: MemoryPhenotype, score: float) -> MemoryPhenotype:
        """Evolve memory phenotype traits."""
        self.total_evolves += 1
        
        genome = traits.to_genome()
        self.memory_engine.report_fitness(genome, score)
        
        best, is_new_gen = self.memory_engine.evolve_step()
        if is_new_gen:
            self.new_generations += 1
        
        candidate = self.memory_engine.get_candidate_for_evaluation()
        return MemoryPhenotype.from_genome(candidate.genome)

    def evolve_anchor(self, traits: AnchorTraits, score: float) -> AnchorTraits:
        """Evolve anchor traits."""
        self.total_evolves += 1
        
        genome = traits.to_genome()
        self.anchor_engine.report_fitness(genome, score)
        
        best, is_new_gen = self.anchor_engine.evolve_step()
        if is_new_gen:
            self.new_generations += 1
        
        candidate = self.anchor_engine.get_candidate_for_evaluation()
        return AnchorTraits.from_genome(candidate.genome)
    
    def get_evolution_state(self, trait_type: str = "personality") -> Dict[str, Any]:
        """Get state of evolution for a trait type."""
        engine = {
            "personality": self.personality_engine,
            "memory": self.memory_engine,
            "anchor": self.anchor_engine,
        }.get(trait_type, self.personality_engine)
        
        return engine.get_state()


class GGGPHandler(BaseHTTPRequestHandler):
    bridge: GGGPBridge = None  # set externally

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})

        if self.path == "/evolve":
            traits = data.get("traits")
            if not traits:
                return self._json(400, {"error": "traits required"})
            score = float(data.get("score", 0.0))
            vec = PersonalityVector(**traits)
            vec.clamp()
            out = self.bridge.evolve(vec, score=score)
            evo_state = self.bridge.get_evolution_state("personality")
            return self._json(200, {
                "traits": asdict(out),
                "evolution": {
                    "generation": evo_state["generation"],
                    "best_fitness": evo_state["best_fitness"],
                }
            })

        if self.path == "/evolve_memory":
            traits = data.get("traits")
            if not traits:
                return self._json(400, {"error": "traits required"})
            score = float(data.get("score", 0.0))
            vec = MemoryPhenotype(**traits)
            vec.clamp()
            out = self.bridge.evolve_memory(vec, score=score)
            evo_state = self.bridge.get_evolution_state("memory")
            return self._json(200, {
                "traits": asdict(out),
                "evolution": {
                    "generation": evo_state["generation"],
                    "best_fitness": evo_state["best_fitness"],
                }
            })

        if self.path == "/evolve_anchor":
            traits = data.get("traits")
            if not traits:
                return self._json(400, {"error": "traits required"})
            score = float(data.get("score", 0.0))
            vec = AnchorTraits(**traits)
            vec.clamp()
            out = self.bridge.evolve_anchor(vec, score=score)
            evo_state = self.bridge.get_evolution_state("anchor")
            return self._json(200, {
                "traits": asdict(out),
                "evolution": {
                    "generation": evo_state["generation"],
                    "best_fitness": evo_state["best_fitness"],
                }
            })

        return self._json(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/gggp/state":
            return self._json(200, {
                "status": "ok",
                "modes": ["personality", "memory", "anchor"],
                "total_evolves": self.bridge.total_evolves,
                "new_generations": self.bridge.new_generations,
                "personality": self.bridge.get_evolution_state("personality"),
                "memory": self.bridge.get_evolution_state("memory"),
                "anchor": self.bridge.get_evolution_state("anchor"),
            })
        
        if self.path.startswith("/gggp/evolution/"):
            trait_type = self.path.split("/")[-1]
            if trait_type in ["personality", "memory", "anchor"]:
                return self._json(200, self.bridge.get_evolution_state(trait_type))
            return self._json(400, {"error": "unknown trait type"})
        
        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="GGGP Bridge service")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--gggp-bin", type=str, default="", help="Path to Rust GGGP binary (optional)")
    parser.add_argument("--workdir", type=str, default="/home/user/mcs/gggp_bundle/rust")
    args = parser.parse_args()

    bridge = GGGPBridge(gggp_bin=args.gggp_bin, workdir=args.workdir)
    GGGPHandler.bridge = bridge
    server = HTTPServer(("0.0.0.0", args.port), GGGPHandler)
    print(f"GGGP Bridge listening on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
