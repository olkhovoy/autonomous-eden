"""Latent Wormhole demo: evolve a GGGP tree toward a target vector and plot
the trajectory inside the "semantic mush" of a random cloud.

Usage:
    cd gggp_bundle/
    cargo run --bin gen_grammar                     # writes test_grammar.cfg
    maturin develop --release --features python \
        --manifest-path rust/Cargo.toml             # installs semiotic_hypercube
    python scripts/demo_wormhole.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import plotly.graph_objects as go

import semiotic_hypercube


def main() -> None:
    print("Initializing Semiotic Hypercube...")
    grammar_path = "test_grammar.cfg"
    if not os.path.exists(grammar_path):
        print(
            f"Error: {grammar_path} not found. "
            "Run `cargo run --bin gen_grammar` from gggp_bundle/rust/ first."
        )
        sys.exit(1)

    try:
        hypercube = semiotic_hypercube.SemioticHypercube(grammar_path)
    except Exception as exc:
        print(f"Failed to load grammar: {exc}")
        sys.exit(1)

    # 1. Mathematical Setup
    # Target 1 (Risk) = [1.0, 0.0, 0.0]
    # Target 2 (Safety) = [0.0, 1.0, 0.0]
    # The Concept Chimera (Target) = [0.707, 0.707, 0.707]
    target_chimera = np.array([0.707, 0.707, 0.707], dtype=np.float64)

    print(f"Target Concept Chimera: {target_chimera}")

    # 2. The Data Collection Loop
    trajectory = []
    
    # We want to trace the "best" vector per generation
    generations = 100
    for gen in range(generations):
        chrom, vec, fitness = hypercube.step_evolution(target_chimera)
        trajectory.append(vec)
        if gen % 10 == 0:
            print(f"Gen {gen}: Fitness = {fitness:.4f}, Vec = {vec}")
            
    print(f"Evolution complete. Final fitness: {fitness:.4f}")

    trajectory = np.array(trajectory)
    
    # 3. The Visualization
    print("Generating Plotly visualization...")
    
    # Grey Cloud: The "Semantic Mush"
    np.random.seed(42)
    cloud_points = np.random.normal(loc=0.5, scale=0.15, size=(500, 3))
    cloud_points[:, 2] = np.random.normal(loc=0.0, scale=0.1, size=500) # Keep it mostly flat in Z
    
    fig = go.Figure()

    # The Semantic Mush
    fig.add_trace(go.Scatter3d(
        x=cloud_points[:, 0],
        y=cloud_points[:, 1],
        z=cloud_points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='rgba(150, 150, 150, 0.2)',
            opacity=0.2
        ),
        name='LLM Zone (Semantic Mush)'
    ))

    # Point A: Risk
    fig.add_trace(go.Scatter3d(
        x=[1.0], y=[0.0], z=[0.0],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            line=dict(color='pink', width=2),
            symbol='diamond'
        ),
        name='V_Risk'
    ))

    # Point B: Safety
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[1.0], z=[0.0],
        mode='markers',
        marker=dict(
            size=10,
            color='cyan',
            line=dict(color='white', width=2),
            symbol='diamond'
        ),
        name='V_Safety'
    ))

    # Target: The Chimera
    fig.add_trace(go.Scatter3d(
        x=[0.707], y=[0.707], z=[0.707],
        mode='markers',
        marker=dict(
            size=12,
            color='purple',
            line=dict(color='magenta', width=3),
            symbol='cross'
        ),
        name='Target: Concept Chimera'
    ))

    # The Hypercube Trajectory
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        z=trajectory[:, 2],
        mode='lines+markers',
        marker=dict(
            size=4,
            color='gold',
        ),
        line=dict(
            color='gold',
            width=5
        ),
        name='Evolutionary Trajectory'
    ))

    fig.update_layout(
        title="Latent Wormhole: Solving the Concept Chimera",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            bgcolor='rgb(10,10,10)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    output_html = "wormhole_demo.html"
    fig.write_html(output_html)
    print(f"Visualization saved to {output_html}")

if __name__ == '__main__':
    main()
