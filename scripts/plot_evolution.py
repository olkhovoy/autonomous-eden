#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "evolution_history.csv")
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return
        
    df = pd.read_csv(log_path)
    
    if df.empty:
        print("Log file is empty.")
        return

    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 16), sharex=True)
    fig.suptitle("Janus Agent: Neurochemical Evolution Trajectory", fontsize=16)

    # Plot Sniper Score
    axes[0].plot(df["generation"], df["score"], color="gold", marker="o", linestyle="-", label="Sniper Score (Fitness)")
    axes[0].set_ylabel("Fitness Score")
    axes[0].set_title("Evolution of Sniper Score")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Plot Stress Penalty
    axes[1].plot(df["generation"], df["stress_penalty"], color="red", marker="s", linestyle="-", label="Stress Penalty")
    axes[1].set_ylabel("Stress Penalty")
    axes[1].set_title("Evolution of Stress Penalty (Cortisol Cost)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Plot Pain Scale
    axes[2].plot(df["generation"], df["pain_scale"], color="darkorange", marker="^", linestyle="-", label="Pain Scale")
    axes[2].set_ylabel("Pain Scale")
    axes[2].set_title("Evolution of Pain Scale (Liquidation Sensitivity)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    # Plot Cortisol Decay & Entropy
    axes[3].plot(df["generation"], df["cortisol_decay"], color="blue", marker="v", linestyle="-", label="Cortisol Decay")
    axes[3].plot(df["generation"], df["entropy_coef"], color="green", marker="x", linestyle="--", label="Entropy Coef")
    axes[3].set_xlabel("Generation")
    axes[3].set_ylabel("Value")
    axes[3].set_title("Evolution of Cortisol Decay & Entropy")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), "..", "logs", "evolution_trajectory.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
