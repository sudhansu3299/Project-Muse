"""
Comprehensive Strategy Plotter

Generates plots and combined tables for multiple exploration strategies:
- Greedy
- Hungarian BFS (hungarian_utility_bfs)
- Hungarian A* (hungarian_utility_astar)
- PPO-A (ppo_50k_a)
- PPO-B (ppo_100k_b)

Plots:
- Coverage vs Steps per seed
- Sensing Redundancy per seed
- Movement Efficiency per seed
- Combined CSV table
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns


class ComprehensiveStrategyPlotter:
    """
    Generates comprehensive plots and tables for multiple exploration strategies.
    """

    def __init__(
        self,
        classical_path: str = "results/classical_benchmark/raw_results.csv",
        ppo_path: str = "results/comparison/raw_comparison.csv",
        output_dir: str = "results/comprehensive_plots",
    ):
        """
        Initialize the plotter with paths to input files.

        Args:
            classical_path: Path to classical strategies results CSV
            ppo_path: Path to PPO comparison results CSV
            output_dir: Directory to save output files
        """
        self.classical_path = classical_path
        self.ppo_path = ppo_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.classical_data = None
        self.ppo_data = None
        self.combined_data = None

        # Strategy name mapping
        self.strategy_mapping = {
            "greedy": "Greedy",
            "hungarian_utility_bfs": "Hungarian BFS",
            "hungarian_utility_astar": "Hungarian A*",
            "ppo_50k_a": "PPO-A",
            "ppo_100k_b": "PPO-B",
        }

        # Color palette
        self.colors = {
            "Greedy": "#e74c3c",
            "Hungarian BFS": "#3498db",
            "Hungarian A*": "#9b59b6",
            "PPO-A": "#2ecc71",
            "PPO-B": "#f39c12",
        }

    def load_data(self):
        """Load data from CSV files."""
        # Load classical strategies data
        self.classical_data = pd.read_csv(self.classical_path)
        # Filter out random strategy and empty rows
        self.classical_data = self.classical_data[
            (self.classical_data["strategy"] != "random") &
            (self.classical_data["time_to_90"].notna())
        ]
        print(f"Loaded classical data: {len(self.classical_data)} records")

        # Load PPO comparison data
        self.ppo_data = pd.read_csv(self.ppo_path)
        print(f"Loaded PPO data: {len(self.ppo_data)} records")

    def combine_data(self) -> pd.DataFrame:
        """
        Combine data from classical and PPO strategies.

        Returns:
            Combined DataFrame with all strategies
        """
        if self.classical_data is None or self.ppo_data is None:
            self.load_data()

        # Rename columns to match
        classical_df = self.classical_data.copy()
        classical_df = classical_df.rename(columns={"time_to_90": "time_to_target"})

        # Combine dataframes
        combined_df = pd.concat([classical_df, self.ppo_data], ignore_index=True)

        # Filter to only the 5 strategies we want
        target_strategies = ["greedy", "hungarian_utility_bfs", "hungarian_utility_astar", "ppo_50k_a", "ppo_100k_b"]
        combined_df = combined_df[combined_df["strategy"].isin(target_strategies)]

        # Map strategy names
        combined_df["strategy_display"] = combined_df["strategy"].map(self.strategy_mapping)

        self.combined_data = combined_df
        print(f"Combined data: {len(self.combined_data)} records")

        return self.combined_data

    def generate_combined_table(self, output_filename: str = "combined_strategy_table.csv"):
        """
        Generate and save the combined strategy table as CSV.

        Args:
            output_filename: Name of the output CSV file
        """
        if self.combined_data is None:
            self.combine_data()

        # Compute aggregated metrics per strategy
        agg_dict = {
            "time_to_target": ["mean", "std"],
            "distance": ["mean", "std"],
            "sensing_redundancy": ["mean", "std"],
            "visit_overlap": ["mean", "std"],
            "movement_efficiency": ["mean", "std"],
        }

        summary = self.combined_data.groupby("strategy_display").agg(agg_dict).reset_index()

        # Flatten column names
        summary.columns = ["Strategy"] + [f"{col}_{stat}" for col in agg_dict.keys() for stat in ["mean", "std"]]

        # Reorder columns
        column_order = [
            "Strategy",
            "time_to_target_mean", "time_to_target_std",
            "distance_mean", "distance_std",
            "sensing_redundancy_mean", "sensing_redundancy_std",
            "visit_overlap_mean", "visit_overlap_std",
            "movement_efficiency_mean", "movement_efficiency_std",
        ]
        summary = summary[column_order]

        output_path = self.output_dir / output_filename
        summary.to_csv(output_path, index=False)
        print(f"Combined table saved to {output_path}")

        return summary

    def plot_coverage_vs_steps_per_seed(self, output_filename: str = "coverage_vs_steps_per_seed.png"):
        """
        Plot coverage vs steps (time_to_target) for each seed.

        Args:
            output_filename: Name of the output plot file
        """
        if self.combined_data is None:
            self.combine_data()

        seeds = sorted(self.combined_data["seed"].unique())
        strategies = [self.strategy_mapping[s] for s in ["greedy", "hungarian_utility_bfs", "hungarian_utility_astar", "ppo_50k_a", "ppo_100k_b"]]

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle('Coverage vs Steps per Seed', fontsize=16, fontweight='bold', y=0.995)

        axes = axes.flatten()

        for idx, seed in enumerate(seeds[:12]):  # Plot first 12 seeds
            ax = axes[idx]
            seed_data = self.combined_data[self.combined_data["seed"] == seed]

            for strategy in strategies:
                strategy_data = seed_data[seed_data["strategy_display"] == strategy]
                if len(strategy_data) > 0:
                    # Since we only have T90 (time to reach 90%), we'll plot that as a point
                    # with coverage at 90%
                    ax.scatter(strategy_data["time_to_target"], [90] * len(strategy_data),
                             color=self.colors[strategy], label=strategy, s=100, alpha=0.7)

            ax.set_xlabel('Steps to 90% Coverage', fontweight='bold')
            ax.set_ylabel('Coverage (%)', fontweight='bold')
            ax.set_title(f'Seed {seed}', fontweight='bold')
            ax.set_ylim(88, 92)
            ax.grid(True, alpha=0.3)

            # Add legend only for first subplot
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)

        # Remove empty subplots
        for idx in range(len(seeds), 12):
            axes[idx].axis('off')

        plt.tight_layout()

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Coverage vs steps plot saved to {output_path}")
        plt.close()

    def plot_sensing_redundancy_per_seed(self, output_filename: str = "sensing_redundancy_per_seed.png"):
        """
        Plot sensing redundancy for each strategy per seed (line graph).

        Args:
            output_filename: Name of the output plot file
        """
        if self.combined_data is None:
            self.combine_data()

        fig, ax = plt.subplots(figsize=(14, 8))

        strategies = [self.strategy_mapping[s] for s in ["greedy", "hungarian_utility_bfs", "hungarian_utility_astar", "ppo_50k_a", "ppo_100k_b"]]
        seeds = sorted(self.combined_data["seed"].unique())

        for strategy in strategies:
            strategy_data = self.combined_data[self.combined_data["strategy_display"] == strategy]
            redundancy_values = []

            for seed in seeds:
                seed_data = strategy_data[strategy_data["seed"] == seed]
                if len(seed_data) > 0:
                    redundancy_values.append(seed_data["sensing_redundancy"].values[0])
                else:
                    redundancy_values.append(np.nan)

            ax.plot(seeds, redundancy_values, marker='o', label=strategy,
                   color=self.colors[strategy], linewidth=2, markersize=8, alpha=0.8)

        ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sensing Redundancy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Sensing Redundancy per Seed', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Sensing redundancy plot saved to {output_path}")
        plt.close()

    def plot_movement_efficiency_per_seed(self, output_filename: str = "movement_efficiency_per_seed.png"):
        """
        Plot movement efficiency for each strategy per seed (line graph).

        Args:
            output_filename: Name of the output plot file
        """
        if self.combined_data is None:
            self.combine_data()

        fig, ax = plt.subplots(figsize=(14, 8))

        strategies = [self.strategy_mapping[s] for s in ["greedy", "hungarian_utility_bfs", "hungarian_utility_astar", "ppo_50k_a", "ppo_100k_b"]]
        seeds = sorted(self.combined_data["seed"].unique())

        for strategy in strategies:
            strategy_data = self.combined_data[self.combined_data["strategy_display"] == strategy]
            efficiency_values = []

            for seed in seeds:
                seed_data = strategy_data[strategy_data["seed"] == seed]
                if len(seed_data) > 0:
                    efficiency_values.append(seed_data["movement_efficiency"].values[0])
                else:
                    efficiency_values.append(np.nan)

            ax.plot(seeds, efficiency_values, marker='o', label=strategy,
                   color=self.colors[strategy], linewidth=2, markersize=8, alpha=0.8)

        ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
        ax.set_ylabel('Movement Efficiency', fontsize=12, fontweight='bold')
        ax.set_title('Movement Efficiency per Seed', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Movement efficiency plot saved to {output_path}")
        plt.close()

    def plot_all_metrics_comparison(self, output_filename: str = "all_metrics_comparison.png"):
        """
        Create a multi-panel plot comparing all metrics across strategies.

        Args:
            output_filename: Name of the output plot file
        """
        if self.combined_data is None:
            self.combine_data()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Comprehensive Strategy Comparison', fontsize=16, fontweight='bold', y=0.995)

        strategies = [self.strategy_mapping[s] for s in ["greedy", "hungarian_utility_bfs", "hungarian_utility_astar", "ppo_50k_a", "ppo_100k_b"]]

        # Compute mean values per strategy
        mean_values = {}
        for strategy in strategies:
            strategy_data = self.combined_data[self.combined_data["strategy_display"] == strategy]
            mean_values[strategy] = {
                "T90": strategy_data["time_to_target"].mean(),
                "Distance": strategy_data["distance"].mean(),
                "Sensing Redundancy": strategy_data["sensing_redundancy"].mean(),
                "Visit Overlap": strategy_data["visit_overlap"].mean(),
                "Movement Efficiency": strategy_data["movement_efficiency"].mean(),
            }

        # T90 (lower is better)
        ax1 = axes[0, 0]
        t90_values = [mean_values[s]["T90"] for s in strategies]
        bars1 = ax1.bar(strategies, t90_values, color=[self.colors[s] for s in strategies], alpha=0.8)
        ax1.set_ylabel('Steps', fontweight='bold')
        ax1.set_title('T90 ↓', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars1, t90_values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.0f}', ha='center', va='bottom', fontsize=9)

        # Distance (lower is better)
        ax2 = axes[0, 1]
        distance_values = [mean_values[s]["Distance"] for s in strategies]
        bars2 = ax2.bar(strategies, distance_values, color=[self.colors[s] for s in strategies], alpha=0.8)
        ax2.set_ylabel('Distance', fontweight='bold')
        ax2.set_title('Distance ↓', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars2, distance_values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.0f}', ha='center', va='bottom', fontsize=9)

        # Sensing Redundancy (lower is better)
        ax3 = axes[0, 2]
        redundancy_values = [mean_values[s]["Sensing Redundancy"] for s in strategies]
        bars3 = ax3.bar(strategies, redundancy_values, color=[self.colors[s] for s in strategies], alpha=0.8)
        ax3.set_ylabel('Redundancy (%)', fontweight='bold')
        ax3.set_title('Sensing Redundancy ↓', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars3, redundancy_values):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # Visit Overlap (lower is better)
        ax4 = axes[1, 0]
        overlap_values = [mean_values[s]["Visit Overlap"] for s in strategies]
        bars4 = ax4.bar(strategies, overlap_values, color=[self.colors[s] for s in strategies], alpha=0.8)
        ax4.set_ylabel('Overlap (%)', fontweight='bold')
        ax4.set_title('Visit Overlap ↓', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars4, overlap_values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # Movement Efficiency (higher is better)
        ax5 = axes[1, 1]
        efficiency_values = [mean_values[s]["Movement Efficiency"] for s in strategies]
        bars5 = ax5.bar(strategies, efficiency_values, color=[self.colors[s] for s in strategies], alpha=0.8)
        ax5.set_ylabel('Efficiency', fontweight='bold')
        ax5.set_title('Movement Efficiency ↑', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars5, efficiency_values):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        # Remove empty subplot
        axes[1, 2].axis('off')

        plt.tight_layout()

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"All metrics comparison plot saved to {output_path}")
        plt.close()

    def generate_all(self):
        """Generate all outputs: table and plots."""
        print("Generating comprehensive strategy plots and tables...")
        self.load_data()
        self.combine_data()
        self.generate_combined_table()
        self.plot_coverage_vs_steps_per_seed()
        self.plot_sensing_redundancy_per_seed()
        self.plot_movement_efficiency_per_seed()
        self.plot_all_metrics_comparison()
        print("\nAll outputs generated successfully!")


# Example usage
if __name__ == "__main__":
    plotter = ComprehensiveStrategyPlotter(
        classical_path="results/classical_benchmark/raw_results.csv",
        ppo_path="results/comparison/raw_comparison.csv",
        output_dir="results/comprehensive_plots",
    )

    # Generate all outputs
    plotter.generate_all()
