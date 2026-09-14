"""
Comparison Table Generator

Generates formatted comparison tables and plots from existing CSV results.
Compares Fixed Weights, PPO-A, and PPO-B strategies across key metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class ComparisonTableGenerator:
    """
    Generates comparison tables and plots from experimental results.
    """

    def __init__(
        self,
        fixed_weights_path: str = "results/baseline/fixed_weight_results.csv",
        raw_comparison_path: str = "results/comparison/raw_comparison.csv",
        output_dir: str = "results/comparison",
    ):
        """
        Initialize the generator with paths to input files.

        Args:
            fixed_weights_path: Path to fixed weights results CSV
            raw_comparison_path: Path to PPO comparison results CSV
            output_dir: Directory to save output files
        """
        self.fixed_weights_path = fixed_weights_path
        self.raw_comparison_path = raw_comparison_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fixed_data = None
        self.ppo_data = None
        self.aggregated_results = None

    def load_data(self):
        """Load data from CSV files."""
        # Load fixed weights data
        self.fixed_data = pd.read_csv(self.fixed_weights_path)
        print(f"Loaded fixed weights data: {len(self.fixed_data)} records")

        # Load PPO comparison data
        self.ppo_data = pd.read_csv(self.raw_comparison_path)
        print(f"Loaded PPO comparison data: {len(self.ppo_data)} records")

    def compute_aggregated_metrics(self) -> pd.DataFrame:
        """
        Compute aggregated metrics for each strategy.

        Returns:
            DataFrame with strategy names as index and metrics as columns
        """
        if self.fixed_data is None or self.ppo_data is None:
            self.load_data()

        results = []

        # Process Fixed Weights data
        fixed_metrics = {
            "Strategy": "Fixed Weights",
            "T90": self.fixed_data["steps"].mean(),
            "Distance": self.fixed_data["total_distance"].mean(),
            "Sensing Redundancy": self.fixed_data["final_redundancy"].mean(),
            "Visit Overlap": 0.0,  # Not available in fixed weights data
            "Movement Efficiency": 0.0,  # Not available in fixed weights data
        }
        results.append(fixed_metrics)

        # Process PPO-A data (ppo_50k_a)
        ppo_a_data = self.ppo_data[self.ppo_data["strategy"] == "ppo_50k_a"]
        if len(ppo_a_data) > 0:
            ppo_a_metrics = {
                "Strategy": "PPO-A",
                "T90": ppo_a_data["time_to_target"].mean(),
                "Distance": ppo_a_data["distance"].mean(),
                "Sensing Redundancy": ppo_a_data["sensing_redundancy"].mean(),
                "Visit Overlap": ppo_a_data["visit_overlap"].mean(),
                "Movement Efficiency": ppo_a_data["movement_efficiency"].mean(),
            }
            results.append(ppo_a_metrics)

        # Process PPO-B data (ppo_100k_b)
        ppo_b_data = self.ppo_data[self.ppo_data["strategy"] == "ppo_100k_b"]
        if len(ppo_b_data) > 0:
            ppo_b_metrics = {
                "Strategy": "PPO-B",
                "T90": ppo_b_data["time_to_target"].mean(),
                "Distance": ppo_b_data["distance"].mean(),
                "Sensing Redundancy": ppo_b_data["sensing_redundancy"].mean(),
                "Visit Overlap": ppo_b_data["visit_overlap"].mean(),
                "Movement Efficiency": ppo_b_data["movement_efficiency"].mean(),
            }
            results.append(ppo_b_metrics)

        self.aggregated_results = pd.DataFrame(results)
        return self.aggregated_results

    def generate_comparison_table(self, output_filename: str = "comparison_table.csv"):
        """
        Generate and save the comparison table as CSV.

        Args:
            output_filename: Name of the output CSV file
        """
        if self.aggregated_results is None:
            self.compute_aggregated_metrics()

        output_path = self.output_dir / output_filename
        self.aggregated_results.to_csv(output_path, index=False)
        print(f"Comparison table saved to {output_path}")

        return self.aggregated_results

    def print_formatted_table(self):
        """Print the comparison table in a formatted manner."""
        if self.aggregated_results is None:
            self.compute_aggregated_metrics()

        print("\n" + "=" * 100)
        print("STRATEGY COMPARISON TABLE")
        print("=" * 100)
        print(f"{'Strategy':<20} {'T90 ↓':<15} {'Distance ↓':<15} {'Sensing Redundancy ↓':<20} {'Visit Overlap ↓':<15} {'Movement Efficiency ↑':<20}")
        print("-" * 100)

        for _, row in self.aggregated_results.iterrows():
            print(
                f"**{row['Strategy']}**".ljust(18) +
                f"{row['T90']:.1f}".ljust(15) +
                f"{row['Distance']:.1f}".ljust(15) +
                f"{row['Sensing Redundancy']:.2f}%".ljust(20) +
                f"{row['Visit Overlap']:.2f}%".ljust(15) +
                f"{row['Movement Efficiency']:.4f}".ljust(20)
            )

        print("=" * 100)

    def plot_t90_comparison(self, output_filename: str = "t90_comparison.png"):
        """
        Create a bar plot comparing T90 across strategies.

        Args:
            output_filename: Name of the output plot file
        """
        if self.aggregated_results is None:
            self.compute_aggregated_metrics()

        fig, ax = plt.subplots(figsize=(10, 6))

        strategies = self.aggregated_results["Strategy"].values
        t90_values = self.aggregated_results["T90"].values

        # Create bar plot
        bars = ax.bar(strategies, t90_values, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)

        # Customize the plot
        ax.set_ylabel('T90 (Steps to 90% Coverage)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax.set_title('Time to 90% Coverage (T90) Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bar, value in zip(bars, t90_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add arrow indicating lower is better
        ax.annotate('↓ Lower is better', xy=(0.5, 0.95), xycoords='axes fraction',
                   ha='center', fontsize=10, style='italic', color='#666')

        plt.tight_layout()

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"T90 comparison plot saved to {output_path}")
        plt.close()

    def plot_all_metrics(self, output_filename: str = "all_metrics_comparison.png"):
        """
        Create a multi-panel plot comparing all metrics across strategies.

        Args:
            output_filename: Name of the output plot file
        """
        if self.aggregated_results is None:
            self.compute_aggregated_metrics()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Strategy Comparison Across All Metrics', fontsize=16, fontweight='bold', y=0.995)

        strategies = self.aggregated_results["Strategy"].values
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        # T90 (lower is better)
        ax1 = axes[0, 0]
        bars1 = ax1.bar(strategies, self.aggregated_results["T90"], color=colors, alpha=0.8)
        ax1.set_ylabel('Steps', fontweight='bold')
        ax1.set_title('T90 ↓', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars1, self.aggregated_results["T90"]):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)

        # Distance (lower is better)
        ax2 = axes[0, 1]
        bars2 = ax2.bar(strategies, self.aggregated_results["Distance"], color=colors, alpha=0.8)
        ax2.set_ylabel('Distance', fontweight='bold')
        ax2.set_title('Distance ↓', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars2, self.aggregated_results["Distance"]):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)

        # Sensing Redundancy (lower is better)
        ax3 = axes[0, 2]
        bars3 = ax3.bar(strategies, self.aggregated_results["Sensing Redundancy"], color=colors, alpha=0.8)
        ax3.set_ylabel('Redundancy (%)', fontweight='bold')
        ax3.set_title('Sensing Redundancy ↓', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars3, self.aggregated_results["Sensing Redundancy"]):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # Visit Overlap (lower is better)
        ax4 = axes[1, 0]
        bars4 = ax4.bar(strategies, self.aggregated_results["Visit Overlap"], color=colors, alpha=0.8)
        ax4.set_ylabel('Overlap (%)', fontweight='bold')
        ax4.set_title('Visit Overlap ↓', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars4, self.aggregated_results["Visit Overlap"]):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # Movement Efficiency (higher is better)
        ax5 = axes[1, 1]
        bars5 = ax5.bar(strategies, self.aggregated_results["Movement Efficiency"], color=colors, alpha=0.8)
        ax5.set_ylabel('Efficiency', fontweight='bold')
        ax5.set_title('Movement Efficiency ↑', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars5, self.aggregated_results["Movement Efficiency"]):
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
        """Generate all outputs: table, T90 plot, and all metrics plot."""
        print("Generating comparison outputs...")
        self.load_data()
        self.compute_aggregated_metrics()
        self.print_formatted_table()
        self.generate_comparison_table()
        self.plot_t90_comparison()
        self.plot_all_metrics()
        print("\nAll outputs generated successfully!")


# Example usage
if __name__ == "__main__":
    generator = ComparisonTableGenerator(
        fixed_weights_path="results/baseline/fixed_weight_results.csv",
        raw_comparison_path="results/comparison/raw_comparison.csv",
        output_dir="results/comparison",
    )

    # Generate all outputs
    generator.generate_all()
