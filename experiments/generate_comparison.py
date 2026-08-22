import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

RESULTS_DIR = "results"
BASELINE_COMPARISON_DIR = os.path.join(RESULTS_DIR, "baseline_comparison")


def load_all_data():
    """
    Load all CSV files from strategy-specific directories.
    Returns a concatenated DataFrame with all experiment data.
    """
    csv_files = glob.glob(
        os.path.join(RESULTS_DIR, "*", "*.csv")
    )
    
    if not csv_files:
        print("No CSV files found in results/ directory.")
        return None
    
    frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)
    
    data = pd.concat(frames, ignore_index=True)
    return data


def calculate_final_metrics(data):
    """
    Calculate final metrics for each strategy across all runs.
    Returns a DataFrame with summary statistics.
    """
    final_metrics = []
    
    for (strategy, run_id), group in data.groupby(["strategy", "run_id"]):
        # Get the last row for this run
        last_row = group.iloc[-1]
        
        steps = last_row["timestep"]
        distance = last_row["total_distance"]
        coverage = last_row["coverage"]
        sensing_redundancy = last_row["sensing_redundancy"]
        visit_overlap = last_row["visit_overlap"]
        
        # Calculate average movement efficiency over the run
        avg_movement_efficiency = group["movement_efficiency"].mean()
        
        # Calculate average pairwise distance over the run
        avg_pairwise_distance = group["mean_pairwise_distance"].mean()
        
        # Calculate total nodes expanded (if available)
        if "nodes_expanded" in group.columns:
            total_nodes_expanded = group["nodes_expanded"].max()
        else:
            total_nodes_expanded = 0
        
        final_metrics.append({
            "strategy": strategy,
            "run_id": run_id,
            "map_seed": last_row["map_seed"],
            "steps": steps,
            "distance": distance,
            "coverage": coverage,
            "sensing_redundancy": sensing_redundancy,
            "visit_overlap": visit_overlap,
            "movement_efficiency": avg_movement_efficiency,
            "mean_pairwise_distance": avg_pairwise_distance,
            "nodes_expanded": total_nodes_expanded
        })
    
    return pd.DataFrame(final_metrics)


def generate_comparison_table(metrics_df):
    """
    Generate a comparison table matching the format from the image.
    Columns: Method | Steps ↓ | Distance ↓ | Sensing Redundancy ↓ | Visit Overlap ↓ | Movement Efficiency ↑
    """
    # Calculate mean and std for each strategy
    summary = metrics_df.groupby("strategy").agg({
        "steps": ["mean", "std"],
        "distance": ["mean", "std"],
        "sensing_redundancy": ["mean", "std"],
        "visit_overlap": ["mean", "std"],
        "movement_efficiency": ["mean", "std"]
    })
    
    # Format the table
    table_data = []
    strategies = summary.index.tolist()
    
    # Define display名称 mapping
    display_names = {
        "random": "Random",
        "nearest_frontier": "Nearest Frontier",
        # "greedy_frontier": "Greedy Frontier",
        "cluster_frontier": "Cluster Frontier",
        "hungarian_bfs_frontier": "Hungarian + BFS",
        "hungarian_astar_frontier": "Hungarian + A*",
        # "hungarian_a": "Hungarian A",
        # "hungarian_b": "Hungarian B",
        # "hungarian_c": "Hungarian C",
        # "hungarian_d": "Hungarian D"
    }
    
    for strategy in strategies:
        display_name = display_names.get(strategy, strategy)
        row = [display_name]
        
        # Steps (lower is better)
        steps_mean = summary.loc[strategy, ("steps", "mean")]
        steps_std = summary.loc[strategy, ("steps", "std")]
        row.append(f"{steps_mean:.0f} ± {steps_std:.0f}")
        
        # Distance (lower is better)
        dist_mean = summary.loc[strategy, ("distance", "mean")]
        dist_std = summary.loc[strategy, ("distance", "std")]
        row.append(f"{dist_mean:.0f} ± {dist_std:.0f}")
        
        # Sensing Redundancy (lower is better)
        sens_mean = summary.loc[strategy, ("sensing_redundancy", "mean")]
        sens_std = summary.loc[strategy, ("sensing_redundancy", "std")]
        row.append(f"{sens_mean:.1f}% ± {sens_std:.1f}%")
        
        # Visit Overlap (lower is better)
        visit_mean = summary.loc[strategy, ("visit_overlap", "mean")]
        visit_std = summary.loc[strategy, ("visit_overlap", "std")]
        row.append(f"{visit_mean:.1f}% ± {visit_std:.1f}%")
        
        # Movement Efficiency (higher is better)
        move_mean = summary.loc[strategy, ("movement_efficiency", "mean")]
        move_std = summary.loc[strategy, ("movement_efficiency", "std")]
        row.append(f"{move_mean:.3f} ± {move_std:.3f}")
        
        table_data.append(row)
    
    # Create table with markdown format
    headers = ["Method", "Steps ↓", "Distance ↓", "Sensing Redundancy ↓", "Visit Overlap ↓", "Movement Efficiency ↑"]
    table = tabulate(
        table_data,
        headers=headers,
        tablefmt="github"
    )
    
    return table, summary


def save_summary_csv(metrics_df, output_dir):
    """
    Save detailed summary CSV with all metrics.
    """
    # Calculate mean and std for each strategy
    summary = metrics_df.groupby("strategy").agg({
        "steps": ["mean", "std"],
        "distance": ["mean", "std"],
        "coverage": ["mean", "std"],
        "sensing_redundancy": ["mean", "std"],
        "visit_overlap": ["mean", "std"],
        "movement_efficiency": ["mean", "std"],
        "mean_pairwise_distance": ["mean", "std"],
        "nodes_expanded": ["mean", "std"]
    })
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    output_file = os.path.join(output_dir, "summary.csv")
    summary.to_csv(output_file, index=False)
    print(f"Summary CSV saved to: {output_file}")


def plot_coverage_vs_steps(data, output_dir):
    """
    Plot Coverage vs Steps for all strategies.
    """
    plt.figure(figsize=(10, 6))
    
    for strategy, strategy_df in data.groupby("strategy"):
        stats = (
            strategy_df
            .groupby("timestep")["coverage"]
            .agg(["mean", "std"])
            .reset_index()
        )
        
        plt.plot(
            stats["timestep"],
            stats["mean"],
            label=strategy
        )
        
        plt.fill_between(
            stats["timestep"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2
        )
    
    plt.xlabel("Simulation Timestep")
    plt.ylabel("Coverage (%)")
    plt.title("Coverage vs Steps")
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(output_dir, "coverage_vs_steps.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_file}")


def plot_sensing_redundancy(data, output_dir):
    """
    Plot Sensing Redundancy vs Timestep for all strategies.
    """
    plt.figure(figsize=(10, 6))
    
    for strategy, strategy_df in data.groupby("strategy"):
        stats = (
            strategy_df
            .groupby("timestep")["sensing_redundancy"]
            .agg(["mean", "std"])
            .reset_index()
        )
        
        plt.plot(
            stats["timestep"],
            stats["mean"],
            label=strategy
        )
        
        plt.fill_between(
            stats["timestep"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2
        )
    
    plt.xlabel("Simulation Timestep")
    plt.ylabel("Sensing Redundancy (%)")
    plt.title("Sensing Redundancy")
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(output_dir, "sensing_redundancy.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_file}")


def plot_visit_overlap(data, output_dir):
    """
    Plot Visit Overlap vs Timestep for all strategies.
    """
    plt.figure(figsize=(10, 6))
    
    for strategy, strategy_df in data.groupby("strategy"):
        stats = (
            strategy_df
            .groupby("timestep")["visit_overlap"]
            .agg(["mean", "std"])
            .reset_index()
        )
        
        plt.plot(
            stats["timestep"],
            stats["mean"],
            label=strategy
        )
        
        plt.fill_between(
            stats["timestep"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2
        )
    
    plt.xlabel("Simulation Timestep")
    plt.ylabel("Visit Overlap (%)")
    plt.title("Visit Overlap")
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(output_dir, "visit_overlap.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_file}")


def plot_movement_efficiency(data, output_dir):
    """
    Plot Movement Efficiency vs Timestep for all strategies.
    """
    plt.figure(figsize=(10, 6))
    
    for strategy, strategy_df in data.groupby("strategy"):
        stats = (
            strategy_df
            .groupby("timestep")["movement_efficiency"]
            .agg(["mean", "std"])
            .reset_index()
        )
        
        plt.plot(
            stats["timestep"],
            stats["mean"],
            label=strategy
        )
        
        plt.fill_between(
            stats["timestep"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2
        )
    
    plt.xlabel("Simulation Timestep")
    plt.ylabel("Movement Efficiency")
    plt.title("Movement Efficiency")
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(output_dir, "movement_efficiency.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_file}")


def main():
    """
    Main function to generate comparison table and plots.
    """
    print("=" * 60)
    print("Generating Baseline Comparison")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(BASELINE_COMPARISON_DIR, exist_ok=True)
    
    # Load all data
    data = load_all_data()
    if data is None:
        return
    
    print(f"Loaded data from {len(data)} records")
    
    # Calculate final metrics
    metrics_df = calculate_final_metrics(data)
    print(f"Calculated metrics for {len(metrics_df)} runs")
    
    # Generate comparison table
    table, summary = generate_comparison_table(metrics_df)
    
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON TABLE")
    print("=" * 60)
    print(table)
    print("=" * 60 + "\n")
    
    # Save table to file
    table_file = os.path.join(BASELINE_COMPARISON_DIR, "comparison_table.md")
    with open(table_file, "w") as f:
        f.write("# Baseline Comparison\n\n")
        f.write(table)
    print(f"Comparison table saved to: {table_file}")
    
    # Save summary CSV
    save_summary_csv(metrics_df, BASELINE_COMPARISON_DIR)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_coverage_vs_steps(data, BASELINE_COMPARISON_DIR)
    plot_sensing_redundancy(data, BASELINE_COMPARISON_DIR)
    plot_visit_overlap(data, BASELINE_COMPARISON_DIR)
    plot_movement_efficiency(data, BASELINE_COMPARISON_DIR)
    
    print("\n" + "=" * 60)
    print("Baseline comparison generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
