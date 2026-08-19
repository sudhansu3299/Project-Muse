import os
import glob
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import plotly.graph_objects as go
from plotly.subplots import make_subplots


RESULTS_DIR = "results/raw"
PLOTS_DIR = "plots"


def plot_run(run_dir):
    """
    Plot Coverage vs Timestep for all strategies
    belonging to one experimental run.
    """

    csv_files = glob.glob(
        os.path.join(run_dir, "*.csv")
    )

    if not csv_files:
        print(f"No CSV files found in {run_dir}")
        return

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:

        data = pd.read_csv(csv_file)

        strategy_name = data["strategy"].iloc[0]

        plt.plot(
            data["timestep"],
            data["coverage"],
            label=strategy_name
        )

    plt.xlabel("Simulation Timestep")
    plt.ylabel("Coverage (%)")

    run_name = os.path.basename(run_dir)

    plt.title(
        f"Multi-UAV Exploration - {run_name}"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        PLOTS_DIR,
        f"{run_name}_coverage.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Plot saved: {output_file}"
    )


def plot_run_sensing_redundancy(run_dir):
    """
    Plot Sensing Redundancy vs Timestep for all strategies
    belonging to one experimental run.
    """

    csv_files = glob.glob(
        os.path.join(run_dir, "*.csv")
    )

    if not csv_files:
        print(f"No CSV files found in {run_dir}")
        return

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:

        data = pd.read_csv(csv_file)

        strategy_name = data["strategy"].iloc[0]

        plt.plot(
            data["timestep"],
            data["sensing_redundancy"],
            label=strategy_name
        )

    plt.xlabel("Simulation Timestep")
    plt.ylabel("Sensing Redundancy (%)")

    run_name = os.path.basename(run_dir)

    plt.title(
        f"Multi-UAV Exploration Sensing Redundancy - {run_name}"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        PLOTS_DIR,
        f"{run_name}_sensing_redundancy.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Plot saved: {output_file}"
    )


def plot_run_visit_overlap(run_dir):
    """
    Plot Visit Overlap vs Timestep for all strategies
    belonging to one experimental run.
    """

    csv_files = glob.glob(
        os.path.join(run_dir, "*.csv")
    )

    if not csv_files:
        print(f"No CSV files found in {run_dir}")
        return

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:

        data = pd.read_csv(csv_file)

        strategy_name = data["strategy"].iloc[0]

        plt.plot(
            data["timestep"],
            data["visit_overlap"],
            label=strategy_name
        )

    plt.xlabel("Simulation Timestep")
    plt.ylabel("Visit Overlap (%)")

    run_name = os.path.basename(run_dir)

    plt.title(
        f"Multi-UAV Exploration Visit Overlap - {run_name}"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        PLOTS_DIR,
        f"{run_name}_visit_overlap.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Plot saved: {output_file}"
    )


def plot_run_efficiency(run_dir):
    """
    Plot Exploration Efficiency vs Timestep for all strategies
    belonging to one experimental run.
    """

    csv_files = glob.glob(
        os.path.join(run_dir, "*.csv")
    )

    if not csv_files:
        print(f"No CSV files found in {run_dir}")
        return

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:

        data = pd.read_csv(csv_file)

        strategy_name = data["strategy"].iloc[0]

        plt.plot(
            data["timestep"],
            data["exploration_efficiency"],
            label=strategy_name
        )

    plt.xlabel("Simulation Timestep")
    plt.ylabel("Exploration Efficiency (ΔCoverage/ΔDistance)")

    run_name = os.path.basename(run_dir)

    plt.title(
        f"Multi-UAV Exploration Efficiency - {run_name}"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        PLOTS_DIR,
        f"{run_name}_efficiency.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Plot saved: {output_file}"
    )


def plot_run_pairwise_distance(run_dir):
    """
    Plot Mean Pairwise Distance vs Timestep for all strategies
    belonging to one experimental run.
    """

    csv_files = glob.glob(
        os.path.join(run_dir, "*.csv")
    )

    if not csv_files:
        print(f"No CSV files found in {run_dir}")
        return

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:

        data = pd.read_csv(csv_file)

        strategy_name = data["strategy"].iloc[0]

        plt.plot(
            data["timestep"],
            data["mean_pairwise_distance"],
            label=strategy_name
        )

    plt.xlabel("Simulation Timestep")
    plt.ylabel("Mean Pairwise Distance")

    run_name = os.path.basename(run_dir)

    plt.title(
        f"Multi-UAV Mean Pairwise Distance - {run_name}"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        PLOTS_DIR,
        f"{run_name}_pairwise_distance.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Plot saved: {output_file}"
    )


def plot_run_movement_efficiency(run_dir):
    """
    Plot Movement Efficiency vs Timestep for all strategies
    belonging to one experimental run.
    """

    csv_files = glob.glob(
        os.path.join(run_dir, "*.csv")
    )

    if not csv_files:
        print(f"No CSV files found in {run_dir}")
        return

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:

        data = pd.read_csv(csv_file)

        strategy_name = data["strategy"].iloc[0]

        plt.plot(
            data["timestep"],
            data["movement_efficiency"],
            label=strategy_name
        )

    plt.xlabel("Simulation Timestep")
    plt.ylabel("Movement Efficiency (unique cells / distance)")

    run_name = os.path.basename(run_dir)

    plt.title(
        f"Multi-UAV Movement Efficiency - {run_name}"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        PLOTS_DIR,
        f"{run_name}_movement_efficiency.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Plot saved: {output_file}"
    )


def analyze_all_runs():

    run_dirs = sorted(
        glob.glob(
            os.path.join(
                RESULTS_DIR,
                "run_*"
            )
        )
    )

    for run_dir in run_dirs:
        plot_run(run_dir)
        plot_run_sensing_redundancy(run_dir)
        plot_run_visit_overlap(run_dir)
        plot_run_efficiency(run_dir)
        plot_run_pairwise_distance(run_dir)
        plot_run_movement_efficiency(run_dir)

def plot_aggregate_coverage():

    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []

    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

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

    plt.title(
        "Multi-UAV Exploration Performance"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "aggregate_coverage.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_aggregate_sensing_redundancy():

    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []

    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

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

    plt.title(
        "Multi-UAV Exploration Sensing Redundancy Performance"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "aggregate_sensing_redundancy.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_aggregate_visit_overlap():

    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []

    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

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

    plt.title(
        "Multi-UAV Exploration Visit Overlap Performance"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "aggregate_visit_overlap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_aggregate_efficiency():

    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []

    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    plt.figure(figsize=(10, 6))

    for strategy, strategy_df in data.groupby("strategy"):

        stats = (
            strategy_df
            .groupby("timestep")["exploration_efficiency"]
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
    plt.ylabel("Exploration Efficiency (ΔCoverage/ΔDistance)")

    plt.title(
        "Multi-UAV Exploration Efficiency Performance"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "aggregate_efficiency.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_aggregate_pairwise_distance():

    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []

    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    plt.figure(figsize=(10, 6))

    for strategy, strategy_df in data.groupby("strategy"):

        stats = (
            strategy_df
            .groupby("timestep")["mean_pairwise_distance"]
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
    plt.ylabel("Mean Pairwise Distance")

    plt.title(
        "Multi-UAV Mean Pairwise Distance Performance"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "aggregate_pairwise_distance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_aggregate_movement_efficiency():

    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []

    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

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
    plt.ylabel("Movement Efficiency (unique cells / distance)")

    plt.title(
        "Multi-UAV Movement Efficiency Performance"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "aggregate_movement_efficiency.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_aggregate_nodes_expanded():

    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []

    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    plt.figure(figsize=(10, 6))

    for strategy, strategy_df in data.groupby("strategy"):

        stats = (
            strategy_df
            .groupby("timestep")["nodes_expanded"]
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
    plt.ylabel("Nodes Expanded")

    plt.title(
        "Multi-UAV Planner Nodes Expanded Performance"
    )

    plt.legend()
    plt.grid(True)

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "aggregate_nodes_expanded.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_nodes_expanded_comparison():
    """
    Create a bar chart comparing total nodes expanded for BFS vs A* planners.
    Only includes hungarian strategies since they use the planners.
    """
    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Filter only hungarian strategies
    hungarian_data = data[data["strategy"].str.contains("hungarian")]

    # Check if nodes_expanded column exists
    if "nodes_expanded" not in hungarian_data.columns:
        print("nodes_expanded column not found in data. Skipping comparison plot.")
        return

    # Calculate total nodes expanded for each run
    final_metrics = []
    for (strategy, run_id), group in hungarian_data.groupby(["strategy", "run_id"]):
        total_nodes = group["nodes_expanded"].max()
        final_metrics.append({
            "strategy": strategy,
            "run_id": run_id,
            "total_nodes_expanded": total_nodes
        })

    metrics_df = pd.DataFrame(final_metrics)

    # Calculate mean and std for each strategy
    summary = metrics_df.groupby("strategy").agg({
        "total_nodes_expanded": ["mean", "std"]
    })

    plt.figure(figsize=(10, 6))

    strategies = summary.index.tolist()
    means = [summary.loc[s, ("total_nodes_expanded", "mean")] for s in strategies]
    stds = [summary.loc[s, ("total_nodes_expanded", "std")] for s in strategies]

    colors = ['#2ca02c', '#d62728']  # Green for A*, Red for BFS
    bars = plt.bar(strategies, means, yerr=stds, capsize=5, color=colors, alpha=0.7)

    plt.ylabel("Total Nodes Expanded")
    plt.title("Planner Efficiency: BFS vs A* (Total Nodes Expanded)")
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{mean:.0f} ± {std:.0f}',
                 ha='center', va='bottom', fontsize=10)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_file = os.path.join(PLOTS_DIR, "nodes_expanded_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Nodes expanded comparison plot saved: {output_file}")


def calculate_coverage_auc(group):
    """
    Calculate Area Under Curve for coverage over time.
    Uses trapezoidal integration.
    """
    timesteps = group["timestep"].values
    coverage = group["coverage"].values
    auc = np.trapezoid(coverage, timesteps)
    return auc


def generate_aggregated_metrics_table():
    """
    Generate a table showing aggregated metrics with standard deviations
    for all strategies across all runs.
    """

    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []

    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    # Calculate final metrics for each run
    final_metrics = []

    for (strategy, run_id), group in data.groupby(["strategy", "run_id"]):
        # Get the last row for this run
        last_row = group.iloc[-1]

        steps = last_row["timestep"]
        sensing_redundancy = last_row["sensing_redundancy"]
        visit_overlap = last_row["visit_overlap"]

        # Calculate average efficiency over the run
        avg_efficiency = group["exploration_efficiency"].mean()

        # Calculate average pairwise distance over the run
        avg_pairwise_distance = group["mean_pairwise_distance"].mean()

        # Calculate average movement efficiency over the run
        avg_movement_efficiency = group["movement_efficiency"].mean()

        # Calculate Coverage AUC
        coverage_auc = calculate_coverage_auc(group)

        # Calculate total nodes expanded (if available)
        if "nodes_expanded" in group.columns:
            total_nodes_expanded = group["nodes_expanded"].max()
        else:
            total_nodes_expanded = 0

        final_metrics.append({
            "strategy": strategy,
            "run_id": run_id,
            "steps": steps,
            "sensing_redundancy": sensing_redundancy,
            "visit_overlap": visit_overlap,
            "efficiency": avg_efficiency,
            "pairwise_distance": avg_pairwise_distance,
            "movement_efficiency": avg_movement_efficiency,
            "coverage_auc": coverage_auc,
            "nodes_expanded": total_nodes_expanded
        })

    metrics_df = pd.DataFrame(final_metrics)

    # Calculate mean and std for each strategy
    summary = metrics_df.groupby("strategy").agg({
        "steps": ["mean", "std"],
        "sensing_redundancy": ["mean", "std"],
        "visit_overlap": ["mean", "std"],
        "efficiency": ["mean", "std"],
        "pairwise_distance": ["mean", "std"],
        "movement_efficiency": ["mean", "std"],
        "coverage_auc": ["mean", "std"],
        "nodes_expanded": ["mean", "std"]
    })

    # Format the table
    table_data = []
    strategies = summary.index.tolist()

    for strategy in strategies:
        row = [strategy]

        # Steps
        steps_mean = summary.loc[strategy, ("steps", "mean")]
        steps_std = summary.loc[strategy, ("steps", "std")]
        row.append(f"{steps_mean:.0f} ± {steps_std:.0f}")

        # Sensing Redundancy
        sensing_mean = summary.loc[strategy, ("sensing_redundancy", "mean")]
        sensing_std = summary.loc[strategy, ("sensing_redundancy", "std")]
        row.append(f"{sensing_mean:.1f}% ± {sensing_std:.1f}%")

        # Visit Overlap
        visit_mean = summary.loc[strategy, ("visit_overlap", "mean")]
        visit_std = summary.loc[strategy, ("visit_overlap", "std")]
        row.append(f"{visit_mean:.1f}% ± {visit_std:.1f}%")

        # Efficiency
        eff_mean = summary.loc[strategy, ("efficiency", "mean")]
        eff_std = summary.loc[strategy, ("efficiency", "std")]
        row.append(f"{eff_mean:.3f} ± {eff_std:.3f}")

        # Pairwise Distance
        dist_mean = summary.loc[strategy, ("pairwise_distance", "mean")]
        dist_std = summary.loc[strategy, ("pairwise_distance", "std")]
        row.append(f"{dist_mean:.2f} ± {dist_std:.2f}")

        # Movement Efficiency
        move_mean = summary.loc[strategy, ("movement_efficiency", "mean")]
        move_std = summary.loc[strategy, ("movement_efficiency", "std")]
        row.append(f"{move_mean:.3f} ± {move_std:.3f}")

        # Coverage AUC
        auc_mean = summary.loc[strategy, ("coverage_auc", "mean")]
        auc_std = summary.loc[strategy, ("coverage_auc", "std")]
        row.append(f"{auc_mean:.1f} ± {auc_std:.1f}")

        # Nodes Expanded
        nodes_mean = summary.loc[strategy, ("nodes_expanded", "mean")]
        nodes_std = summary.loc[strategy, ("nodes_expanded", "std")]
        row.append(f"{nodes_mean:.0f} ± {nodes_std:.0f}")

        table_data.append(row)

    # Create and print the table
    headers = ["Strategy", "Steps", "Sensing Redundancy", "Visit Overlap", "Efficiency", "Mean Separation", "Movement Efficiency", "Coverage AUC", "Nodes Expanded"]
    table = tabulate(
        table_data,
        headers=headers,
        tablefmt="grid",
        floatfmt=".2f"
    )

    print("\n" + "=" * 60)
    print("AGGREGATED METRICS SUMMARY (Mean ± Std)")
    print("=" * 60)
    print(table)
    print("=" * 60 + "\n")

    # Save table to file
    os.makedirs(PLOTS_DIR, exist_ok=True)
    table_file = os.path.join(PLOTS_DIR, "aggregated_metrics_table.txt")
    
    with open(table_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("AGGREGATED METRICS SUMMARY (Mean ± Std)\n")
        f.write("=" * 60 + "\n")
        f.write(table + "\n")
        f.write("=" * 60 + "\n")
    
    print(f"Table saved to: {table_file}")


def plot_spider_chart():
    """
    Create a spider/radar plot comparing all strategies across all metrics.
    Metrics are normalized to 0-1 scale for fair comparison.
    """
    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Calculate final metrics for each run
    final_metrics = []

    for (strategy, run_id), group in data.groupby(["strategy", "run_id"]):
        last_row = group.iloc[-1]

        steps = last_row["timestep"]
        sensing_redundancy = last_row["sensing_redundancy"]
        visit_overlap = last_row["visit_overlap"]
        avg_efficiency = group["exploration_efficiency"].mean()
        avg_pairwise_distance = group["mean_pairwise_distance"].mean()
        avg_movement_efficiency = group["movement_efficiency"].mean()
        coverage_auc = calculate_coverage_auc(group)
        
        if "nodes_expanded" in group.columns:
            total_nodes_expanded = group["nodes_expanded"].max()
        else:
            total_nodes_expanded = 0

        final_metrics.append({
            "strategy": strategy,
            "steps": steps,
            "sensing_redundancy": sensing_redundancy,
            "visit_overlap": visit_overlap,
            "efficiency": avg_efficiency,
            "pairwise_distance": avg_pairwise_distance,
            "movement_efficiency": avg_movement_efficiency,
            "coverage_auc": coverage_auc,
            "nodes_expanded": total_nodes_expanded
        })

    metrics_df = pd.DataFrame(final_metrics)

    # Calculate mean for each strategy
    summary = metrics_df.groupby("strategy").agg({
        "steps": "mean",
        "sensing_redundancy": "mean",
        "visit_overlap": "mean",
        "efficiency": "mean",
        "pairwise_distance": "mean",
        "movement_efficiency": "mean",
        "coverage_auc": "mean",
        "nodes_expanded": "mean"
    })

    # Normalize metrics to 0-1 scale
    # For steps: lower is better, so invert
    summary["steps_norm"] = 1 - (summary["steps"] - summary["steps"].min()) / (summary["steps"].max() - summary["steps"].min())
    # For sensing_redundancy: lower is better, so invert
    summary["sensing_redundancy_norm"] = 1 - (summary["sensing_redundancy"] - summary["sensing_redundancy"].min()) / (summary["sensing_redundancy"].max() - summary["sensing_redundancy"].min())
    # For visit_overlap: lower is better, so invert
    summary["visit_overlap_norm"] = 1 - (summary["visit_overlap"] - summary["visit_overlap"].min()) / (summary["visit_overlap"].max() - summary["visit_overlap"].min())
    # For efficiency: higher is better
    summary["efficiency_norm"] = (summary["efficiency"] - summary["efficiency"].min()) / (summary["efficiency"].max() - summary["efficiency"].min())
    # For pairwise_distance: higher is better (more separation)
    summary["pairwise_distance_norm"] = (summary["pairwise_distance"] - summary["pairwise_distance"].min()) / (summary["pairwise_distance"].max() - summary["pairwise_distance"].min())
    # For movement_efficiency: higher is better
    summary["movement_efficiency_norm"] = (summary["movement_efficiency"] - summary["movement_efficiency"].min()) / (summary["movement_efficiency"].max() - summary["movement_efficiency"].min())
    # For coverage_auc: higher is better
    summary["coverage_auc_norm"] = (summary["coverage_auc"] - summary["coverage_auc"].min()) / (summary["coverage_auc"].max() - summary["coverage_auc"].min())
    # For nodes_expanded: lower is better, so invert
    summary["nodes_expanded_norm"] = 1 - (summary["nodes_expanded"] - summary["nodes_expanded"].min()) / (summary["nodes_expanded"].max() - summary["nodes_expanded"].min())

    # Categories for radar chart
    categories = ["Steps\n(fewer is better)", "Sensing Redundancy\n(lower is better)", 
                  "Visit Overlap\n(lower is better)", "Efficiency\n(higher is better)",
                  "Mean Separation\n(higher is better)", "Movement Efficiency\n(higher is better)",
                  "Coverage AUC\n(higher is better)", "Nodes Expanded\n(lower is better)"]

    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Number of variables
    N = len(categories)

    # Angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Plot each strategy
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (strategy, row) in enumerate(summary.iterrows()):
        values = [
            row["steps_norm"],
            row["sensing_redundancy_norm"],
            row["visit_overlap_norm"],
            row["efficiency_norm"],
            row["pairwise_distance_norm"],
            row["movement_efficiency_norm"],
            row["coverage_auc_norm"],
            row["nodes_expanded_norm"]
        ]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True)

    plt.title("Multi-UAV Exploration Strategy Comparison\n(Normalized Metrics)", 
              size=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_file = os.path.join(PLOTS_DIR, "strategy_spider_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Spider chart saved: {output_file}")


def plot_interactive_dashboard():
    """
    Create an interactive Plotly dashboard with all metrics.
    Includes time series plots with hover tooltips and metric toggling.
    """
    csv_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "run_*",
            "*.csv"
        )
    )

    frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Create subplots for each metric
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=("Coverage", "Sensing Redundancy", "Visit Overlap",
                       "Exploration Efficiency", "Mean Pairwise Distance", "Movement Efficiency",
                       "Nodes Expanded", "Coverage AUC (Cumulative)"),
        vertical_spacing=0.10,
        horizontal_spacing=0.10
    )

    # Color palette for strategies
    colors = {
        'cluster_frontier': '#1f77b4',
        'cluster_utility_frontier': '#ff7f0e',
        'hungarian_astar_frontier': '#2ca02c',
        'hungarian_bfs_frontier': '#d62728'
    }

    # Plot each metric
    metrics = [
        ('coverage', 'Coverage (%)', 1, 1),
        ('sensing_redundancy', 'Sensing Redundancy (%)', 1, 2),
        ('visit_overlap', 'Visit Overlap (%)', 2, 1),
        ('exploration_efficiency', 'Exploration Efficiency', 2, 2),
        ('mean_pairwise_distance', 'Mean Pairwise Distance', 3, 1),
        ('movement_efficiency', 'Movement Efficiency', 3, 2),
        ('nodes_expanded', 'Nodes Expanded', 4, 1)
    ]

    for metric, y_label, row, col in metrics:
        for strategy, strategy_df in data.groupby("strategy"):
            stats = (
                strategy_df
                .groupby("timestep")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )

            fig.add_trace(
                go.Scatter(
                    x=stats["timestep"],
                    y=stats["mean"],
                    mode='lines',
                    name=strategy if row == 1 and col == 1 else None,
                    legendgroup=strategy,
                    showlegend=row == 1 and col == 1,
                    line=dict(color=colors.get(strategy, '#000000'), width=2),
                    hovertemplate=f"<b>{strategy}</b><br>" +
                                 f"Timestep: %{{x}}<br>" +
                                 f"{y_label}: %{{y:.3f}}<br>" +
                                 f"<extra></extra>",
                    customdata=stats["std"],
                    hoverlabel=dict(bgcolor="white")
                ),
                row=row, col=col
            )

            # Add confidence band
            fig.add_trace(
                go.Scatter(
                    x=stats["timestep"].tolist() + stats["timestep"].tolist()[::-1],
                    y=(stats["mean"] + stats["std"]).tolist() + (stats["mean"] - stats["std"]).tolist()[::-1],
                    fill='toself',
                    fillcolor=colors.get(strategy, '#000000'),
                    opacity=0.1,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

    # Add cumulative Coverage AUC plot
    for strategy, strategy_df in data.groupby("strategy"):
        # Calculate cumulative AUC for each run, then average
        cumulative_auc_data = []
        for run_id, run_group in strategy_df.groupby("run_id"):
            run_group = run_group.sort_values("timestep")
            timesteps = run_group["timestep"].values
            coverage = run_group["coverage"].values
            cumulative_auc = np.cumsum(coverage * np.gradient(timesteps))
            cumulative_auc_data.append(pd.DataFrame({
                "timestep": timesteps,
                "cumulative_auc": cumulative_auc
            }))
        
        # Average across runs
        if cumulative_auc_data:
            combined = pd.concat(cumulative_auc_data)
            stats = combined.groupby("timestep")["cumulative_auc"].agg(["mean", "std"]).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=stats["timestep"],
                    y=stats["mean"],
                    mode='lines',
                    name=strategy,
                    legendgroup=strategy,
                    showlegend=False,
                    line=dict(color=colors.get(strategy, '#000000'), width=2),
                    hovertemplate=f"<b>{strategy}</b><br>" +
                                 f"Timestep: %{{x}}<br>" +
                                 f"Cumulative AUC: %{{y:.1f}}<br>" +
                                 f"<extra></extra>",
                    hoverlabel=dict(bgcolor="white")
                ),
                row=4, col=2
            )

    # Update layout
    fig.update_xaxes(title_text="Simulation Timestep", row=3, col=1)
    fig.update_xaxes(title_text="Simulation Timestep", row=3, col=2)
    fig.update_xaxes(title_text="Simulation Timestep", row=4, col=1)
    fig.update_xaxes(title_text="Simulation Timestep", row=4, col=2)
    fig.update_yaxes(title_text="Coverage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sensing Redundancy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Visit Overlap (%)", row=2, col=1)
    fig.update_yaxes(title_text="Exploration Efficiency", row=2, col=2)
    fig.update_yaxes(title_text="Mean Pairwise Distance", row=3, col=1)
    fig.update_yaxes(title_text="Movement Efficiency", row=3, col=2)
    fig.update_yaxes(title_text="Nodes Expanded", row=4, col=1)
    fig.update_yaxes(title_text="Cumulative AUC", row=4, col=2)

    fig.update_layout(
        title="Multi-UAV Exploration - Interactive Performance Dashboard",
        height=1200,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_file = os.path.join(PLOTS_DIR, "interactive_dashboard.html")
    fig.write_html(output_file)

    print(f"Interactive dashboard saved: {output_file}")


if __name__ == "__main__":
    # analyze_all_runs()

    # Aggregate plots:
    # Mean +/- std across ALL runs
    plot_aggregate_coverage()
    plot_aggregate_sensing_redundancy()
    plot_aggregate_visit_overlap()
    plot_aggregate_efficiency()
    plot_aggregate_pairwise_distance()
    plot_aggregate_movement_efficiency()
    plot_aggregate_nodes_expanded()

    # Generate aggregated metrics table
    generate_aggregated_metrics_table()

    # New rich visualizations
    plot_spider_chart()
    plot_interactive_dashboard()
    plot_nodes_expanded_comparison()