import os
import glob

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


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

        final_metrics.append({
            "strategy": strategy,
            "run_id": run_id,
            "steps": steps,
            "sensing_redundancy": sensing_redundancy,
            "visit_overlap": visit_overlap,
            "efficiency": avg_efficiency,
            "pairwise_distance": avg_pairwise_distance,
            "movement_efficiency": avg_movement_efficiency
        })

    metrics_df = pd.DataFrame(final_metrics)

    # Calculate mean and std for each strategy
    summary = metrics_df.groupby("strategy").agg({
        "steps": ["mean", "std"],
        "sensing_redundancy": ["mean", "std"],
        "visit_overlap": ["mean", "std"],
        "efficiency": ["mean", "std"],
        "pairwise_distance": ["mean", "std"],
        "movement_efficiency": ["mean", "std"]
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

        table_data.append(row)

    # Create and print the table
    headers = ["Strategy", "Steps", "Sensing Redundancy", "Visit Overlap", "Efficiency", "Mean Separation", "Movement Efficiency"]
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


if __name__ == "__main__":
    analyze_all_runs()

    # Aggregate plots:
    # Mean +/- std across ALL runs
    plot_aggregate_coverage()
    plot_aggregate_sensing_redundancy()
    plot_aggregate_visit_overlap()
    plot_aggregate_efficiency()
    plot_aggregate_pairwise_distance()
    plot_aggregate_movement_efficiency()

    # Generate aggregated metrics table
    generate_aggregated_metrics_table()