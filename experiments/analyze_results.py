import os
import glob

import pandas as pd
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    analyze_all_runs()