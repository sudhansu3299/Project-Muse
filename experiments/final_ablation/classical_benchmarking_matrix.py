import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "results/classical_benchmark/raw_results.csv"

OUTPUT_DIR = "results/analysis"

# Metrics where LOWER is better
LOWER_IS_BETTER = [
    "time_to_90",
    "distance",
    "sensing_redundancy",
]

# Metrics where HIGHER is better
HIGHER_IS_BETTER = [
    "movement_efficiency",
    "total_reward",
]


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\nLoaded:")
print(df.head())

print("\nStrategies:")
print(df["strategy"].unique())

print("\nSeeds:")
print(sorted(df["seed"].unique()))


# ============================================================
# Create output directory
# ============================================================

import os

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. WINNER MATRIX
# ============================================================

def create_winner_matrix(df, metric, lower_is_better=True):

    table = df.pivot(
        index="seed",
        columns="strategy",
        values=metric,
    )

    if lower_is_better:
        winners = table.idxmin(axis=1)
    else:
        winners = table.idxmax(axis=1)

    winner_matrix = table.copy()

    for seed in table.index:
        winner = winners.loc[seed]

        for strategy in table.columns:
            if strategy == winner:
                winner_matrix.loc[seed, strategy] = 1
            else:
                winner_matrix.loc[seed, strategy] = 0

    return table, winner_matrix, winners


# ============================================================
# 2. PRINT WINNER COUNTS
# ============================================================

def analyze_metric(df, metric, lower_is_better=True):

    table, winner_matrix, winners = create_winner_matrix(
        df,
        metric,
        lower_is_better,
    )

    print()
    print("=" * 70)
    print(f"WINNER ANALYSIS: {metric}")
    print("=" * 70)

    print("\nPer-seed winner:")
    print(winners)

    print("\nWin counts:")
    print(winners.value_counts())

    print("\nWin percentage:")
    print(
        winners.value_counts(normalize=True).mul(100).round(1)
    )

    return table, winner_matrix, winners


# ============================================================
# 3. WINNER HEATMAP
# ============================================================

def plot_winner_matrix(
        winner_matrix,
        metric,
        output_path,
):

    plt.figure(
        figsize=(
            max(8, len(winner_matrix.columns) * 1.5),
            max(5, len(winner_matrix) * 0.45),
        )
    )

    plt.imshow(
        winner_matrix.values,
        aspect="auto",
        interpolation="nearest",
    )

    plt.xticks(
        range(len(winner_matrix.columns)),
        winner_matrix.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(winner_matrix.index)),
        winner_matrix.index,
    )

    plt.xlabel("Strategy")
    plt.ylabel("Seed")

    plt.title(
        f"Per-Seed Winner Matrix — {metric}"
    )

    plt.colorbar(
        label="Winner (1 = yes)"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 4. REGRET
# ============================================================

def calculate_regret(
        df,
        metric,
        lower_is_better=True,
):

    table = df.pivot(
        index="seed",
        columns="strategy",
        values=metric,
    )

    if lower_is_better:

        best = table.min(axis=1)

        regret = table.sub(
            best,
            axis=0,
        )

    else:

        best = table.max(axis=1)

        regret = table.rsub(
            best,
            axis=0,
        )

    return table, regret


# ============================================================
# 5. NORMALIZED REGRET (%)
# ============================================================

def calculate_percentage_regret(
        df,
        metric,
        lower_is_better=True,
):

    table = df.pivot(
        index="seed",
        columns="strategy",
        values=metric,
    )

    if lower_is_better:

        best = table.min(axis=1)

        regret = table.sub(
            best,
            axis=0,
        )

        percentage_regret = (
                regret
                .div(best, axis=0)
                * 100
        )

    else:

        best = table.max(axis=1)

        regret = table.rsub(
            best,
            axis=0,
        )

        percentage_regret = (
                regret
                .div(best, axis=0)
                * 100
        )

    return percentage_regret


# ============================================================
# 6. RUN ANALYSIS
# ============================================================

metrics_to_analyze = {
    "time_to_90": True,
    "distance": True,
    "sensing_redundancy": True,
}

for metric, lower_is_better in metrics_to_analyze.items():

    if metric not in df.columns:
        print(
            f"\nSkipping {metric}: "
            "column not found."
        )
        continue

    # --------------------------------------------------------
    # Winner analysis
    # --------------------------------------------------------

    table, winner_matrix, winners = analyze_metric(
        df,
        metric,
        lower_is_better,
    )

    # --------------------------------------------------------
    # Save raw per-seed table
    # --------------------------------------------------------

    table.to_csv(
        f"{OUTPUT_DIR}/{metric}_per_seed.csv"
    )

    # --------------------------------------------------------
    # Save winner matrix
    # --------------------------------------------------------

    winner_matrix.to_csv(
        f"{OUTPUT_DIR}/{metric}_winner_matrix.csv"
    )

    # --------------------------------------------------------
    # Plot winner matrix
    # --------------------------------------------------------

    plot_winner_matrix(
        winner_matrix,
        metric,
        f"{OUTPUT_DIR}/{metric}_winner_matrix.png",
    )

    # --------------------------------------------------------
    # Regret
    # --------------------------------------------------------

    _, regret = calculate_regret(
        df,
        metric,
        lower_is_better,
    )

    regret.to_csv(
        f"{OUTPUT_DIR}/{metric}_regret.csv"
    )

    print()
    print("Raw regret:")
    print(regret.round(2))

    # --------------------------------------------------------
    # Percentage regret
    # --------------------------------------------------------

    percentage_regret = calculate_percentage_regret(
        df,
        metric,
        lower_is_better,
    )

    percentage_regret.to_csv(
        f"{OUTPUT_DIR}/{metric}_percentage_regret.csv"
    )

    print()
    print("Percentage regret:")
    print(
        percentage_regret.round(2)
    )


# ============================================================
# 7. OVERALL WIN COUNT
# ============================================================

print()
print("=" * 70)
print("OVERALL WIN COUNTS")
print("=" * 70)

for metric, lower_is_better in metrics_to_analyze.items():

    if metric not in df.columns:
        continue

    _, _, winners = analyze_metric(
        df,
        metric,
        lower_is_better,
    )

    print(f"\n{metric}:")
    print(winners.value_counts())


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to: {OUTPUT_DIR}/"
)