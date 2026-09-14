import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PPO_RESULTS = "results/ppo/evaluation/ppo_all_checkpoint_trajectories.csv"

FIXED_RESULTS = "results/baseline/fixed_weight_baseline_new.csv"

OUTPUT_DIR = "results/ppo/plots"

TEST_SEEDS = list(range(101, 111))

CHECKPOINT_ORDER = [
    "ppo_a_25k",
    "ppo_a_50k",
    "ppo_a_75k",
    "ppo_a_100k",
    "ppo_b_25k",
    "ppo_b_50k",
    "ppo_b_75k",
    "ppo_b_100k",
]


# ============================================================
# Setup
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Load data
# ============================================================

ppo = pd.read_csv(PPO_RESULTS)

fixed = pd.read_csv(FIXED_RESULTS)


# ============================================================
# Restrict to unseen seeds
# ============================================================

ppo = ppo[
    ppo["seed"].isin(TEST_SEEDS)
].copy()

fixed = fixed[
    fixed["seed"].isin(TEST_SEEDS)
].copy()


# ============================================================
# Check required columns
# ============================================================

required_ppo_columns = [
    "seed",
    "steps",
    "reached_90",
    "final_coverage",
    "total_distance",
    "final_redundancy",
    "mean_redundancy",
    "total_reward",
    "mean_reward",
    "checkpoint",
]

missing = [
    column
    for column in required_ppo_columns
    if column not in ppo.columns
]

if missing:
    raise ValueError(
        f"Missing PPO columns: {missing}"
    )


required_fixed_columns = [
    "seed",
    "steps",
    "reached_90",
    "final_coverage",
    "total_distance",
    "final_redundancy",
    "mean_redundancy",
    "total_reward",
    "mean_reward",
]

missing = [
    column
    for column in required_fixed_columns
    if column not in fixed.columns
]

if missing:
    raise ValueError(
        f"Missing fixed-baseline columns: {missing}"
    )


# ============================================================
# Keep checkpoint ordering
# ============================================================

ppo["checkpoint"] = pd.Categorical(
    ppo["checkpoint"],
    categories=CHECKPOINT_ORDER,
    ordered=True,
)


# ============================================================
# Summary statistics
# ============================================================

metrics = [
    "steps",
    "total_distance",
    "final_redundancy",
    "mean_redundancy",
    "total_reward",
    "mean_reward",
    "final_coverage",
]


summary = (
    ppo
    .groupby(
        "checkpoint",
        observed=True,
    )[metrics]
    .agg(["mean", "std"])
)


print()
print("=" * 70)
print("PPO CHECKPOINT SUMMARY — UNSEEN SEEDS 101–110")
print("=" * 70)

print(summary.round(3))


# ============================================================
# Success rate
# ============================================================

success_rate = (
        ppo
        .groupby(
            "checkpoint",
            observed=True,
        )["reached_90"]
        .mean()
        * 100
)

print()
print("Success rate:")
print(
    success_rate.round(2).to_string()
)


# ============================================================
# Fixed baseline summary
# ============================================================

fixed_mean = fixed[metrics].mean()
fixed_std = fixed[metrics].std()

fixed_success = (
        fixed["reached_90"].mean()
        * 100
)

print()
print("=" * 70)
print("FIXED-WEIGHT BASELINE — UNSEEN SEEDS 101–110")
print("=" * 70)

for metric in metrics:

    print(
        f"{metric:20s}: "
        f"{fixed_mean[metric]:.3f} "
        f"+/- {fixed_std[metric]:.3f}"
    )

print(
    f"success_rate        : "
    f"{fixed_success:.2f}%"
)


# ============================================================
# Helper function
# ============================================================

def checkpoint_stats(column):

    grouped = (
        ppo
        .groupby(
            "checkpoint",
            observed=True,
        )[column]
    )

    means = grouped.mean()
    stds = grouped.std()

    return (
        means,
        stds,
    )


def plot_checkpoint_metric(
        column,
        ylabel,
        title,
        filename,
        lower_is_better=False,
):

    means, stds = checkpoint_stats(column)

    x = np.arange(
        len(CHECKPOINT_ORDER)
    )

    y = means.reindex(
        CHECKPOINT_ORDER
    ).values

    error = stds.reindex(
        CHECKPOINT_ORDER
    ).values

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        x,
        y,
        yerr=error,
        marker="o",
        linewidth=2,
        capsize=5,
    )

    plt.xticks(
        x,
        ["25k_a", "50k_a", "75k_a", "100k_a", "25k_b", "50k_b", "75k_b", "100k_b"],
    )

    plt.xlabel(
        "PPO training timesteps"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 1. Steps to 90%
# ============================================================

plot_checkpoint_metric(
    column="steps",
    ylabel="Steps to 90% coverage",
    title=(
        "Exploration Time vs PPO Training"
    ),
    filename="01_steps_to_90.png",
)


# ============================================================
# 2. Total distance
# ============================================================

plot_checkpoint_metric(
    column="total_distance",
    ylabel="Total distance",
    title=(
        "Travel Distance vs PPO Training"
    ),
    filename="02_total_distance.png",
)


# ============================================================
# 3. Redundancy
# ============================================================

plot_checkpoint_metric(
    column="final_redundancy",
    ylabel="Final sensing redundancy (%)",
    title=(
        "Sensing Redundancy vs PPO Training"
    ),
    filename="03_final_redundancy.png",
)


# ============================================================
# 4. Reward
# ============================================================

plot_checkpoint_metric(
    column="total_reward",
    ylabel="Total episode reward",
    title=(
        "Episode Reward vs PPO Training"
    ),
    filename="04_total_reward.png",
)


# ============================================================
# 5. Mean redundancy
# ============================================================

plot_checkpoint_metric(
    column="mean_redundancy",
    ylabel="Mean sensing redundancy (%)",
    title=(
        "Mean Sensing Redundancy vs PPO Training"
    ),
    filename="05_mean_redundancy.png",
)


# ============================================================
# 6. Learned weights
# ============================================================

weight_columns = [
    "mean_alpha",
    "mean_beta",
    "mean_gamma",
    "mean_delta",
]

if all(
        column in ppo.columns
        for column in weight_columns
):

    weight_means = (
        ppo
        .groupby(
            "checkpoint",
            observed=True,
        )[weight_columns]
        .mean()
        .reindex(CHECKPOINT_ORDER)
    )

    plt.figure(
        figsize=(8, 5)
    )

    x = np.arange(
        len(CHECKPOINT_ORDER)
    )

    plt.plot(
        x,
        weight_means["mean_alpha"],
        marker="o",
        label="alpha — information gain",
    )

    plt.plot(
        x,
        weight_means["mean_beta"],
        marker="o",
        label="beta — cost",
    )

    plt.plot(
        x,
        weight_means["mean_gamma"],
        marker="o",
        label="gamma — redundancy",
    )

    plt.plot(
        x,
        weight_means["mean_delta"],
        marker="o",
        label="delta — cluster size",
    )

    plt.xticks(
        x,
        ["25k_a", "50k_a", "75k_a", "100k_a", "25k_b", "50k_b", "75k_b", "100k_b"],
    )

    plt.xlabel(
        "PPO training timesteps"
    )

    plt.ylabel(
        "Mean learned weight"
    )

    plt.title(
        "Learned Utility Weights"
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "06_learned_weights.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Improvement relative to fixed baseline
# ============================================================

fixed_steps_mean = fixed["steps"].mean()
fixed_distance_mean = fixed["total_distance"].mean()
fixed_redundancy_mean = fixed["final_redundancy"].mean()


improvement_rows = []


for checkpoint in CHECKPOINT_ORDER:

    checkpoint_data = ppo[
        ppo["checkpoint"] == checkpoint
        ]

    ppo_steps = checkpoint_data[
        "steps"
    ].mean()

    ppo_distance = checkpoint_data[
        "total_distance"
    ].mean()

    ppo_redundancy = checkpoint_data[
        "final_redundancy"
    ].mean()

    # Lower is better for steps/distance.
    steps_improvement = (
            (
                    fixed_steps_mean
                    - ppo_steps
            )
            / fixed_steps_mean
            * 100
    )

    distance_improvement = (
            (
                    fixed_distance_mean
                    - ppo_distance
            )
            / fixed_distance_mean
            * 100
    )

    # Lower redundancy is treated as an improvement.
    redundancy_improvement = (
            (
                    fixed_redundancy_mean
                    - ppo_redundancy
            )
            / fixed_redundancy_mean
            * 100
    )

    improvement_rows.append(
        {
            "checkpoint": checkpoint,
            "steps_improvement": steps_improvement,
            "distance_improvement": distance_improvement,
            "redundancy_improvement": redundancy_improvement,
        }
    )


improvement_df = pd.DataFrame(
    improvement_rows
)

print()
print("=" * 70)
print("IMPROVEMENT RELATIVE TO FIXED-WEIGHT BASELINE")
print("=" * 70)

print(
    improvement_df.round(2)
)


# ============================================================
# Plot improvement
# ============================================================

x = np.arange(
    len(CHECKPOINT_ORDER)
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    x,
    improvement_df[
        "steps_improvement"
    ],
    marker="o",
    label="Steps to 90%",
)

plt.plot(
    x,
    improvement_df[
        "distance_improvement"
    ],
    marker="o",
    label="Distance",
)

plt.plot(
    x,
    improvement_df[
        "redundancy_improvement"
    ],
    marker="o",
    label="Redundancy",
)

plt.axhline(
    0,
    linewidth=1,
)

plt.xticks(
    x,
    ["25k_a", "50k_a", "75k_a", "100k_a", "25k_b", "50k_b", "75k_b", "100k_b"],
)

plt.xlabel(
    "PPO training timesteps"
)

plt.ylabel(
    "Improvement over fixed baseline (%)"
)

plt.title(
    "PPO Improvement Relative to Fixed-Weight Baseline"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "07_improvement_vs_fixed.png",
)

plt.savefig(
    output_path,
    dpi=300,
)

plt.close()

print(
    f"Saved: {output_path}"
)


# ============================================================
# Best checkpoint
# ============================================================

checkpoint_means = (
    ppo
    .groupby(
        "checkpoint",
        observed=True,
    )["steps"]
    .mean()
    .reindex(CHECKPOINT_ORDER)
)

best_checkpoint = (
    checkpoint_means.idxmin()
)

best_value = (
    checkpoint_means.min()
)

print()
print("=" * 70)
print("BEST CHECKPOINT")
print("=" * 70)

print(
    f"Best checkpoint by mean T90: "
    f"{best_checkpoint}"
)

print(
    f"Mean T90: "
    f"{best_value:.2f} steps"
)

print()
print(
    "Analysis complete."
)