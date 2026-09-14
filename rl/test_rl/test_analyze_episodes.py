import pandas as pd

df = pd.read_csv(
    "results/ppo/training_metrics.csv"
)

summary = (
    df.groupby("episode")
    .agg(
        steps=("timestep", "count"),

        final_coverage=("coverage", "last"),

        total_reward=("reward", "sum"),

        mean_assigned_drones=(
            "assigned_drones",
            "mean",
        ),

        mean_no_centroid_path=(
            "no_centroid_path",
            "mean",
        ),

        mean_no_frontier_path=(
            "no_frontier_path",
            "mean",
        ),
    )
)

print(
    summary[
        [
            "steps",
            "final_coverage",
            "mean_assigned_drones",
            "mean_no_centroid_path",
            "mean_no_frontier_path",
        ]
    ]
)