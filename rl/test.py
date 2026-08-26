import pandas as pd

df = pd.read_csv(
    "results/ppo/training_metrics.csv"
)

for episode in [0, 1, 2]:

    ep = df[df["episode"] == episode]

    print(f"\n========== EPISODE {episode} ==========")

    print("\nFirst 10 actions:")

    print(
        ep[
            [
                "timestep",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "reward",
                "coverage",
                "redundancy",
                "assigned_drones"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nMean weights:")

    print(
        ep[
            [
                "alpha",
                "beta",
                "gamma",
                "delta",
            ]
        ].mean()
    )

    print("\nFinal weights:")

    print(
        ep[
            [
                "alpha",
                "beta",
                "gamma",
                "delta",
            ]
        ].iloc[-1]
    )