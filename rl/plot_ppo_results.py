import os

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "results/ppo/training_metrics.csv"
OUTPUT_DIR = "results/ppo"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

df = pd.read_csv(CSV_PATH)


# ============================================================
# EPISODE-LEVEL AGGREGATION
# ============================================================

episode_stats = (
    df.groupby("episode")
    .agg(
        episode_reward=("reward", "sum"),
        final_coverage=("coverage", "last"),
        mean_coverage=("coverage", "mean"),
        final_redundancy=("redundancy", "last"),

        alpha=("alpha", "mean"),
        beta=("beta", "mean"),
        gamma=("gamma", "mean"),
        delta=("delta", "mean"),
    )
    .reset_index()
)


# ============================================================
# 1. EPISODE REWARD
# ============================================================

plt.figure()

plt.plot(
    episode_stats["episode"],
    episode_stats["episode_reward"],
)

plt.xlabel("Episode")
plt.ylabel("Total episode reward")
plt.title("PPO Episode Reward")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/episode_reward.png",
    dpi=300,
)

plt.close()


# ============================================================
# 2. FINAL COVERAGE
# ============================================================

plt.figure()

plt.plot(
    episode_stats["episode"],
    episode_stats["final_coverage"],
)

plt.xlabel("Episode")
plt.ylabel("Final coverage (%)")
plt.title("Final Exploration Coverage per Episode")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/episode_coverage.png",
    dpi=300,
)

plt.close()


# ============================================================
# 3. MEAN PPO WEIGHTS
# ============================================================

plt.figure()

plt.plot(
    episode_stats["episode"],
    episode_stats["alpha"],
    label="alpha",
)

plt.plot(
    episode_stats["episode"],
    episode_stats["beta"],
    label="beta",
)

plt.plot(
    episode_stats["episode"],
    episode_stats["gamma"],
    label="gamma",
)

plt.plot(
    episode_stats["episode"],
    episode_stats["delta"],
    label="delta",
)

plt.xlabel("Episode")
plt.ylabel("Mean weight")
plt.title("PPO Utility Weight Evolution")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/episode_weights.png",
    dpi=300,
)

plt.close()


# ============================================================
# 4. MEAN COVERAGE
# ============================================================

plt.figure()

plt.plot(
    episode_stats["episode"],
    episode_stats["mean_coverage"],
)

plt.xlabel("Episode")
plt.ylabel("Mean coverage (%)")
plt.title("Mean Coverage During Episode")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/mean_episode_coverage.png",
    dpi=300,
)

plt.close()


print("\nGenerated:")
print("  episode_reward.png")
print("  episode_coverage.png")
print("  episode_weights.png")
print("  mean_episode_coverage.png")