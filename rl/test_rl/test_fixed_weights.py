import csv
import os
import numpy as np

from rl.exploration_env import ExplorationEnv


# ============================================================
# Configuration
# ============================================================

EVALUATION_SEEDS = list(range(115, 116))

MAX_STEPS = 3000

TARGET_COVERAGE = 90.0

# Your fixed baseline weights
FIXED_WEIGHTS = {
    "alpha": 1.0,
    "beta": 0.5,
    "gamma": 0.5,
    "delta": 0.1,
}

OUTPUT_DIR = "results/baseline"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "fixed_weight_baseline_new.csv",
)


# ============================================================
# Directories
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Results
# ============================================================

results = []


# ============================================================
# Evaluate every seed
# ============================================================

for seed in EVALUATION_SEEDS:

    print()
    print("========================================")
    print(f"FIXED WEIGHT BASELINE — SEED {seed}")
    print("========================================")

    env = ExplorationEnv(
        grid_width=100,
        grid_height=100,
        num_drones=5,
        obstacle_percentage=0.1,
        max_steps=MAX_STEPS,
    )

    # --------------------------------------------------------
    # Reset with evaluation seed
    # --------------------------------------------------------

    state, info = env.reset(
        seed=seed
    )

    total_reward = 0.0

    reached_90 = False

    # --------------------------------------------------------
    # Per-step metrics
    #
    # These allow the baseline to produce the same type of
    # aggregate metrics used by the learned/PPO runs.
    # --------------------------------------------------------

    reward_history = []

    redundancy_history = []

    # --------------------------------------------------------
    # Run episode
    # --------------------------------------------------------

    for t in range(MAX_STEPS):

        action = np.array(
            [
                FIXED_WEIGHTS["alpha"],
                FIXED_WEIGHTS["beta"],
                FIXED_WEIGHTS["gamma"],
                FIXED_WEIGHTS["delta"],
            ],
            dtype=np.float32,
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # ----------------------------------------------------
        # Record per-step metrics
        # ----------------------------------------------------

        total_reward += reward

        reward_history.append(float(reward))

        redundancy_history.append(
            float(info["redundancy"])
        )

        state = next_state

        # ----------------------------------------------------
        # Check success
        # ----------------------------------------------------

        if info["coverage"] >= TARGET_COVERAGE:

            reached_90 = True

        # ----------------------------------------------------
        # Periodic logging
        # ----------------------------------------------------

        if t % 100 == 0:

            print(
                f"t={t:4d} | "
                f"coverage={info['coverage']:.2f}% | "
                f"redundancy={info['redundancy']:.2f}% | "
                f"distance={info.get('distance', 0):.2f} | "
                f"reward={reward:.3f}"
            )

        # ----------------------------------------------------
        # Episode finished
        # ----------------------------------------------------

        if terminated or truncated:
            break

    # --------------------------------------------------------
    # Number of steps
    # --------------------------------------------------------

    steps = t + 1

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    final_coverage = info["coverage"]

    final_redundancy = info["redundancy"]

    total_distance = info["distance"]

    # --------------------------------------------------------
    # Mean metrics
    #
    # These are calculated across the actual steps taken in
    # the episode, not across MAX_STEPS.
    # --------------------------------------------------------

    mean_redundancy = np.mean(
        redundancy_history
    )

    mean_reward = np.mean(
        reward_history
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print(
        f"Seed {seed}"
    )

    print(
        f"  Steps:           {steps}"
    )

    print(
        f"  Coverage:        {final_coverage:.2f}%"
    )

    print(
        f"  Distance:        {total_distance:.2f}"
    )

    print(
        f"  Final redundancy:{final_redundancy:.2f}%"
    )

    print(
        f"  Mean redundancy: {mean_redundancy:.2f}%"
    )

    print(
        f"  Total reward:    {total_reward:.3f}"
    )

    print(
        f"  Mean reward:     {mean_reward:.3f}"
    )

    print(
        f"  Reached 90%:     {reached_90}"
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        {
            "seed": seed,
            "method": "fixed",
            "checkpoint": "baseline",
            "steps": steps,
            "reached_90": reached_90,
            "final_coverage": final_coverage,
            "total_distance": total_distance,

            # Final metrics
            "final_redundancy": final_redundancy,
            "total_reward": total_reward,

            # Mean metrics for comparison with PPO
            "mean_redundancy": mean_redundancy,
            "mean_reward": mean_reward,

            # Fixed weights
            "alpha": FIXED_WEIGHTS["alpha"],
            "beta": FIXED_WEIGHTS["beta"],
            "gamma": FIXED_WEIGHTS["gamma"],
            "delta": FIXED_WEIGHTS["delta"],
        }
    )

    env.close()


# ============================================================
# Save CSV
# ============================================================

fieldnames = [
    "seed",
    "method",
    "checkpoint",
    "steps",
    "reached_90",
    "final_coverage",
    "total_distance",

    # Final metrics
    "final_redundancy",
    "total_reward",

    # Mean metrics
    "mean_redundancy",
    "mean_reward",

    # Fixed weights
    "alpha",
    "beta",
    "gamma",
    "delta",
]


with open(
        OUTPUT_FILE,
        "w",
        newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# Final summary
# ============================================================

print()
print("========================================")
print("FIXED-WEIGHT BASELINE COMPLETE")
print("========================================")

print(
    f"Seeds: {EVALUATION_SEEDS[0]}–"
    f"{EVALUATION_SEEDS[-1]}"
)

print(
    f"Successful runs: "
    f"{sum(r['reached_90'] for r in results)}/"
    f"{len(results)}"
)

print(
    f"Mean steps: "
    f"{np.mean([r['steps'] for r in results]):.2f}"
)

print(
    f"Mean distance: "
    f"{np.mean([r['total_distance'] for r in results]):.2f}"
)

print(
    f"Mean final redundancy: "
    f"{np.mean([r['final_redundancy'] for r in results]):.2f}%"
)

print(
    f"Mean redundancy: "
    f"{np.mean([r['mean_redundancy'] for r in results]):.2f}%"
)

print(
    f"Mean total reward: "
    f"{np.mean([r['total_reward'] for r in results]):.3f}"
)

print(
    f"Mean reward: "
    f"{np.mean([r['mean_reward'] for r in results]):.3f}"
)

print()
print(
    "Saved results to:"
)

print(
    OUTPUT_FILE
)