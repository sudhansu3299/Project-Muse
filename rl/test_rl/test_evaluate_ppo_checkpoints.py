import os

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from rl.exploration_env import ExplorationEnv


# ============================================================
# Configuration
# ============================================================

CHECKPOINTS = {
    "ppo_a_25k": "models/ppo/ppo-a/ppo_utility_25k.zip",
    "ppo_a_50k": "models/ppo/ppo-a/ppo_utility_50k.zip",
    "ppo_a_75k": "models/ppo/ppo-a/ppo_utility_75k.zip",
    "ppo_a_100k": "models/ppo/ppo-a/ppo_utility_100k.zip",

    "ppo_b_25k": "models/ppo/ppo-b/ppo_utility_25k.zip",
    "ppo_b_50k": "models/ppo/ppo-b/ppo_utility_50k.zip",
    "ppo_b_75k": "models/ppo/ppo-b/ppo_utility_75k.zip",
    "ppo_b_100k": "models/ppo/ppo-b/ppo_utility_100k.zip",
}

EVALUATION_SEEDS = list(range(105, 115))

MAX_STEPS = 3000

TARGET_COVERAGE = 90.0

OUTPUT_DIR = "results/ppo/evaluation"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ppo_all_checkpoint_trajectories.csv",
)


# ============================================================
# Create environment
# ============================================================

def create_environment():

    return ExplorationEnv(
        grid_width=100,
        grid_height=100,
        num_drones=5,
        obstacle_percentage=0.1,
        max_steps=MAX_STEPS,
    )


# ============================================================
# Evaluate one PPO checkpoint on one seed
# ============================================================

def evaluate_checkpoint_on_seed(
        model,
        env,
        seed,
):

    # --------------------------------------------------------
    # Reset to a deterministic unseen map
    # --------------------------------------------------------

    state, _ = env.reset(
        seed=seed
    )

    total_reward = 0.0

    steps = 0

    reached_target = False

    coverage_history = []

    redundancy_history = []

    distance_history = []

    weight_history = []

    # --------------------------------------------------------
    # Run trajectory
    # --------------------------------------------------------

    for step in range(MAX_STEPS):

        # IMPORTANT:
        # deterministic=True means we use the learned
        # policy mean/action rather than sampling randomly.
        action, _ = model.predict(
            state,
            deterministic=True,
        )

        action = np.asarray(
            action,
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
        # Record metrics
        # ----------------------------------------------------

        total_reward += float(reward)

        steps = step + 1

        coverage = float(
            info["coverage"]
        )

        redundancy = float(
            info["redundancy"]
        )

        distance = float(
            info["distance"]
        )

        coverage_history.append(
            coverage
        )

        redundancy_history.append(
            redundancy
        )

        distance_history.append(
            distance
        )

        weight_history.append(
            {
                "alpha": float(action[0]),
                "beta": float(action[1]),
                "gamma": float(action[2]),
                "delta": float(action[3]),
            }
        )

        state = next_state

        # ----------------------------------------------------
        # Target reached
        # ----------------------------------------------------

        if coverage >= TARGET_COVERAGE:

            reached_target = True

            break

        if terminated or truncated:
            break

    # ========================================================
    # Final metrics
    # ========================================================

    final_coverage = coverage_history[-1]

    final_redundancy = redundancy_history[-1]

    final_distance = distance_history[-1]

    mean_redundancy = float(
        np.mean(redundancy_history)
    )

    mean_distance_per_step = float(
        final_distance / steps
    ) if steps > 0 else 0.0

    mean_reward = (
        total_reward / steps
        if steps > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Average PPO weights over trajectory
    # --------------------------------------------------------

    mean_alpha = float(
        np.mean(
            [w["alpha"] for w in weight_history]
        )
    )

    mean_beta = float(
        np.mean(
            [w["beta"] for w in weight_history]
        )
    )

    mean_gamma = float(
        np.mean(
            [w["gamma"] for w in weight_history]
        )
    )

    mean_delta = float(
        np.mean(
            [w["delta"] for w in weight_history]
        )
    )

    return {
        "seed": seed,

        "steps": steps,

        "reached_90": reached_target,

        "final_coverage": final_coverage,

        "total_distance": final_distance,

        "final_redundancy": final_redundancy,

        "mean_redundancy": mean_redundancy,

        "total_reward": total_reward,

        "mean_reward": mean_reward,

        "mean_alpha": mean_alpha,

        "mean_beta": mean_beta,

        "mean_gamma": mean_gamma,

        "mean_delta": mean_delta,
    }


# ============================================================
# Evaluate one checkpoint
# ============================================================

def evaluate_checkpoint(
        checkpoint_name,
        checkpoint_path,
):

    print()
    print(
        "================================================"
    )
    print(
        f"EVALUATING {checkpoint_name}"
    )
    print(
        "================================================"
    )

    print(
        f"Model: {checkpoint_path}"
    )

    # --------------------------------------------------------
    # Load frozen PPO policy
    # --------------------------------------------------------

    model = PPO.load(
        checkpoint_path
    )

    # --------------------------------------------------------
    # Create fresh environment
    # --------------------------------------------------------

    env = create_environment()

    results = []

    # --------------------------------------------------------
    # Same evaluation seeds for every checkpoint
    # --------------------------------------------------------

    for seed in EVALUATION_SEEDS:

        print()
        print(
            f"Seed {seed}"
        )

        result = evaluate_checkpoint_on_seed(
            model=model,
            env=env,
            seed=seed,
        )

        result["checkpoint"] = checkpoint_name

        results.append(result)

        print(
            f"  Steps: "
            f"{result['steps']}"
        )

        print(
            f"  Coverage: "
            f"{result['final_coverage']:.2f}%"
        )

        print(
            f"  Distance: "
            f"{result['total_distance']:.2f}"
        )

        print(
            f"  Redundancy: "
            f"{result['final_redundancy']:.2f}%"
        )

        print(
            f"  Reward: "
            f"{result['total_reward']:.3f}"
        )

        print(
            f"  Reached 90%: "
            f"{result['reached_90']}"
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    all_results = []

    # --------------------------------------------------------
    # Evaluate 25k and 50k
    # --------------------------------------------------------

    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():

        results = evaluate_checkpoint(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
        )

        all_results.extend(
            results
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "================================================"
    )
    print(
        "PPO CHECKPOINT COMPARISON"
    )
    print(
        "================================================"
    )

    summary = (
        df
        .groupby("checkpoint")
        [
            [
                "steps",
                "final_coverage",
                "total_distance",
                "final_redundancy",
                "mean_redundancy",
                "total_reward",
                "mean_alpha",
                "mean_beta",
                "mean_gamma",
                "mean_delta",
            ]
        ]
        .agg(["mean", "std"])
    )

    print(
        summary.to_string()
    )

    # --------------------------------------------------------
    # Success rate
    # --------------------------------------------------------

    success_rate = (
            df
            .groupby("checkpoint")["reached_90"]
            .mean()
            * 100.0
    )

    print()
    print(
        "90% success rate:"
    )

    print(
        success_rate
    )

    print()
    print(
        f"Saved results to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()