# rl/evaluate.py

import os

import numpy as np
import pandas as pd

from rl.exploration_env import ExplorationEnv


# ============================================================
# Configuration
# ============================================================

GRID_WIDTH = 100
GRID_HEIGHT = 100

NUM_DRONES = 5

OBSTACLE_PERCENTAGE = 0.20

COMMUNICATION_RADIUS = 10

MAX_STEPS = 5000

TARGET_COVERAGE = 90.0

# Seeds used ONLY for evaluation.
# These should NOT be used while training PPO.
TEST_SEEDS = [
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115
]

# Your current hand-tuned baseline
FIXED_WEIGHTS = np.array(
    [1.0, 0.5, 0.5, 0.1],
    dtype=np.float32,
)

OUTPUT_DIR = "results/baseline"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "fixed_weight_results.csv",
)


# ============================================================
# Environment
# ============================================================

def create_environment():

    return ExplorationEnv(
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        num_drones=NUM_DRONES,
        obstacle_percentage=OBSTACLE_PERCENTAGE,
        communication_radius=COMMUNICATION_RADIUS,
        max_steps=MAX_STEPS,
    )


# ============================================================
# Fixed-weight policy
# ============================================================

def fixed_weight_policy(state):
    """
    Fixed Hungarian utility weights.

    PPO is NOT used here.

    The returned vector is interpreted by
    ExplorationEnv.step() as:

        action[0] -> alpha
        action[1] -> beta
        action[2] -> gamma
        action[3] -> delta
    """

    return FIXED_WEIGHTS.copy()


# ============================================================
# Single episode evaluation
# ============================================================

def evaluate_episode(
        env,
        action_provider,
        seed,
        target_coverage=TARGET_COVERAGE,
        max_steps=MAX_STEPS,
):
    """
    Evaluate one policy on one map.

    The environment is reset using the supplied seed.

    Returns a dictionary containing the metrics
    needed for comparison against PPO.
    """

    state, _ = env.reset(
        seed=seed
    )

    total_reward = 0.0

    cumulative_redundancy = 0.0

    cumulative_distance = 0.0

    coverage_history = []

    redundancy_history = []

    reward_history = []

    reached_target = False

    steps_taken = 0

    for step in range(max_steps):

        # ----------------------------------------------------
        # Get action from policy
        # ----------------------------------------------------

        action = action_provider(state)

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        total_reward += float(reward)

        current_coverage = float(
            info["coverage"]
        )

        current_distance = float(
            info["distance"]
        )

        current_redundancy = float(
            info["redundancy"]
        )

        coverage_history.append(
            current_coverage
        )

        redundancy_history.append(
            current_redundancy
        )

        reward_history.append(
            float(reward)
        )

        cumulative_redundancy += (
            current_redundancy
        )

        # Distance is already cumulative in
        # Simulator.get_total_distance(), so
        # we don't sum it here.

        steps_taken = step + 1

        state = next_state

        # ----------------------------------------------------
        # Check target coverage
        # ----------------------------------------------------

        if current_coverage >= target_coverage:

            reached_target = True

            break

        # ----------------------------------------------------
        # Check environment termination
        # ----------------------------------------------------

        if terminated or truncated:
            break

    # ========================================================
    # Final metrics
    # ========================================================

    final_coverage = float(
        info["coverage"]
    )

    final_distance = float(
        info["distance"]
    )

    final_redundancy = float(
        info["redundancy"]
    )

    # Average redundancy experienced throughout
    # the trajectory.
    mean_redundancy = float(
        np.mean(redundancy_history)
    ) if redundancy_history else 0.0

    # Maximum redundancy encountered.
    max_redundancy = float(
        np.max(redundancy_history)
    ) if redundancy_history else 0.0

    # Average reward per timestep.
    mean_reward = (
        total_reward / steps_taken
        if steps_taken > 0
        else 0.0
    )

    return {
        "seed": seed,

        "steps": steps_taken,

        "reached_target": reached_target,

        "final_coverage": final_coverage,

        "total_distance": final_distance,

        "final_redundancy": final_redundancy,

        "mean_redundancy": mean_redundancy,

        "max_redundancy": max_redundancy,

        "cumulative_redundancy": (
            cumulative_redundancy
        ),

        "total_reward": total_reward,

        "mean_reward": mean_reward,

        # Explicitly store the fixed weights
        # so the experiment is self-documenting.
        "alpha": float(FIXED_WEIGHTS[0]),
        "beta": float(FIXED_WEIGHTS[1]),
        "gamma": float(FIXED_WEIGHTS[2]),
        "delta": float(FIXED_WEIGHTS[3]),
    }


# ============================================================
# Evaluate all seeds
# ============================================================

def run_fixed_weight_baseline():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    env = create_environment()

    results = []

    print()
    print(
        "=============================================="
    )
    print(
        "       FIXED-WEIGHT BASELINE EVALUATION"
    )
    print(
        "=============================================="
    )

    print(
        f"Weights: "
        f"alpha={FIXED_WEIGHTS[0]:.2f}, "
        f"beta={FIXED_WEIGHTS[1]:.2f}, "
        f"gamma={FIXED_WEIGHTS[2]:.2f}, "
        f"delta={FIXED_WEIGHTS[3]:.2f}"
    )

    print(
        f"Target coverage: "
        f"{TARGET_COVERAGE:.1f}%"
    )

    print(
        f"Max steps: {MAX_STEPS}"
    )

    print(
        f"Evaluation seeds: {TEST_SEEDS}"
    )

    print()

    # --------------------------------------------------------
    # Run each seed
    # --------------------------------------------------------

    for seed in TEST_SEEDS:

        print(
            f"Running seed {seed}..."
        )

        result = evaluate_episode(
            env=env,
            action_provider=fixed_weight_policy,
            seed=seed,
            target_coverage=TARGET_COVERAGE,
            max_steps=MAX_STEPS,
        )

        results.append(result)

        print(
            f"  Steps: "
            f"{result['steps']}"
        )

        print(
            f"  Final coverage: "
            f"{result['final_coverage']:.2f}%"
        )

        print(
            f"  Distance: "
            f"{result['total_distance']:.2f}"
        )

        print(
            f"  Final redundancy: "
            f"{result['final_redundancy']:.2f}%"
        )

        print(
            f"  Total reward: "
            f"{result['total_reward']:.3f}"
        )

        print(
            f"  Reached 90%: "
            f"{result['reached_target']}"
        )

        print()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "=============================================="
    )
    print(
        "              BASELINE SUMMARY"
    )
    print(
        "=============================================="
    )

    print()

    print(
        df[
            [
                "seed",
                "steps",
                "final_coverage",
                "total_distance",
                "final_redundancy",
                "total_reward",
                "reached_target",
            ]
        ].to_string(index=False)
    )

    print()

    # --------------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------------

    print(
        "--------------- MEAN ± STD ----------------"
    )

    metrics = [
        "steps",
        "final_coverage",
        "total_distance",
        "final_redundancy",
        "mean_redundancy",
        "cumulative_redundancy",
        "total_reward",
    ]

    for metric in metrics:

        mean = df[metric].mean()

        std = df[metric].std()

        print(
            f"{metric:25s}: "
            f"{mean:.3f} ± {std:.3f}"
        )

    print()

    success_rate = (
            df["reached_target"].mean()
            * 100.0
    )

    print(
        f"90% success rate: "
        f"{success_rate:.1f}%"
    )

    print()

    print(
        f"Saved results to:"
    )

    print(
        OUTPUT_FILE
    )

    return df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_fixed_weight_baseline()