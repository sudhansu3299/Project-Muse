import os

from stable_baselines3 import PPO

from rl.exploration_env import ExplorationEnv
from rl.ppo_callback import PPOMetricsCallback


# ============================================================
# Configuration
# ============================================================

TOTAL_TIMESTEPS = 100_000

CHECKPOINTS = [
    25_000,
    50_000,
    75_000,
    100_000,
]

# ------------------------------------------------------------
# PPO-B experiment
# Training environments: seeds 1–101
# ------------------------------------------------------------

RUN_NAME = "ppo-b"

MODEL_DIR = (
    f"models/ppo/{RUN_NAME}"
)

RESULTS_DIR = (
    f"results/ppo/{RUN_NAME}"
)

METRICS_PATH = (
    f"{RESULTS_DIR}/training_metrics.csv"
)

TRAINING_SEEDS = list(
    range(1, 102)
)


# ============================================================
# Directories
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# Environment
# ============================================================

env = ExplorationEnv(
    grid_width=100,
    grid_height=100,
    num_drones=5,
    obstacle_percentage=0.1,
    max_steps=5000,
)

env.set_training_seeds(
    TRAINING_SEEDS
)


# ============================================================
# PPO metrics callback
# ============================================================

callback = PPOMetricsCallback(
    log_path=METRICS_PATH
)


# ============================================================
# PPO model
# ============================================================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)


# ============================================================
# Train in checkpoint intervals
# ============================================================

previous_checkpoint = 0

for checkpoint in CHECKPOINTS:

    timesteps = (
            checkpoint
            - previous_checkpoint
    )

    print()
    print(
        "========================================"
    )
    print(
        f"PPO-B: Training to "
        f"{checkpoint:,} timesteps"
    )
    print(
        "Training seeds: 1–101"
    )
    print(
        "========================================"
    )

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        reset_num_timesteps=False,
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"ppo_utility_{checkpoint // 1000}k",
    )

    model.save(
        model_path
    )

    print()
    print(
        "Saved PPO-B checkpoint:"
    )

    print(
        model_path
    )

    previous_checkpoint = checkpoint


# ============================================================
# Done
# ============================================================

print()
print(
    "========================================"
)
print(
    "PPO-B TRAINING COMPLETE"
)
print(
    "========================================"
)

print(
    "Training seeds: 1–101"
)

print(
    "Saved checkpoints:"
)

for checkpoint in CHECKPOINTS:

    print(
        f"  {MODEL_DIR}/"
        f"ppo_utility_{checkpoint // 1000}k.zip"
    )

print()
print(
    f"Metrics:"
)

print(
    METRICS_PATH
)