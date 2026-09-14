import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

from rl.exploration_env import ExplorationEnv


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/ppo/ppo-b/ppo_utility_50k"

SEED = 101
NUM_STEPS = 100

OUTPUT_DIR = "results/ppo"
GIF_PATH = os.path.join(
    OUTPUT_DIR,
    "ppo_weight_adaptation.gif",
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

obs, info = env.reset(seed=SEED)


# ============================================================
# Load trained PPO
# ============================================================

model = PPO.load(MODEL_PATH)


# ============================================================
# Collect PPO actions
# ============================================================

weights = []
coverages = []
rewards = []

for step in range(NUM_STEPS):

    # PPO predicts the four utility weights
    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    action = np.asarray(action).flatten()

    # Store weights
    weights.append(action.copy())

    # Execute action in environment
    obs, reward, terminated, truncated, info = env.step(action)

    rewards.append(float(reward))

    # Try to obtain coverage from info
    coverage = info.get(
        "coverage",
        np.nan,
    )

    coverages.append(float(coverage))

    # Episode ended
    if terminated or truncated:

        print(
            f"Episode ended at step {step + 1}"
        )

        break


weights = np.asarray(weights)

steps = np.arange(
    len(weights)
)


# ============================================================
# Print values
# ============================================================

print()
print("PPO WEIGHT ADAPTATION")
print("=" * 60)

for i in range(len(weights)):

    print(
        f"t={i:3d} | "
        f"alpha={weights[i, 0]:.3f} | "
        f"beta={weights[i, 1]:.3f} | "
        f"gamma={weights[i, 2]:.3f} | "
        f"delta={weights[i, 3]:.3f}"
    )


# ============================================================
# Create animation
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.set_xlim(
    0,
    max(NUM_STEPS - 1, 1),
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_xlabel(
    "Exploration timestep"
)

ax.set_ylabel(
    "Utility weight"
)

ax.set_title(
    "PPO Adaptive Utility Weights"
)

ax.grid(
    True,
    alpha=0.3,
)


# Lines
line_alpha, = ax.plot(
    [],
    [],
    label=r"$\alpha$ — Information Gain",
)

line_beta, = ax.plot(
    [],
    [],
    label=r"$\beta$ — Path Cost",
)

line_gamma, = ax.plot(
    [],
    [],
    label=r"$\gamma$ — Redundancy",
)

line_delta, = ax.plot(
    [],
    [],
    label=r"$\delta$ — Cluster Size",
)


ax.legend(
    loc="upper right"
)


# ============================================================
# Animation function
# ============================================================

def update(frame):

    line_alpha.set_data(
        steps[:frame + 1],
        weights[:frame + 1, 0],
    )

    line_beta.set_data(
        steps[:frame + 1],
        weights[:frame + 1, 1],
    )

    line_gamma.set_data(
        steps[:frame + 1],
        weights[:frame + 1, 2],
    )

    line_delta.set_data(
        steps[:frame + 1],
        weights[:frame + 1, 3],
    )

    ax.set_title(
        f"PPO Adaptive Utility Weights — "
        f"Step {frame + 1}/{len(weights)}"
    )

    return (
        line_alpha,
        line_beta,
        line_gamma,
        line_delta,
    )


# ============================================================
# Generate GIF
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=len(weights),
    interval=80,
    blit=True,
)


animation.save(
    GIF_PATH,
    writer="pillow",
    fps=12,
)


plt.close(fig)


print()
print("=" * 60)
print("GIF SAVED")
print("=" * 60)
print(GIF_PATH)