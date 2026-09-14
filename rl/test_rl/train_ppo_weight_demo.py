import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "results/ppo/weight_demo/weight_trajectory.csv"

GIF_PATH = "results/ppo/weight_demo/ppo_weight_learning.gif"

TOTAL_TIMESTEPS = 10_000

# Show one trajectory point every N timesteps
PLOT_INTERVAL = 100


# ============================================================
# Load trajectory
# ============================================================

df = pd.read_csv(CSV_PATH)

steps = df["timestep"].to_numpy()

alpha = df["alpha"].to_numpy()
beta = df["beta"].to_numpy()
gamma = df["gamma"].to_numpy()
delta = df["delta"].to_numpy()


# ============================================================
# Downsample
# ============================================================

target_steps = np.arange(
    0,
    TOTAL_TIMESTEPS + 1,
    PLOT_INTERVAL,
    )

indices = np.searchsorted(
    steps,
    target_steps,
)

indices = indices[
    indices < len(steps)
    ]

steps_plot = steps[indices]

alpha_plot = alpha[indices]
beta_plot = beta[indices]
gamma_plot = gamma[indices]
delta_plot = delta[indices]


print(
    f"Original points: {len(steps):,}"
)

print(
    f"Plot points: {len(steps_plot):,}"
)


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.set_xlim(
    0,
    TOTAL_TIMESTEPS,
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_xlabel(
    "Training timestep"
)

ax.set_ylabel(
    "Utility weight"
)

ax.set_title(
    "PPO Learning: Adaptive Utility Weights"
)

ax.grid(
    True,
    alpha=0.3,
)


# ============================================================
# Lines
# ============================================================

line_alpha, = ax.plot(
    [],
    [],
    label="α — Information Gain",
)

line_beta, = ax.plot(
    [],
    [],
    label="β — Path Cost",
)

line_gamma, = ax.plot(
    [],
    [],
    label="γ — Redundancy",
)

line_delta, = ax.plot(
    [],
    [],
    label="δ — Cluster Size",
)


ax.legend(
    loc="upper right"
)


# ============================================================
# Animation
# ============================================================

def update(frame):

    line_alpha.set_data(
        steps_plot[:frame + 1],
        alpha_plot[:frame + 1],
    )

    line_beta.set_data(
        steps_plot[:frame + 1],
        beta_plot[:frame + 1],
    )

    line_gamma.set_data(
        steps_plot[:frame + 1],
        gamma_plot[:frame + 1],
    )

    line_delta.set_data(
        steps_plot[:frame + 1],
        delta_plot[:frame + 1],
    )

    ax.set_title(
        "PPO Learning: Adaptive Utility Weights "
        f"(t = {steps_plot[frame]:,})"
    )

    return (
        line_alpha,
        line_beta,
        line_gamma,
        line_delta,
    )


animation = FuncAnimation(
    fig,
    update,
    frames=len(steps_plot),
    interval=100,
    blit=True,
)


# ============================================================
# Save GIF
# ============================================================

print()
print("Creating GIF...")

animation.save(
    GIF_PATH,
    writer="pillow",
    fps=10,
)

plt.close(fig)

print()
print("DONE")
print(
    f"GIF: {GIF_PATH}"
)