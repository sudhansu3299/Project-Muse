import numpy as np

from rl.exploration_env import ExplorationEnv


env = ExplorationEnv(
    grid_width=100,
    grid_height=100,
    num_drones=5,
    obstacle_percentage=0.1,
    max_steps=1000,
)

state, info = env.reset(seed=42)

action = np.array(
    [1.0, 0.5, 0.5, 0.1],
    dtype=np.float32,
)

total_reward = 0.0

for t in range(1000):

    (
        next_state,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    total_reward += reward

    if t % 100 == 0:
        print(
            f"t={t:4d} | "
            f"coverage={info['coverage']:.2f}% | "
            f"redundancy={info['redundancy']:.2f}% | "
            f"reward={reward:.3f}"
        )

    state = next_state

    if terminated or truncated:
        break


print("\n========== FIXED WEIGHT TEST ==========")
print("Steps:", t + 1)
print(
    "Final coverage:",
    info["coverage"],
)
print(
    "Final redundancy:",
    info["redundancy"],
)
print(
    "Total reward:",
    total_reward,
)