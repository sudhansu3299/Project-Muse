import numpy as np

from rl.exploration_env import ExplorationEnv


env = ExplorationEnv(
    grid_width=100,
    grid_height=100,
    num_drones=5,
    obstacle_percentage=0.1,
    max_steps=1000,
)

for episode in range(3):

    print(
        f"\n\n"
        f"========== EPISODE {episode} =========="
    )

    state, info = env.reset(seed=42)

    total_reward = 0.0

    for t in range(1000):

        # Fixed weights — NOT PPO
        action = np.array(
            [1.0, 0.5, 0.5, 0.1],
            dtype=np.float32,
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        # Print every 100 steps
        if t % 100 == 0 or terminated or truncated:

            print(
                f"t={t:4d} | "
                f"coverage={info['coverage']:.2f}% | "
                f"redundancy={info['redundancy']:.2f}% | "
                f"assigned={info['assigned_drones']} | "
                f"reward={reward:.3f}"
            )

        state = next_state

        if terminated or truncated:
            break

    print(
        f"\nEpisode {episode} SUMMARY:"
    )

    print(
        f"  Steps: {t + 1}"
    )

    print(
        f"  Final coverage: "
        f"{info['coverage']:.2f}%"
    )

    print(
        f"  Final redundancy: "
        f"{info['redundancy']:.2f}%"
    )

    print(
        f"  Total reward: "
        f"{total_reward:.3f}"
    )

    print(
        f"  Last assigned drones: "
        f"{info['assigned_drones']}"
    )