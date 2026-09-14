import numpy as np

from rl.test_rl.test_exploration_env import ExplorationEnv


env = ExplorationEnv(
    grid_width=100,
    grid_height=100,
    num_drones=5,
    obstacle_percentage=0.1,
    max_steps=1000,
)

state = env.reset(map_seed=42)

trajectory = []

for t in range(1000):

    # Fixed utility weights for now
    action = np.array(
        [1.0, 0.5, 0.5, 0.1],
        dtype=np.float32,
    )

    next_state, reward, terminated, truncated, info = (
        env.step(action)
    )

    trajectory.append({
        "timestep": t,
        "state": state.copy(),
        "action": action.copy(),
        "reward": reward,
        "next_state": next_state.copy(),
    })

    state = next_state

    if terminated or truncated:
        break


print("\n========== EPISODE ==========")
print("Steps:", len(trajectory))
print("Total reward:", sum(
    step["reward"] for step in trajectory
))

print("\nFirst 5 transitions:")

for transition in trajectory[:5]:

    print(
        f"t={transition['timestep']}, "
        f"reward={transition['reward']:.4f}, "
        f"action={transition['action']}"
    )

def calculate_returns(rewards, gamma=0.99):

    returns = [0.0] * len(rewards)

    G = 0.0

    for t in reversed(range(len(rewards))):

        G = rewards[t] + gamma * G

        returns[t] = G

    return returns

rewards = [
    step["reward"]
    for step in trajectory
]

returns = calculate_returns(
    rewards,
    gamma=0.99,
)

for t in range(min(10, len(returns))):

    print(
        f"t={t}, "
        f"reward={rewards[t]:.3f}, "
        f"G={returns[t]:.3f}"
    )