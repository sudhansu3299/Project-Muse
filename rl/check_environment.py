from gymnasium.utils.env_checker import check_env

from rl.exploration_env import ExplorationEnv


env = ExplorationEnv(
    max_steps=100,
)

check_env(env)

print("Environment check passed.")