from stable_baselines3 import PPO

from rl.exploration_env import ExplorationEnv
from rl.ppo_callback import PPOMetricsCallback


env = ExplorationEnv(
    grid_width=100,
    grid_height=100,
    num_drones=5,
    obstacle_percentage=0.1,
    max_steps=1000,
)


callback = PPOMetricsCallback(
    log_path="results/ppo/training_metrics.csv"
)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)


model.learn(
    total_timesteps=5000,
    callback=callback,
)


model.save(
    "models/ppo_utility_25k"
)