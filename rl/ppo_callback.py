import csv
import os

from stable_baselines3.common.callbacks import BaseCallback


class PPOMetricsCallback(BaseCallback):

    def __init__(
            self,
            log_path="results/ppo/training_metrics.csv",
            verbose=0,
    ):
        super().__init__(verbose)

        self.log_path = log_path

        self.episode = 0
        self.episode_rewards = []

        os.makedirs(
            os.path.dirname(log_path),
            exist_ok=True,
        )

        self.file = None
        self.writer = None

    def _on_training_start(self):

        self.file = open(
            self.log_path,
            "w",
            newline="",
        )

        self.writer = csv.writer(
            self.file
        )

        self.writer.writerow([
            "timestep",
            "episode",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "reward",
            "coverage",
            "distance",
            "redundancy",
            "assigned_drones",
            "no_centroid_path",
            "no_frontier_path",
            "delta_coverage",
            "delta_distance",
            "delta_redundancy",
            "coverage_reward",
            "distance_penalty",
            "redundancy_penalty",
        ])

    def _on_step(self):

        # SB3 stores information from env.step()
        infos = self.locals.get("infos")

        if infos is None:
            return True

        # We currently have one environment,
        # so take the first info dictionary.
        info = infos[0]

        reward = self.locals["rewards"][0]

        self.writer.writerow([
            self.num_timesteps,

            self.episode,

            info.get("alpha"),
            info.get("beta"),
            info.get("gamma"),
            info.get("delta"),

            float(reward),

            info.get("coverage"),
            info.get("distance"),
            info.get("redundancy"),
            info.get("assigned_drones"),

            info.get("no_centroid_path"),
            info.get("no_frontier_path"),

            info.get("delta_coverage"),
            info.get("delta_distance"),
            info.get("delta_redundancy"),

            info.get("coverage_reward"),
            info.get("distance_penalty"),
            info.get("redundancy_penalty"),
        ])

        self.file.flush()

        # Detect episode termination
        dones = self.locals.get("dones")

        if dones is not None and dones[0]:
            self.episode += 1

        return True

    def _on_training_end(self):

        if self.file is not None:
            self.file.close()