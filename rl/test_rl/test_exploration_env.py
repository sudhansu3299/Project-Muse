import numpy as np
import gymnasium as gym

from environment.simulator import Simulator
from environment.coordination.hungarian_frontier_assigner import (
    HungarianFrontierAssigner,
)
from environment.planner.bfs_planner import BFSPlanner
from environment.utils.frontier_utility import FrontierUtility
from strategy.frontier_strategy import FrontierStrategy


class ExplorationEnv:

    def __init__(
            self,
            grid_width=100,
            grid_height=100,
            num_drones=5,
            obstacle_percentage=0.1,
            communication_radius=10,
            max_steps=10000,
    ):

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_drones = num_drones
        self.obstacle_percentage = obstacle_percentage
        self.communication_radius = communication_radius
        self.max_steps = max_steps

        # --------------------------------------------------
        # Utility
        # --------------------------------------------------

        self.utility = FrontierUtility(
            alpha=1.0,
            beta=0.5,
            gamma=0.5,
            delta=0.1,
        )

        # --------------------------------------------------
        # Planner
        # --------------------------------------------------

        self.planner = BFSPlanner()

        # --------------------------------------------------
        # Hungarian assignment
        # --------------------------------------------------

        self.frontier_assigner = HungarianFrontierAssigner(
            planner=self.planner,
            utility=self.utility,
        )

        # --------------------------------------------------
        # Strategy
        # --------------------------------------------------

        self.strategy = FrontierStrategy(
            self.frontier_assigner
        )

        # Simulator will be created in reset()
        self.simulator = None

        # Previous metrics
        self.previous_coverage = 0.0
        self.previous_distance = 0.0
        self.previous_redundancy = 0.0

    def reset(self, map_seed=None):

        self.simulator = Simulator(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_drones=self.num_drones,
            obstacle_percentage=self.obstacle_percentage,
            strategy=self.strategy,
            communication_radius=self.communication_radius,
            map_seed=map_seed,
        )

        self.previous_coverage = (
            self.simulator.get_coverage()
        )

        self.previous_distance = (
            self.simulator.get_total_distance()
        )

        self.previous_redundancy = (
            self.simulator.get_sensing_redundancy()
        )

        return self._get_state()

    def _get_state(self):

        coverage = (
                self.simulator.get_coverage()
                / 100.0
        )

        sensing_redundancy = (
                self.simulator.get_sensing_redundancy()
                / 100.0
        )

        visit_overlap = (
                self.simulator.get_visit_overlap_percentage()
                / 100.0
        )

        movement_efficiency = (
            self.simulator.get_movement_efficiency()
        )

        pairwise_distance = (
            self.simulator.get_mean_pairwise_distance()
        )


        return np.array(
            [
                coverage,
                sensing_redundancy,
                visit_overlap,
                movement_efficiency,
                pairwise_distance,
            ],
            dtype=np.float32,
        )

    def step(self, action):

        # --------------------------------------------------
        # 1. Extract utility weights from PPO action
        # --------------------------------------------------

        alpha = float(action[0])
        beta = float(action[1])
        gamma = float(action[2])
        delta = float(action[3])

        # --------------------------------------------------
        # 2. Update utility
        # --------------------------------------------------

        self.utility.set_weights(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        # --------------------------------------------------
        # 3. Save previous metrics
        # --------------------------------------------------

        previous_coverage = (
            self.simulator.get_coverage()
        )

        previous_distance = (
            self.simulator.get_total_distance()
        )

        previous_redundancy = (
            self.simulator.get_sensing_redundancy()
        )

        # --------------------------------------------------
        # 4. Execute ONE simulator timestep
        # --------------------------------------------------

        self.simulator.step()

        # --------------------------------------------------
        # 5. Get new metrics
        # --------------------------------------------------

        current_coverage = (
            self.simulator.get_coverage()
        )

        current_distance = (
            self.simulator.get_total_distance()
        )

        current_redundancy = (
            self.simulator.get_sensing_redundancy()
        )

        # --------------------------------------------------
        # 6. Calculate changes
        # --------------------------------------------------

        delta_coverage = (
                current_coverage
                - previous_coverage
        )

        delta_distance = (
                current_distance
                - previous_distance
        )

        delta_redundancy = (
                current_redundancy
                - previous_redundancy
        )

        print(
            f"\nTimestep {self.simulator.timestep}"
        )

        print(
            f"Coverage: "
            f"{previous_coverage:.4f} -> "
            f"{current_coverage:.4f} "
            f"(Δ={delta_coverage:.4f})"
        )

        print(
            f"Distance: "
            f"{previous_distance:.2f} -> "
            f"{current_distance:.2f} "
            f"(Δ={delta_distance:.2f})"
        )

        print(
            f"Redundancy: "
            f"{previous_redundancy:.4f} -> "
            f"{current_redundancy:.4f} "
            f"(Δ={delta_redundancy:.4f})"
        )

        # --------------------------------------------------
        # 7. Calculate reward
        # --------------------------------------------------

        reward = self._calculate_reward(
            delta_coverage,
            delta_distance,
            delta_redundancy,
        )

        # --------------------------------------------------
        # 8. Get next state
        # --------------------------------------------------

        next_state = self._get_state()

        # --------------------------------------------------
        # 9. Check episode termination
        # --------------------------------------------------

        terminated = (
                current_coverage >= 90.0
        )

        truncated = (
                self.simulator.timestep
                >= self.max_steps
        )

        return (
            next_state,
            reward,
            terminated,
            truncated,
            {},
        )

    def _calculate_reward(
            self,
            delta_coverage,
            delta_distance,
            delta_redundancy,
    ):

        coverage_reward = 10.0 * delta_coverage
        distance_penalty = 0.05 * delta_distance
        redundancy_penalty = 1.0 * delta_redundancy

        reward = (
                coverage_reward
                - distance_penalty
                - redundancy_penalty
        )

        print(
            f"Reward components: "
            f"coverage={coverage_reward:.4f}, "
            f"distance=-{distance_penalty:.4f}, "
            f"redundancy=-{redundancy_penalty:.4f}, "
            f"TOTAL={reward:.4f}"
        )

        return reward

if __name__ == "__main__":

    env = ExplorationEnv()

    state = env.reset(
        map_seed=42
    )

    print(
        "Initial state:",
        state
    )

    for step in range(10):

        # Fixed weights for now
        action = np.array(
            [
                1.0,
                0.5,
                0.5,
                0.1,
            ],
            dtype=np.float32,
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"Step {step + 1}: "
            f"reward={reward:.4f}, "
            f"state={next_state}"
        )

        if terminated or truncated:
            break