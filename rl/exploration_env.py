import gymnasium as gym
import numpy as np

from environment.simulator import Simulator

from environment.coordination.hungarian_frontier_assigner import (
    HungarianFrontierAssigner,
)

from environment.planner.bfs_planner import BFSPlanner

from environment.utils.frontier_utility import FrontierUtility

from strategy.frontier_strategy import FrontierStrategy


class ExplorationEnv(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
            self,
            grid_width=100,
            grid_height=100,
            num_drones=5,
            obstacle_percentage=0.1,
            communication_radius=10,
            max_steps=1000,
    ):

        super().__init__()

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_drones = num_drones
        self.obstacle_percentage = obstacle_percentage
        self.communication_radius = communication_radius
        self.max_steps = max_steps

        self.last_action = None

        self.training_seeds = None
        self.training_seed_index = 0

        # ==================================================
        # ACTION SPACE
        # ==================================================
        #
        # PPO outputs:
        #
        # [alpha, beta, gamma, delta]
        #
        # alpha -> information gain
        # beta  -> path cost
        # gamma -> redundancy
        # delta -> cluster size
        #
        # ==================================================

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # ==================================================
        # OBSERVATION SPACE
        # ==================================================
        #
        # State:
        #
        # [coverage,
        #  sensing_redundancy,
        #  visit_overlap,
        #  movement_efficiency,
        #  mean_pairwise_distance]
        #
        # We will normalize percentage-based values to [0, 1].
        #
        # ==================================================

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        # ==================================================
        # UTILITY
        # ==================================================

        self.default_weights = {
            "alpha": 1.0,
            "beta": 0.5,
            "gamma": 0.5,
            "delta": 0.1,
        }

        self.utility = FrontierUtility(**self.default_weights)

        # ==================================================
        # PLANNER
        # ==================================================

        self.planner = BFSPlanner()

        # ==================================================
        # HUNGARIAN ASSIGNER
        # ==================================================

        self.frontier_assigner = (
            HungarianFrontierAssigner(
                planner=self.planner,
                utility=self.utility,
            )
        )

        # ==================================================
        # FRONTIER STRATEGY
        # ==================================================

        self.strategy = FrontierStrategy(
            self.frontier_assigner
        )

        # ==================================================
        # SIMULATOR
        # ==================================================

        self.simulator = None

        # ==================================================
        # PREVIOUS METRICS
        # ==================================================

        self.previous_coverage = 0.0
        self.previous_distance = 0.0
        self.previous_redundancy = 0.0

        # ==================================================
        # CURRENT SEED
        # ==================================================

        self.current_seed = None


    def set_training_seeds(self, seeds):

        self.training_seeds = list(seeds)

        self.training_seed_index = 0

    # ======================================================
    # RESET
    # ======================================================

    def reset(
            self,
            *,
            seed=None,
            options=None,
    ):

        super().reset(seed=seed)

        if self.training_seeds is not None:

            map_seed = self.training_seeds[
                self.training_seed_index
            ]

            self.training_seed_index = (
                                               self.training_seed_index + 1
                                       ) % len(self.training_seeds)

        else:
            map_seed = seed

        self.current_seed = map_seed


        # print("\n========== STRATEGY BEFORE RESET ==========")
        # print(self.strategy)

        if hasattr(self.strategy, "reset"):
            self.strategy.reset()

        # --------------------------------------------------
        # Create a fresh simulator for every episode
        # --------------------------------------------------

        self.simulator = Simulator(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_drones=self.num_drones,
            obstacle_percentage=self.obstacle_percentage,
            communication_radius=self.communication_radius,
            strategy=self.strategy,
            map_seed=map_seed,
        )

        # print("\n========== STRATEGY AFTER SIMULATOR RESET ==========")
        # print(self.strategy)

        # --------------------------------------------------
        # Reset utility weights to defaults
        # --------------------------------------------------

        self.utility.set_weights(**self.default_weights)

        # --------------------------------------------------
        # Initial metrics
        # --------------------------------------------------

        self.previous_coverage = (
            self.simulator.get_coverage()
        )

        self.previous_distance = (
            self.simulator.get_total_distance()
        )

        self.previous_redundancy = (
            self.simulator.get_sensing_redundancy()
        )

        # --------------------------------------------------
        # Initial state
        # --------------------------------------------------

        state = self._get_state()

        # print("\n========== ENV RESET ==========")

        # for drone in self.simulator.drones:
        #     print(
        #         f"Drone {drone.id}: "
        #         f"position=({drone.x}, {drone.y}), "
        #     )

        return state, {}

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        self.last_action = np.asarray(
            action,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # 1. Convert PPO action to utility weights
        # --------------------------------------------------

        alpha = float(action[0])
        beta = float(action[1])
        gamma = float(action[2])
        delta = float(action[3])

        # --------------------------------------------------
        # 2. Update utility weights
        # --------------------------------------------------

        self.utility.set_weights(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        # --------------------------------------------------
        # 3. Metrics BEFORE simulator step
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
        # 4. Execute ONE simulation timestep
        # --------------------------------------------------

        stay_drones = self.simulator.step()

        # --------------------------------------------------
        # 5. Metrics AFTER simulator step
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
        # 6. Calculate metric changes
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

        # --------------------------------------------------
        # 7. Calculate reward
        # --------------------------------------------------

        (
            reward,
            coverage_reward,
            distance_penalty,
            redundancy_penalty,
        ) = self._calculate_reward(
            delta_coverage=delta_coverage,
            delta_distance=delta_distance,
            delta_redundancy=delta_redundancy,
        )

        # --------------------------------------------------
        # 8. New state
        # --------------------------------------------------

        next_state = self._get_state()

        # --------------------------------------------------
        # 9. Episode termination
        # --------------------------------------------------

        terminated = (
                current_coverage >= 90.0
        )

        truncated = (
                self.simulator.timestep
                >= self.max_steps
        )

        # --------------------------------------------------
        # 10. Information for debugging/evaluation
        # --------------------------------------------------

        active_drones = 0

        for drone in self.simulator.drones:
            assignment = self.strategy.assignments.get(
                drone.id
            )

            if assignment is None:
                continue

            path = assignment.get("path")

            if path is None:
                continue

            path_index = assignment.get(
                "path_index",
                0,
            )

            if path_index < len(path) - 1:
                active_drones += 1

        info = {
            "seed": self.current_seed,
            "coverage": current_coverage,
            "distance": current_distance,
            "redundancy": current_redundancy,
            "active_drones": active_drones,
            "new_assignments": (
                self.strategy
                .frontier_assigner
                .num_assigned_drones
            ),
            "stay_drones": stay_drones,

            "no_centroid_path":
                self.strategy.frontier_assigner.assignment_failures[
                    "no_centroid_path"
                ],

            "no_frontier_path":
                self.strategy.frontier_assigner.assignment_failures[
                    "no_frontier_path"
                ],

            "delta_coverage": delta_coverage,
            "delta_distance": delta_distance,
            "delta_redundancy": delta_redundancy,

            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,

            "coverage_reward": coverage_reward,
            "distance_penalty": distance_penalty,
            "redundancy_penalty": redundancy_penalty,
        }

        # print(f"\n========== ENV STEP ==========")
        #
        # for drone in self.simulator.drones:
        #     print(
        #         f"Drone {drone.id}: "
        #     )

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    # ======================================================
    # STATE
    # ======================================================

    def _get_state(self):

        coverage = (
                self.simulator.get_coverage()
                / 100.0
        )

        redundancy = (
                self.simulator.get_sensing_redundancy()
                / 100.0
        )

        overlap = (
                self.simulator.get_visit_overlap_percentage()
                / 100.0
        )

        movement_efficiency = (
            self.simulator.get_movement_efficiency()
        )

        mean_pairwise_distance = (
            self.simulator.get_mean_pairwise_distance()
        )

        state = np.array(
            [
                coverage,
                redundancy,
                overlap,
                movement_efficiency,
                mean_pairwise_distance,
            ],
            dtype=np.float32,
        )

        return state

    # ======================================================
    # REWARD
    # ======================================================

    def _calculate_reward(
            self,
            delta_coverage,
            delta_distance,
            delta_redundancy,
    ):

        # Coverage is currently represented by your
        # simulator in percentage points.
        coverage_reward = (
                10.0 * delta_coverage
        )

        # Distance is measured in grid cells.
        distance_penalty = (
                0.05 * delta_distance
        )

        # Redundancy is currently represented by your
        # simulator in percentage points.
        redundancy_penalty = (
                1.0 * delta_redundancy
        )

        reward = (
                coverage_reward
                - distance_penalty
                - redundancy_penalty
        )

        return (
            float(reward),
            float(coverage_reward),
            float(distance_penalty),
            float(redundancy_penalty),
        )