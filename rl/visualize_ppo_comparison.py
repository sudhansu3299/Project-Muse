import time
import numpy as np
import pygame

from stable_baselines3 import PPO

from rl.exploration_env import ExplorationEnv
from environment.simulator import Simulator
from environment.coordination.greedy_frontier_assigner import GreedyFrontierAssigner
from environment.coordination.cluster_frontier_assigner import ClusterFrontierAssigner
from environment.coordination.cluster_frontier_utility_assigner import ClusterFrontierUtilityAssigner
from environment.coordination.hungarian_frontier_assigner import HungarianFrontierAssigner
from environment.planner.bfs_planner import BFSPlanner
from environment.planner.a_star_planner import AStarPlanner
from environment.utils.frontier_utility import FrontierUtility
from strategy.frontier_strategy import FrontierStrategy
from strategy.random_strategy import RandomStrategy


class StrategyWrapper:
    """
    Unified wrapper for both classical strategies and PPO models.
    Provides a consistent interface for visualization.
    """

    def __init__(self, name, strategy_type, config, grid_width, grid_height, num_drones, 
                 obstacle_percentage, communication_radius=10, max_steps=1000):
        self.name = name
        self.strategy_type = strategy_type  # 'ppo' or 'classical'
        self.config = config
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_drones = num_drones
        self.obstacle_percentage = obstacle_percentage
        self.communication_radius = communication_radius
        self.max_steps = max_steps

        self.env = None
        self.simulator = None
        self.model = None
        self.state = None

    def setup(self):
        """Initialize the environment/simulator based on strategy type."""
        if self.strategy_type == 'ppo':
            self._setup_ppo()
        elif self.strategy_type == 'classical':
            self._setup_classical()

    def _setup_ppo(self):
        """Setup PPO environment."""
        model_path = self.config
        self.model = PPO.load(model_path)
        self.env = ExplorationEnv(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_drones=self.num_drones,
            obstacle_percentage=self.obstacle_percentage,
            communication_radius=self.communication_radius,
            max_steps=self.max_steps,
        )

    def _setup_classical(self):
        """Setup classical strategy simulator."""
        strategy = self.config
        self.simulator = Simulator(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_drones=self.num_drones,
            obstacle_percentage=self.obstacle_percentage,
            strategy=strategy,
            communication_radius=self.communication_radius,
        )

    def reset(self, seed):
        """Reset the environment/simulator with given seed."""
        if self.strategy_type == 'ppo':
            self.state, _ = self.env.reset(seed=seed)
            self.env.last_action = None
        elif self.strategy_type == 'classical':
            # Recreate simulator with new seed
            strategy = self.config
            self.simulator = Simulator(
                grid_width=self.grid_width,
                grid_height=self.grid_height,
                num_drones=self.num_drones,
                obstacle_percentage=self.obstacle_percentage,
                strategy=strategy,
                communication_radius=self.communication_radius,
                map_seed=seed,
            )

    def step(self):
        """Execute one step of the strategy."""
        if self.strategy_type == 'ppo':
            return self._step_ppo()
        elif self.strategy_type == 'classical':
            return self._step_classical()

    def _step_ppo(self):
        """Step PPO model."""
        action, _ = self.model.predict(self.state, deterministic=True)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.state = next_state
        return {
            'terminated': terminated or truncated,
            'reward': reward,
        }

    def _step_classical(self):
        """Step classical strategy."""
        self.simulator.step()
        coverage = self.simulator.get_coverage()
        return {
            'terminated': coverage >= 95.0 or self.simulator.timestep >= self.max_steps,
            'reward': 0.0,  # Classical strategies don't have rewards
        }

    def get_simulator(self):
        """Get the simulator for visualization."""
        if self.strategy_type == 'ppo':
            return self.env.simulator
        elif self.strategy_type == 'classical':
            return self.simulator
        return None

    def get_metrics(self):
        """Get current metrics."""
        sim = self.get_simulator()
        if sim is None:
            return {}
        
        metrics = {
            'coverage': sim.get_coverage(),
            'redundancy': sim.get_sensing_redundancy(),
            'distance': sim.get_total_distance(),
            'timestep': sim.timestep,
        }
        
        # Add PPO-specific metrics
        if self.strategy_type == 'ppo' and hasattr(self.env, 'last_action'):
            action = self.env.last_action
            if action is not None and len(action) >= 4:
                metrics['alpha'] = float(action[0])
                metrics['beta'] = float(action[1])
                metrics['gamma'] = float(action[2])
                metrics['delta'] = float(action[3])
        
        return metrics


class ComparePPOVisualization:

    def __init__(
            self,
            strategies,
            seed,
            grid_width=100,
            grid_height=100,
            num_drones=5,
            obstacle_percentage=0.1,
            communication_radius=10,
            max_steps=1000,
            cell_size=7,
            step_delay=0.03,
    ):
        """
        strategies:
            Dictionary of:
                {
                    "PPO 25k": ('ppo', 'models/ppo/ppo_25k.zip'),
                    "Greedy BFS": ('classical', FrontierStrategy(GreedyFrontierAssigner(BFSPlanner()))),
                    "Hungarian A*": ('classical', FrontierStrategy(HungarianFrontierAssigner(AStarPlanner()))),
                }

        seed:
            Same map seed is used for every strategy.

        This is important:
            Every strategy gets EXACTLY the same environment/map.
        """

        self.strategies_config = strategies
        self.seed = seed

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_drones = num_drones
        self.obstacle_percentage = obstacle_percentage
        self.communication_radius = communication_radius
        self.max_steps = max_steps

        self.cell_size = cell_size
        self.step_delay = step_delay

        self.strategy_wrappers = {}
        self.total_rewards = {}

        # --------------------------------------------------
        # Setup strategy wrappers
        # --------------------------------------------------

        for name, (strategy_type, config) in strategies.items():
            print(f"Setting up {name} ({strategy_type})")
            
            wrapper = StrategyWrapper(
                name=name,
                strategy_type=strategy_type,
                config=config,
                grid_width=grid_width,
                grid_height=grid_height,
                num_drones=num_drones,
                obstacle_percentage=obstacle_percentage,
                communication_radius=communication_radius,
                max_steps=max_steps,
            )
            wrapper.setup()
            self.strategy_wrappers[name] = wrapper

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        self.results = {}
        self.dones = {}

        # --------------------------------------------------
        # Pygame
        # --------------------------------------------------

        self.panel_width = (
                grid_width * cell_size
        )

        self.panel_height = (
                grid_height * cell_size
        )

        self.info_height = 130

        self.window_width = (
                self.panel_width * len(strategies)
        )

        self.window_height = (
                self.panel_height
                + self.info_height
        )

        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                self.window_width,
                self.window_height,
            )
        )

        pygame.display.set_caption(
            "Strategy Comparison Visualization"
        )

        self.font = pygame.font.SysFont(
            "Arial",
            18,
        )

        self.small_font = pygame.font.SysFont(
            "Arial",
            15,
        )

        self.clock = pygame.time.Clock()

    # ======================================================
    # RESET ALL STRATEGIES
    # ======================================================

    def reset(self):

        print()
        print("========================================")
        print(f"COMPARISON SEED = {self.seed}")
        print("========================================")

        for name, wrapper in self.strategy_wrappers.items():
            wrapper.reset(self.seed)
            sim = wrapper.get_simulator()
            print(
                f"{name}: "
                f"initial coverage="
                f"{sim.get_coverage():.2f}%"
            )

        self.dones = {name: False for name in self.strategy_wrappers}
        self.total_rewards = {name: 0.0 for name in self.strategy_wrappers}

    # ======================================================
    # DRAW ENVIRONMENT
    # ======================================================

    def draw_environment(
            self,
            wrapper,
            name,
            panel_index,
    ):

        x_offset = (
                panel_index
                * self.panel_width
        )

        simulator = wrapper.get_simulator()

        # --------------------------------------------------
        # Background
        # --------------------------------------------------

        pygame.draw.rect(
            self.screen,
            (245, 245, 245),
            (
                x_offset,
                0,
                self.panel_width,
                self.panel_height,
            ),
        )

        # --------------------------------------------------
        # Draw grid
        # --------------------------------------------------

        robot_map = simulator.robot_map

        for y in range(self.grid_height):

            for x in range(self.grid_width):

                try:
                    value = robot_map.grid[y][x]
                except Exception:
                    continue

                # Unknown
                color = (200, 200, 200)

                # Free
                if value == 0:
                    color = (245, 245, 245)

                # Obstacle
                elif value == 1:
                    color = (40, 40, 40)

                # Covered
                elif value == 2:
                    color = (180, 220, 180)

                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        x_offset
                        + x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

        # --------------------------------------------------
        # Draw drones
        # --------------------------------------------------

        for drone in simulator.drones:

            cx = (
                    x_offset
                    + drone.x * self.cell_size
                    + self.cell_size // 2
            )

            cy = (
                    drone.y * self.cell_size
                    + self.cell_size // 2
            )

            radius = max(
                3,
                self.cell_size // 2,
                )

            pygame.draw.circle(
                self.screen,
                self.drone_color(drone.id),
                (cx, cy),
                radius,
            )

        # --------------------------------------------------
        # Border
        # --------------------------------------------------

        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (
                x_offset,
                0,
                self.panel_width,
                self.panel_height,
            ),
            2,
        )

        # --------------------------------------------------
        # Info panel
        # --------------------------------------------------

        coverage = (
            simulator.get_coverage()
        )

        redundancy = (
            simulator.get_sensing_redundancy()
        )

        distance = (
            simulator.get_total_distance()
        )

        timestep = simulator.timestep

        # --------------------------------------------------
        # Get metrics
        # --------------------------------------------------

        metrics = wrapper.get_metrics()
        coverage = metrics.get('coverage', 0)
        redundancy = metrics.get('redundancy', 0)
        distance = metrics.get('distance', 0)
        timestep = metrics.get('timestep', 0)
        alpha = metrics.get('alpha', 0)
        beta = metrics.get('beta', 0)
        gamma = metrics.get('gamma', 0)
        delta = metrics.get('delta', 0)

        # --------------------------------------------------
        # Text
        # --------------------------------------------------

        text_x = (
                x_offset + 10
        )

        text_y = (
                self.panel_height + 8
        )

        self.draw_text(
            name,
            text_x,
            text_y,
            self.font,
        )

        self.draw_text(
            f"timestep: {timestep}",
            text_x,
            text_y + 24,
            self.small_font,
            )

        self.draw_text(
            f"coverage: {coverage:.2f}%",
            text_x,
            text_y + 44,
            self.small_font,
            )

        self.draw_text(
            f"redundancy: {redundancy:.2f}%",
            text_x,
            text_y + 64,
            self.small_font,
            )

        self.draw_text(
            f"distance: {distance:.0f}",
            text_x,
            text_y + 84,
            self.small_font,
            )

        # Only show weights for PPO strategies
        if wrapper.strategy_type == 'ppo':
            self.draw_text(
                (
                    f"α={alpha:.2f} "
                    f"β={beta:.2f} "
                    f"γ={gamma:.2f} "
                    f"δ={delta:.2f}"
                ),
                text_x,
                text_y + 104,
                self.small_font,
            )
        else:
            # Show strategy type for classical
            self.draw_text(
                f"Type: {wrapper.strategy_type.upper()}",
                text_x,
                text_y + 104,
                self.small_font,
            )

    # ======================================================
    # DRONE COLORS
    # ======================================================

    def drone_color(self, drone_id):

        colors = [
            (220, 50, 50),
            (50, 100, 220),
            (50, 180, 70),
            (220, 150, 30),
            (160, 60, 180),
            (30, 180, 180),
        ]

        return colors[
            drone_id % len(colors)
            ]

    # ======================================================
    # TEXT
    # ======================================================

    def draw_text(
            self,
            text,
            x,
            y,
            font,
    ):

        surface = font.render(
            text,
            True,
            (0, 0, 0),
        )

        self.screen.blit(
            surface,
            (x, y),
        )

    # ======================================================
    # RUN
    # ======================================================

    def run(self):

        self.reset()

        running = True

        while running:

            # --------------------------------------------------
            # Pygame events
            # --------------------------------------------------

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    if event.key == pygame.K_SPACE:
                        self.step_delay = (
                            0.0
                            if self.step_delay > 0
                            else 0.03
                        )

            # --------------------------------------------------
            # Step each strategy
            # --------------------------------------------------

            all_done = True

            for name, wrapper in self.strategy_wrappers.items():

                if self.dones[name]:
                    continue

                all_done = False

                result = wrapper.step()
                self.total_rewards[name] += result['reward']

                if result['terminated']:
                    self.dones[name] = True

            # --------------------------------------------------
            # Draw
            # --------------------------------------------------

            self.screen.fill(
                (255, 255, 255)
            )

            for index, name in enumerate(
                    self.strategy_wrappers.keys()
            ):

                self.draw_environment(
                    self.strategy_wrappers[name],
                    name,
                    index,
                )

            pygame.display.flip()

            # --------------------------------------------------
            # Finished?
            # --------------------------------------------------

            if all_done:

                self.print_results()

                # Keep final frame visible
                waiting = True

                while waiting:

                    for event in pygame.event.get():

                        if event.type == pygame.QUIT:
                            waiting = False
                            running = False

                        elif event.type == pygame.KEYDOWN:

                            if event.key in (
                                    pygame.K_ESCAPE,
                                    pygame.K_RETURN,
                                    pygame.K_SPACE,
                            ):
                                waiting = False
                                running = False

                    self.clock.tick(30)

            time.sleep(
                self.step_delay
            )

        pygame.quit()

    # ======================================================
    # RESULTS
    # ======================================================

    def print_results(self):

        print()
        print("========================================")
        print("STRATEGY COMPARISON")
        print("========================================")

        for name, wrapper in self.strategy_wrappers.items():

            simulator = wrapper.get_simulator()
            metrics = wrapper.get_metrics()

            print()
            print(name)
            print(
                f"  Type:       {wrapper.strategy_type.upper()}"
            )
            print(
                f"  Steps:      {metrics.get('timestep', 0)}"
            )
            print(
                f"  Coverage:   {metrics.get('coverage', 0):.2f}%"
            )
            print(
                f"  Distance:   {metrics.get('distance', 0):.2f}"
            )
            print(
                f"  Redundancy: {metrics.get('redundancy', 0):.2f}%"
            )
            print(
                f"  Reward:     "
                f"{self.get_total_reward(name):.3f}"
            )

    # ======================================================
    # TOTAL REWARD
    # ======================================================

    def get_total_reward(self, name):

        # Reconstruct from the environment if available.
        # Otherwise return zero.
        return getattr(
            self,
            "total_rewards",
            {},
        ).get(
            name,
            0.0,
        )


# ==========================================================
# MAIN
# ==========================================================

def get_classical_strategies():
    """
    Factory function to create classical strategies.
    Easy to add/remove strategies.
    """
    # Default fixed weights (hand-tuned)
    fixed_utility = FrontierUtility(
        alpha=1.0,
        beta=0.5,
        gamma=0.5,
        delta=0.1
    )

    strategies = {
        # Add/remove classical strategies here
        "Greedy BFS": ('classical', FrontierStrategy(GreedyFrontierAssigner(BFSPlanner()))),
        "Cluster Frontier": ('classical', FrontierStrategy(ClusterFrontierAssigner(BFSPlanner()))),
        "Cluster Utility": ('classical', FrontierStrategy(ClusterFrontierUtilityAssigner(BFSPlanner(), fixed_utility))),
        "Hungarian BFS": ('classical', FrontierStrategy(HungarianFrontierAssigner(planner=BFSPlanner(), utility=fixed_utility))),
        "Hungarian A*": ('classical', FrontierStrategy(HungarianFrontierAssigner(planner=AStarPlanner(), utility=fixed_utility))),
    }
    
    return strategies


def get_ppo_models():
    """
    Factory function to create PPO model configurations.
    Easy to add/remove models.
    """
    models = {
        # Add/remove PPO models here
        "PPO 50k": ('ppo', 'models/ppo/ppo-a/ppo_utility_50k'),
        "PPO 100k": ('ppo', 'models/ppo/ppo-b/ppo_utility_100k'),
    }
    
    return models


if __name__ == "__main__":

    # Mix and match any combination of classical and PPO strategies
    # Just add or remove entries from the dictionaries
    
    classical = get_classical_strategies()
    ppo = get_ppo_models()
    
    # Combine them - pick and choose what you want to visualize
    strategies = {
        **classical,  # Add all classical
        **ppo,       # Add all PPO
        # Or selectively add specific ones:
        # "Greedy BFS": classical["Greedy BFS"],
        # "PPO 50k": ppo["PPO 50k"],
    }

    visualizer = ComparePPOVisualization(
        strategies=strategies,
        seed=115,
        grid_width=100,
        grid_height=100,
        num_drones=5,
        obstacle_percentage=0.1,
        communication_radius=10,
        max_steps=5000,
        cell_size=7,
        step_delay=0.03,
    )

    visualizer.run()