"""
Comparison class for PPO weights vs classical exploration methodologies.

This class provides a unified interface to compare:
- PPO-learned utility weights
- Fixed hand-tuned weights
- Various classical frontier assignment strategies
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import glob
import pygame
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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
from rl.exploration_env import ExplorationEnv
from models.constants import Cell


class PPOClassicalComparator:
    """
    Compares PPO-learned weights against classical exploration strategies.
    """

    def __init__(
        self,
        grid_width: int = 100,
        grid_height: int = 100,
        num_drones: int = 5,
        obstacle_percentage: float = 0.20,
        communication_radius: int = 10,
        max_steps: int = 10000,
        target_coverage: float = 90.0,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_drones = num_drones
        self.obstacle_percentage = obstacle_percentage
        self.communication_radius = communication_radius
        self.max_steps = max_steps
        self.target_coverage = target_coverage

        self.results = []
        self.ppo_models = {}
        self.enable_visualization = False
        self.visualizer = None

    def register_ppo_model(self, name: str, model_path: str):
        """
        Register a PPO model for comparison.

        Args:
            name: Identifier for this PPO model
            model_path: Path to the trained PPO model
        """
        self.ppo_models[name] = model_path
        print(f"Registered PPO model: {name} -> {model_path}")

    def setup_visualization(self, cell_size=6, step_delay=0.05):
        """
        Enable pygame visualization for experiments.

        Args:
            cell_size: Size of each grid cell in pixels
            step_delay: Delay between steps in seconds
        """
        self.enable_visualization = True
        self.visualizer = ExperimentVisualizer(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            cell_size=cell_size,
            step_delay=step_delay,
        )

    def get_classical_strategies(self) -> Dict[str, callable]:
        """
        Returns factory functions for classical strategies.
        """
        # Default fixed weights (hand-tuned)
        fixed_utility = FrontierUtility(
            alpha=1.0,
            beta=0.5,
            gamma=0.5,
            delta=0.1
        )

        return {
            # "random": lambda: RandomStrategy(),
            # "greedy": lambda: FrontierStrategy(GreedyFrontierAssigner(BFSPlanner())),
            # "cluster_frontier": lambda: FrontierStrategy(
            #     ClusterFrontierAssigner(BFSPlanner())
            # ),
            # "cluster_utility": lambda: FrontierStrategy(
            #     ClusterFrontierUtilityAssigner(BFSPlanner(), fixed_utility)
            # ),
            # "hungarian_utility_bfs": lambda: FrontierStrategy(
            #     HungarianFrontierAssigner(planner=BFSPlanner(), utility=fixed_utility)
            # ),
            # "hungarian_utility_astar": lambda: FrontierStrategy(
            #     HungarianFrontierAssigner(planner=AStarPlanner(), utility=fixed_utility)
            # ),
        }

    def get_ppo_strategies(self) -> Dict[str, callable]:
        """
        Returns factory functions for PPO-based strategies.
        """
        strategies = {}

        for name, model_path in self.ppo_models.items():
            strategies[f"ppo_{name}"] = lambda mp=model_path: self._create_ppo_strategy(mp)

        return strategies

    def _create_ppo_strategy(self, model_path: str):
        """
        Create a PPO-based strategy from a trained model.
        """
        # This would load the PPO model and create a strategy
        # For now, we'll use the ExplorationEnv with the model
        # This is a placeholder - actual implementation depends on your PPO setup
        env = ExplorationEnv(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_drones=self.num_drones,
            obstacle_percentage=self.obstacle_percentage,
            communication_radius=self.communication_radius,
            max_steps=self.max_steps,
        )
        # Load model here
        # env.load_model(model_path)
        return env

    def run_classical_experiment(
        self,
        strategy_name: str,
        seed: int,
    ) -> Optional[Dict]:
        """
        Run a single classical strategy experiment.

        Args:
            strategy_name: Name of the classical strategy
            seed: Random seed for the map

        Returns:
            Dictionary of metrics or None if experiment failed
        """
        strategies = self.get_classical_strategies()

        if strategy_name not in strategies:
            print(f"Unknown strategy: {strategy_name}")
            return None

        strategy_factory = strategies[strategy_name]
        strategy = strategy_factory()

        simulator = Simulator(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_drones=self.num_drones,
            obstacle_percentage=self.obstacle_percentage,
            strategy=strategy,
            communication_radius=self.communication_radius,
            map_seed=seed,
        )

        # Initialize visualization if enabled
        if self.enable_visualization:
            self.visualizer.reset(strategy_name)

        metrics_at_target = None
        running = True

        while simulator.timestep < self.max_steps and running:
            simulator.step()

            coverage = simulator.get_coverage()

            if coverage >= self.target_coverage and metrics_at_target is None:
                metrics_at_target = {
                    "strategy": strategy_name,
                    "seed": seed,
                    "time_to_target": simulator.timestep,
                    "coverage": coverage,
                    "distance": simulator.get_total_distance(),
                    "sensing_redundancy": simulator.get_sensing_redundancy(),
                    "visit_overlap": simulator.get_visit_overlap_percentage(),
                    "movement_efficiency": simulator.get_movement_efficiency(),
                }

                # Get nodes expanded if available
                if hasattr(strategy, 'get_metrics'):
                    metrics = strategy.get_metrics()
                    metrics_at_target["nodes_expanded"] = metrics.get('nodes_expanded', 0)

            if coverage >= self.target_coverage + 5:
                break

            # Update visualization if enabled
            if self.enable_visualization:
                running = self.visualizer.update(simulator)
                if not running:
                    break

        return metrics_at_target

    def run_ppo_experiment(
        self,
        model_name: str,
        seed: int,
    ) -> Optional[Dict]:
        """
        Run a single PPO model experiment.

        Args:
            model_name: Name of the registered PPO model
            seed: Random seed for the map

        Returns:
            Dictionary of metrics or None if experiment failed
        """
        if model_name not in self.ppo_models:
            print(f"Unknown PPO model: {model_name}")
            return None

        model_path = self.ppo_models[model_name]

        # Create environment with PPO model
        env = ExplorationEnv(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_drones=self.num_drones,
            obstacle_percentage=self.obstacle_percentage,
            communication_radius=self.communication_radius,
            max_steps=self.max_steps,
        )

        # Load PPO model (placeholder - implement based on your PPO setup)
        # env.load_model(model_path)

        state, _ = env.reset(seed=seed)

        metrics_at_target = None

        for step in range(self.max_steps):
            # Get action from PPO model
            action = env.action_space.sample()  # Placeholder - use model.predict(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            coverage = info["coverage"]

            if coverage >= self.target_coverage and metrics_at_target is None:
                metrics_at_target = {
                    "strategy": f"ppo_{model_name}",
                    "seed": seed,
                    "time_to_target": step,
                    "coverage": coverage,
                    "distance": info["distance"],
                    "sensing_redundancy": info["redundancy"],
                    "visit_overlap": info.get("visit_overlap", 0),
                    "movement_efficiency": info.get("movement_efficiency", 0),
                }

            if coverage >= self.target_coverage + 5 or terminated or truncated:
                break

            state = next_state

        return metrics_at_target

    def run_comparison(
        self,
        seeds: List[int],
        classical_strategies: Optional[List[str]] = None,
        ppo_models: Optional[List[str]] = None,
    ):
        """
        Run comparison across all specified strategies and models.

        Args:
            seeds: List of random seeds to test
            classical_strategies: List of classical strategy names (None = all)
            ppo_models: List of PPO model names (None = all registered)
        """
        if classical_strategies is None:
            classical_strategies = list(self.get_classical_strategies().keys())

        if ppo_models is None:
            ppo_models = list(self.ppo_models.keys())

        all_strategies = classical_strategies + [f"ppo_{name}" for name in ppo_models]

        print(f"Running comparison on seeds {seeds}")
        print(f"Classical strategies: {classical_strategies}")
        print(f"PPO models: {ppo_models}")
        print("-" * 60)

        for seed in seeds:
            print(f"\n===== SEED {seed} =====")

            # Run classical strategies
            for strategy_name in classical_strategies:
                print(f"  Running {strategy_name}...")
                result = self.run_classical_experiment(strategy_name, seed)
                if result:
                    self.results.append(result)
                    print(f"    ✓ Reached target at step {result['time_to_target']}")
                else:
                    print(f"    ✗ Failed")

            # Run PPO models
            for model_name in ppo_models:
                print(f"  Running PPO {model_name}...")
                result = self.run_ppo_experiment(model_name, seed)
                if result:
                    self.results.append(result)
                    print(f"    ✓ Reached target at step {result['time_to_target']}")
                else:
                    print(f"    ✗ Failed")

    def compute_aggregated_stats(self) -> pd.DataFrame:
        """
        Compute aggregated statistics for all strategies.
        """
        if not self.results:
            print("No results to analyze")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Group by strategy and compute stats
        agg_dict = {
            "time_to_target": ["mean", "std"],
            "coverage": ["mean", "std"],
            "distance": ["mean", "std"],
            "sensing_redundancy": ["mean", "std"],
            "visit_overlap": ["mean", "std"],
            "movement_efficiency": ["mean", "std"],
        }
        
        # Only add nodes_expanded if the column exists
        if "nodes_expanded" in df.columns:
            agg_dict["nodes_expanded"] = ["mean", "std"]
        
        summary = df.groupby("strategy").agg(agg_dict).reset_index()

        return summary

    def print_comparison_table(self):
        """
        Print a formatted comparison table.
        """
        summary = self.compute_aggregated_stats()

        if summary.empty:
            print("No results to display")
            return

        print("\n" + "=" * 80)
        print("PPO vs CLASSICAL COMPARISON (Mean ± Std)")
        print("=" * 80)

        for _, row in summary.iterrows():
            strategy = row["strategy"]
            print(f"\n{strategy}:")
            print(f"  Time to target: {row['time_to_target']['mean']:.0f} ± {row['time_to_target']['std']:.0f}")
            print(f"  Coverage: {row['coverage']['mean']:.1f}% ± {row['coverage']['std']:.1f}%")
            print(f"  Distance: {row['distance']['mean']:.0f} ± {row['distance']['std']:.0f}")
            print(f"  Sensing redundancy: {row['sensing_redundancy']['mean']:.1f}% ± {row['sensing_redundancy']['std']:.1f}%")
            print(f"  Visit overlap: {row['visit_overlap']['mean']:.1f}% ± {row['visit_overlap']['std']:.1f}%")
            print(f"  Movement efficiency: {row['movement_efficiency']['mean']:.3f} ± {row['movement_efficiency']['std']:.3f}")

            if "nodes_expanded" in row and not pd.isna(row["nodes_expanded"]["mean"]):
                print(f"  Nodes expanded: {row['nodes_expanded']['mean']:.0f} ± {row['nodes_expanded']['std']:.0f}")

        print("\n" + "=" * 80)

    def save_results(self, output_dir: str = "results/comparison"):
        """
        Save results to CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save raw results
        if self.results:
            df = pd.DataFrame(self.results)
            raw_file = os.path.join(output_dir, "raw_comparison.csv")
            df.to_csv(raw_file, index=False)
            print(f"Raw results saved to {raw_file}")

        # Save aggregated stats
        summary = self.compute_aggregated_stats()
        if not summary.empty:
            agg_file = os.path.join(output_dir, "aggregated_comparison.csv")
            summary.to_csv(agg_file, index=False)
            print(f"Aggregated results saved to {agg_file}")

    def load_existing_results(self, results_dir: str = "results/comparison"):
        """
        Load existing comparison results from CSV.
        """
        raw_file = os.path.join(results_dir, "raw_comparison.csv")

        if os.path.exists(raw_file):
            df = pd.read_csv(raw_file)
            self.results = df.to_dict("records")
            print(f"Loaded {len(self.results)} existing results from {raw_file}")
            return True
        else:
            print(f"No existing results found at {raw_file}")
            return False


class ExperimentVisualizer:
    """
    Pygame visualization for classical exploration experiments.
    Shows real-time drone movement with metrics overlay.
    """

    def __init__(self, grid_width, grid_height, cell_size=6, step_delay=0.05):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.step_delay = step_delay

        self.panel_width = grid_width * cell_size
        self.panel_height = grid_height * cell_size
        self.info_height = 120

        self.window_width = self.panel_width
        self.window_height = self.panel_height + self.info_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Exploration Strategy Visualization")

        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        self.metrics_font = pygame.font.SysFont("Arial", 16)
        self.small_font = pygame.font.SysFont("Arial", 14)

        self.clock = pygame.time.Clock()
        self.strategy_name = ""

        # Colors
        self.colors = {
            Cell.UNEXPLORED: (200, 200, 200),
            Cell.FREE: (245, 245, 245),
            Cell.OBSTACLE: (40, 40, 40),
        }

        self.drone_colors = [
            (220, 50, 50),
            (50, 100, 220),
            (50, 180, 70),
            (220, 150, 30),
            (160, 60, 180),
            (30, 180, 180),
        ]

    def reset(self, strategy_name):
        """Reset visualization for a new strategy."""
        self.strategy_name = strategy_name
        print(f"Starting visualization for: {strategy_name}")

    def update(self, simulator):
        """Update visualization with current simulator state."""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.step_delay = 0.0 if self.step_delay > 0 else 0.05

        # Draw
        self.screen.fill((255, 255, 255))

        # Draw grid
        self._draw_grid(simulator.robot_map)

        # Draw drones
        self._draw_drones(simulator.drones, simulator.robot_map)

        # Draw info panel
        self._draw_info_panel(simulator)

        pygame.display.flip()

        # Control speed
        if self.step_delay > 0:
            time.sleep(self.step_delay)
        else:
            self.clock.tick(60)

        return True

    def _draw_grid(self, robot_map):
        """Draw the robot map grid."""
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_value = robot_map.get_cell(x, y)
                color = self.colors.get(cell_value, (200, 200, 200))

                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

                # Draw grid lines
                pygame.draw.rect(
                    self.screen,
                    (220, 220, 220),
                    (
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                    1,
                )

    def _draw_drones(self, drones, robot_map):
        """Draw drones on the grid with LIDAR sensing rays to discovered cells."""
        from models.constants import Cell
        import math

        for drone in drones:
            cx = drone.x * self.cell_size + self.cell_size // 2
            cy = drone.y * self.cell_size + self.cell_size // 2
            radius = max(3, self.cell_size // 2)

            color = self.drone_colors[drone.id % len(self.drone_colors)]

            # Draw rays to cells within sensor radius
            sensor_radius = drone.sensor_radius
            for dy in range(-sensor_radius, sensor_radius + 1):
                for dx in range(-sensor_radius, sensor_radius + 1):
                    # Check if within circular sensor radius
                    distance_sq = dx * dx + dy * dy
                    if distance_sq > sensor_radius * sensor_radius:
                        continue

                    target_x = drone.x + dx
                    target_y = drone.y + dy

                    if not robot_map.is_inside(target_x, target_y):
                        continue

                    # Draw ray to this cell
                    target_cx = target_x * self.cell_size + self.cell_size // 2
                    target_cy = target_y * self.cell_size + self.cell_size // 2

                    # Different color for unexplored vs explored cells
                    cell_type = robot_map.get_cell(target_x, target_y)
                    if cell_type == Cell.UNEXPLORED:
                        ray_color = (255, 100, 100)  # Red for unexplored
                    elif cell_type == Cell.FREE:
                        ray_color = (100, 255, 100)  # Green for free
                    else:
                        ray_color = (150, 150, 150)  # Gray for obstacles

                    # Draw ray from drone center to cell center
                    pygame.draw.line(self.screen, ray_color, (cx, cy), (target_cx, target_cy), 1)

            # Draw drone body
            pygame.draw.circle(self.screen, color, (cx, cy), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), radius, 1)

    def _draw_info_panel(self, simulator):
        """Draw metrics info panel."""
        panel_y = self.panel_height

        # Background
        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            (0, panel_y, self.window_width, self.info_height),
        )
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (0, panel_y),
            (self.window_width, panel_y),
            2,
        )

        # Strategy name
        title = self.font.render(f"Strategy: {self.strategy_name}", True, (0, 0, 0))
        self.screen.blit(title, (10, panel_y + 10))

        # Metrics
        coverage = simulator.get_coverage()
        redundancy = simulator.get_sensing_redundancy()
        distance = simulator.get_total_distance()
        timestep = simulator.timestep
        movement_eff = simulator.get_movement_efficiency()

        metrics = [
            f"Timestep: {timestep}",
            f"Coverage: {coverage:.2f}%",
            f"Redundancy: {redundancy:.2f}%",
            f"Distance: {distance:.0f}",
            f"Movement Efficiency: {movement_eff:.4f}",
        ]

        for i, metric in enumerate(metrics):
            text = self.metrics_font.render(metric, True, (50, 50, 100))
            self.screen.blit(text, (10, panel_y + 40 + i * 18))

        # Controls hint
        hint = self.small_font.render("SPACE: toggle speed | ESC: quit", True, (100, 100, 100))
        hint_rect = hint.get_rect(right=self.window_width - 10, top=panel_y + 10)
        self.screen.blit(hint, hint_rect)

    def close(self):
        """Close pygame window."""
        pygame.quit()


# Example usage
if __name__ == "__main__":
    comparator = PPOClassicalComparator(
        grid_width=100,
        grid_height=100,
        num_drones=5,
        obstacle_percentage=0.20,
        communication_radius=10,
        max_steps=10000,
        target_coverage=90.0,
    )

    # Enable visualization (remove this line to run without visualization)
    comparator.setup_visualization(cell_size=6, step_delay=0.05)

    # Register PPO models
    comparator.register_ppo_model("50k_a", "models/ppo/ppo-a/ppo_utility_50k")
    comparator.register_ppo_model("100k_b", "models/ppo/ppo-b/ppo_utility_100k")

    # Run comparison
    comparator.run_comparison(
        seeds=list(range(105, 116)),
        # classical_strategies=["greedy", "cluster_frontier", "hungarian_utility_bfs", "hungarian_utility_astar"],
        ppo_models=["50k_a", "100k_b"],
    )

    # Print and save results
    comparator.print_comparison_table()
    comparator.save_results()

    # Close visualization if enabled
    if comparator.enable_visualization and comparator.visualizer:
        comparator.visualizer.close()
