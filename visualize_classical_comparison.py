import time
import pygame

from environment.simulator import Simulator
from models.constants import Cell


class ClassicalComparisonVisualization:
    """
    Visualizes multiple classical exploration algorithms side by side.
    Each algorithm runs on the same map seed for fair comparison.
    """

    def __init__(
        self,
        strategies,
        seed,
        grid_width=100,
        grid_height=100,
        num_drones=5,
        obstacle_percentage=0.20,
        communication_radius=10,
        max_steps=10000,
        cell_size=6,
        step_delay=0.05,
    ):
        """
        strategies:
            Dictionary of strategy name to strategy instance:
                {
                    "cluster_frontier": FrontierStrategy(ClusterFrontierAssigner(BFSPlanner())),
                    "hungarian_utility": FrontierStrategy(HungarianFrontierAssigner(AStarPlanner(), utility)),
                }

        seed:
            Same map seed used for every strategy to ensure fair comparison.
        """
        self.strategies = strategies
        self.seed = seed

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_drones = num_drones
        self.obstacle_percentage = obstacle_percentage
        self.communication_radius = communication_radius
        self.max_steps = max_steps

        self.cell_size = cell_size
        self.step_delay = step_delay

        self.simulators = {}
        self.dones = {}

        # Create simulators for each strategy
        for name, strategy in strategies.items():
            simulator = Simulator(
                grid_width=grid_width,
                grid_height=grid_height,
                num_drones=num_drones,
                obstacle_percentage=obstacle_percentage,
                strategy=strategy,
                communication_radius=communication_radius,
                map_seed=seed,
            )
            self.simulators[name] = simulator
            self.dones[name] = False

        # Pygame setup
        self.panel_width = grid_width * cell_size
        self.panel_height = grid_height * cell_size
        self.info_height = 120

        self.window_width = self.panel_width * len(strategies)
        self.window_height = self.panel_height + self.info_height

        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )

        pygame.display.set_caption("Classical Algorithm Comparison")

        self.font = pygame.font.SysFont("Arial", 18, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 14)

        self.clock = pygame.time.Clock()

        # Grid colors
        self.grid_colors = {
            Cell.UNEXPLORED: (200, 200, 200),
            Cell.FREE: (245, 245, 245),
            Cell.OBSTACLE: (40, 40, 40),
        }

        # Drone colors
        self.drone_colors = [
            (220, 50, 50),
            (50, 100, 220),
            (50, 180, 70),
            (220, 150, 30),
            (160, 60, 180),
            (30, 180, 180),
        ]

    def drone_color(self, drone_id):
        return self.drone_colors[drone_id % len(self.drone_colors)]

    def draw_environment(self, simulator, name, panel_index):
        x_offset = panel_index * self.panel_width

        # Background
        pygame.draw.rect(
            self.screen,
            (245, 245, 245),
            (x_offset, 0, self.panel_width, self.panel_height),
        )

        # Draw grid
        robot_map = simulator.robot_map

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                value = robot_map.get_cell(x, y)
                color = self.grid_colors.get(value, (200, 200, 200))

                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        x_offset + x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

        # Draw drones
        for drone in simulator.drones:
            cx = x_offset + drone.x * self.cell_size + self.cell_size // 2
            cy = drone.y * self.cell_size + self.cell_size // 2
            radius = max(3, self.cell_size // 2)

            pygame.draw.circle(
                self.screen,
                self.drone_color(drone.id),
                (cx, cy),
                radius,
            )

        # Border
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (x_offset, 0, self.panel_width, self.panel_height),
            2,
        )

        # Info panel
        coverage = simulator.get_coverage()
        redundancy = simulator.get_sensing_redundancy()
        distance = simulator.get_total_distance()
        movement_eff = simulator.get_movement_efficiency()
        timestep = simulator.timestep

        # Get strategy-specific metrics if available
        nodes_expanded = 0
        num_clusters = 0
        if hasattr(simulator.strategy, 'get_metrics'):
            metrics = simulator.strategy.get_metrics()
            nodes_expanded = metrics.get('nodes_expanded', 0)
            num_clusters = metrics.get('num_clusters', 0)

        # Draw text
        text_x = x_offset + 10
        text_y = self.panel_height + 8

        self.draw_text(name, text_x, text_y, self.font)
        self.draw_text(f"Step: {timestep}", text_x, text_y + 22, self.small_font)
        self.draw_text(f"Coverage: {coverage:.2f}%", text_x, text_y + 38, self.small_font)
        self.draw_text(f"Redundancy: {redundancy:.2f}%", text_x, text_y + 54, self.small_font)
        self.draw_text(f"Distance: {distance:.0f}", text_x, text_y + 70, self.small_font)
        self.draw_text(f"Movement Eff: {movement_eff:.3f}", text_x, text_y + 86, self.small_font)

        if nodes_expanded > 0:
            self.draw_text(f"Nodes: {nodes_expanded}", text_x, text_y + 102, self.small_font)

    def draw_text(self, text, x, y, font):
        surface = font.render(text, True, (0, 0, 0))
        self.screen.blit(surface, (x, y))

    def run(self):
        print()
        print("=" * 60)
        print(f"CLASSICAL ALGORITHM COMPARISON (SEED = {self.seed})")
        print("=" * 60)

        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    if event.key == pygame.K_SPACE:
                        self.step_delay = 0.0 if self.step_delay > 0 else 0.05

            # Step each simulator
            all_done = True

            for name, simulator in self.simulators.items():
                if self.dones[name]:
                    continue

                all_done = False

                simulator.step()

                # Check termination conditions
                coverage = simulator.get_coverage()
                if coverage >= 90.0 or simulator.timestep >= self.max_steps:
                    self.dones[name] = True
                    print(f"{name} finished at step {simulator.timestep} with {coverage:.2f}% coverage")

            # Draw
            self.screen.fill((255, 255, 255))

            for index, name in enumerate(self.simulators.keys()):
                self.draw_environment(
                    self.simulators[name],
                    name,
                    index,
                )

            pygame.display.flip()

            # Check if all done
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
                            if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                                waiting = False
                                running = False

                    self.clock.tick(30)

            time.sleep(self.step_delay)

        pygame.quit()

    def print_results(self):
        print()
        print("=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)

        for name, simulator in self.simulators.items():
            coverage = simulator.get_coverage()
            redundancy = simulator.get_sensing_redundancy()
            distance = simulator.get_total_distance()
            movement_eff = simulator.get_movement_efficiency()
            steps = simulator.timestep

            nodes_expanded = 0
            if hasattr(simulator.strategy, 'get_metrics'):
                metrics = simulator.strategy.get_metrics()
                nodes_expanded = metrics.get('nodes_expanded', 0)

            print()
            print(f"{name}:")
            print(f"  Steps: {steps}")
            print(f"  Coverage: {coverage:.2f}%")
            print(f"  Distance: {distance:.2f}")
            print(f"  Redundancy: {redundancy:.2f}%")
            print(f"  Movement Efficiency: {movement_eff:.3f}")
            if nodes_expanded > 0:
                print(f"  Nodes Expanded: {nodes_expanded}")


if __name__ == "__main__":
    from strategy.frontier_strategy import FrontierStrategy
    from environment.coordination.cluster_frontier_assigner import ClusterFrontierAssigner
    from environment.coordination.cluster_frontier_utility_assigner import ClusterFrontierUtilityAssigner
    from environment.coordination.hungarian_frontier_assigner import HungarianFrontierAssigner
    from environment.planner.bfs_planner import BFSPlanner
    from environment.planner.a_star_planner import AStarPlanner
    from environment.utils.frontier_utility import FrontierUtility

    # Define utility function
    utility = FrontierUtility(
        alpha=1.0,
        beta=0.5,
        gamma=0.5,
        delta=0.1
    )

    # Define strategies to compare
    strategies = {
        "Cluster Frontier": FrontierStrategy(
            ClusterFrontierAssigner(BFSPlanner())
        ),
        "Cluster Utility": FrontierStrategy(
            ClusterFrontierUtilityAssigner(BFSPlanner(), utility)
        ),
        "Hungarian BFS": FrontierStrategy(
            HungarianFrontierAssigner(planner=BFSPlanner(), utility=utility)
        ),
        "Hungarian A*": FrontierStrategy(
            HungarianFrontierAssigner(planner=AStarPlanner(), utility=utility)
        ),
    }

    visualizer = ClassicalComparisonVisualization(
        strategies=strategies,
        seed=42,
        grid_width=100,
        grid_height=100,
        num_drones=5,
        obstacle_percentage=0.20,
        communication_radius=10,
        max_steps=10000,
        cell_size=6,
        step_delay=0.05,
    )

    visualizer.run()
