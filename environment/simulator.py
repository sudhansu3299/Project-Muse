from environment.grid import OccupancyGrid
from agents.drone import Drone
from models.constants import Cell


# Simulator captures the state of the environment
class Simulator:

    def __init__(
            self,
            grid_width,
            grid_height,
            num_drones,
            obstacle_percentage,
            strategy,
            communication_radius=10,
            map_seed=None,
    ):

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.communication_radius = communication_radius

        # Create maps
        self.true_map = OccupancyGrid(
            grid_height,
            grid_width
        )

        self.robot_map = OccupancyGrid(
            grid_height,
            grid_width
        )

        self.map_seed = map_seed

        self.true_map.randomize_obstacles(
            obstacle_percentage,
            seed=map_seed
        )

        #This is so that the spawned drones are not trapped inside the obstacle
        center_x = grid_width // 2
        center_y = grid_height // 2

        spawn_clearance = 3

        for dy in range(
                -spawn_clearance,
                spawn_clearance + 1
        ):
            for dx in range(
                    -spawn_clearance,
                    spawn_clearance + 1
            ):

                x = center_x + dx
                y = center_y + dy

                if self.true_map.is_inside(x, y):
                    self.true_map.set_cell(
                        x,
                        y,
                        Cell.FREE
                    )
        start_positions = [
            (center_x, center_y),
            (center_x + 1, center_y),
            (center_x - 1, center_y),
            (center_x, center_y + 1),
            (center_x, center_y - 1),
        ]

        self.robot_map.reset()

        # Create drones
        # Each drone receives the strategy and ensure that all of them start from different starting points
        self.drones = []

        for i in range(num_drones):

            x, y = start_positions[
                i % len(start_positions)
                ]

            drone = Drone(
                x,
                y,
                strategy
            )

            drone.id = i

            self.drones.append(drone)

        self.timestep = 0

    def step(self):
        """
        Advance the entire simulation by one timestep.
        """

        for drone in self.drones:

            neighbours = self.get_nearby_agents(
                drone
            )

            # Drone asks its strategy for an action,
            # executes it, and updates the robot map.
            drone.step(
                self.true_map,
                self.robot_map,
                neighbours
            )

        self.timestep += 1

    def get_nearby_agents(self, drone):
        """
        Return drones within communication radius.
        """

        nearby_agents = []

        for other in self.drones:

            if other is drone:
                continue

            dx = drone.x - other.x
            dy = drone.y - other.y

            distance_squared = (
                    dx * dx +
                    dy * dy
            )

            if (
                    distance_squared
                    <= self.communication_radius ** 2
            ):
                nearby_agents.append(other)

        return nearby_agents

    def get_coverage(self):

        known_cells = 0
        total_cells = (
                self.grid_width
                * self.grid_height
        )

        for y in range(self.grid_height):
            for x in range(self.grid_width):

                if (
                        self.robot_map.get_cell(x, y)
                        != Cell.UNEXPLORED
                ):
                    known_cells += 1

        return (
                known_cells
                / total_cells
        ) * 100

    def get_total_distance(self):

        return sum(
            drone.total_distance
            for drone in self.drones
        )

    def get_overlap_percentage(self):

        total_sensed = sum(
            drone.total_sensed_cells
            for drone in self.drones
        )

        redundant_sensed = sum(
            drone.redundant_sensed_cells
            for drone in self.drones
        )

        if total_sensed == 0:
            return 0.0

        return (
                redundant_sensed
                / total_sensed
        ) * 100