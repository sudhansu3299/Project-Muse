from environment.grid import OccupancyGrid
from agents.drone import Drone


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

        self.true_map.randomize_obstacles(
            obstacle_percentage
        )

        self.robot_map.reset()

        # Create drones
        # Each drone receives the strategy
        self.drones = [
            Drone(0, 0, strategy)
            for _ in range(num_drones)
        ]

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