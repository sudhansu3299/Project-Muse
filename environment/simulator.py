from collections import Counter
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
        self.strategy = strategy

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

        self.robot_map.reset()

        # Create drones with different start positions
        start_positions = self._find_start_positions(num_drones)

        self.drones = []

        for drone_id in range(num_drones):
            drone = Drone(
                x=start_positions[drone_id][0],
                y=start_positions[drone_id][1],
                strategy=strategy,
            )
            drone.id = drone_id
            self.drones.append(drone)

        self.timestep = 0

    def _is_safe_start_cell(self, x, y):
        """
        Check that the start cell has enough free space around it.
        """

        if not self.true_map.is_inside(x, y):
            return False

        if self.true_map.get_cell(x, y) != Cell.FREE:
            return False

        free_neighbors = 0

        for dx, dy in [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ]:
            nx = x + dx
            ny = y + dy

            if (
                    self.true_map.is_inside(nx, ny)
                    and self.true_map.get_cell(nx, ny) == Cell.FREE
            ):
                free_neighbors += 1

        return free_neighbors >= 2

    def _find_start_positions(self, num_drones):
        """
        Find a safe starting formation near the top-left corner.

        All drones start in the same local region so that the only
        experimental change is the swarm's initial location.
        """

        positions = []

        # Search progressively farther from the top-left corner
        for radius in range(1, 15):

            for y in range(1, radius + 1):
                for x in range(1, radius + 1):

                    if not self._is_safe_start_cell(x, y):
                        continue

                    if (x, y) not in positions:
                        positions.append((x, y))

                    if len(positions) >= num_drones:
                        return positions

        raise RuntimeError(
            f"Could not find {num_drones} safe start positions "
            "near the top-left corner."
        )

    def step(self):
        """
        Advance the entire simulation by one timestep.
        """

        self.strategy.prepare_step(
            self.drones,
            self.robot_map
        )

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

    def get_sensing_redundancy(self):
        """
        Calculates sensing redundancy as percentage of redundant sensing.
        Returns the percentage of cells that were sensed multiple times.
        """
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

    def get_visit_overlap_percentage(self):
        """
        Calculates visit overlap percentage - how many times drones physically
        visited the same cells.
        """
        visit_counts = Counter()

        for drone in self.drones:
            for cell in drone.visited_cells:
                visit_counts[cell] += 1

        total_visits = sum(visit_counts.values())

        if total_visits == 0:
            return 0.0

        redundant_visits = sum(
            count - 1
            for count in visit_counts.values()
            if count > 1
        )

        return (
            redundant_visits
            / total_visits
            * 100
        )

    def get_active_drones_count(self):
        """
        Returns the number of drones with valid assignments.
        """
        if hasattr(self.strategy, 'get_metrics'):
            metrics = self.strategy.get_metrics()
            return metrics.get('num_assigned_drones', len(self.drones))
        return len(self.drones)

    def get_exploration_efficiency(self):
        """
        Calculates aggregate exploration efficiency as:

            unique physically visited cells
            --------------------------------
                 total distance travelled

        Higher is better.
        """

        total_distance = self.get_total_distance()

        if total_distance <= 0:
            return 0.0

        unique_visited = set()

        for drone in self.drones:
            unique_visited.update(
                drone.visited_cells
            )

        return (
                len(unique_visited)
                / total_distance
        )

    def get_mean_pairwise_distance(self):
        """
        Calculates the mean pairwise distance between all drones.
        For n drones, there are C(n,2) pairs.
        Returns the average Euclidean distance between all drone pairs.
        """
        if len(self.drones) < 2:
            return 0.0

        total_distance = 0.0
        pair_count = 0

        for i in range(len(self.drones)):
            for j in range(i + 1, len(self.drones)):
                drone1 = self.drones[i]
                drone2 = self.drones[j]

                # Calculate Euclidean distance
                dx = drone1.x - drone2.x
                dy = drone1.y - drone2.y
                distance = (dx ** 2 + dy ** 2) ** 0.5

                total_distance += distance
                pair_count += 1

        if pair_count == 0:
            return 0.0

        return total_distance / pair_count

    def get_movement_efficiency(self):
        """
        Calculates movement efficiency as unique visited cells / total distance.
        Returns the movement efficiency (cells visited per unit distance).
        """
        total_distance = self.get_total_distance()

        if total_distance == 0:
            return 0.0

        # Count unique visited cells across all drones
        unique_visited = set()
        for drone in self.drones:
            unique_visited.update(drone.visited_cells)

        return len(unique_visited) / total_distance