from environment.coordination.cluster_frontier_assigner import ClusterFrontierAssigner
from models.constants import Cell
class ClusterFrontierUtilityAssigner(ClusterFrontierAssigner):

    def __init__(
            self,
            planner,
            utility,
    ):
        super().__init__(planner)

        self.utility = utility

    def _cluster_utility(
            self,
            drone,
            cluster,
            cost_matrix,
            drones,
            robot_map,
    ):

        entry = cost_matrix[
            drone.id
        ][
            cluster.id
        ]

        cost = entry["cost"]
        ig = entry["ig"]

        if cost == float("inf"):
            return float("-inf")

        predicted_redundancy = self._estimate_redundancy(
            drone,
            cluster.centroid,
            robot_map
        )

        cluster_size = len(cluster.cells)

        return self.utility.calculate(
            base_information_gain=ig,
            path_cost=cost,
            redundancy=predicted_redundancy,
            cluster_size=cluster_size,
        )

    def _estimate_redundancy(
            self,
            drone,
            target,
            robot_map,
    ):

        sensor_cells = self._get_sensor_cells(
            target,
            robot_map,
        )

        if not sensor_cells:
            return 0.0

        known_cells = sum(
            1
            for x, y in sensor_cells
            if robot_map.get_cell(x, y)
            != Cell.UNEXPLORED
        )

        return (
                known_cells
                / len(sensor_cells)
        )

    def _get_sensor_cells(
            self,
            position,
            robot_map,
            sensor_radius=3,
    ):
        """
        Returns the cells sensed by a drone at a given position.

        Uses a circular sensing footprint.
        """

        x, y = position

        cells = set()

        for dy in range(
                -sensor_radius,
                sensor_radius + 1,
        ):
            for dx in range(
                    -sensor_radius,
                    sensor_radius + 1,
            ):

                nx = x + dx
                ny = y + dy

                if not robot_map.is_inside(
                        nx,
                        ny,
                ):
                    continue

                distance_squared = (
                        dx * dx + dy * dy
                )

                if (
                        distance_squared
                        > sensor_radius ** 2
                ):
                    continue

                cells.add((nx, ny))

        return cells