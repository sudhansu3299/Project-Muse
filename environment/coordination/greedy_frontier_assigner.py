from environment.coordination.nearest_frontier_assigner import (
    NearestFrontierAssigner
)


class GreedyFrontierAssigner(
    NearestFrontierAssigner
):

    def assign(
            self,
            drones,
            robot_map,
    ):

        remaining_frontiers = set(
            self.frontier_detector.detect_frontiers(
                robot_map
            )
        )

        assignments = {}

        for drone in drones:

            target = self.find_nearest_frontier(
                drone,
                remaining_frontiers,
                robot_map,
            )

            assignments[drone.id] = target

            if target is not None:
                remaining_frontiers.remove(target)

        return assignments