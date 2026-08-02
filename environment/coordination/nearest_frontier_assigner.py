from collections import deque

from environment.coordination.frontier_assigner import FrontierAssigner
from models.constants import Cell


class NearestFrontierAssigner(FrontierAssigner):

    def assign(
            self,
            drones,
            robot_map,
    ):

        frontiers = set(
            self.frontier_detector.detect_frontiers(
                robot_map
            )
        )

        assignments = {}

        for drone in drones:

            target = self.find_nearest_frontier(
                drone,
                frontiers,
                robot_map,
            )

            assignments[drone.id] = target

        return assignments

    def find_nearest_frontier(
            self,
            drone,
            frontiers,
            robot_map,
    ):

        start = (
            drone.x,
            drone.y,
        )

        if not frontiers:
            return None

        queue = deque([start])

        visited = {start}

        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ]

        while queue:

            current = queue.popleft()

            # Found nearest frontier
            if current in frontiers:
                return current

            x, y = current

            for dx, dy in directions:

                nx = x + dx
                ny = y + dy

                if not robot_map.is_inside(nx, ny):
                    continue

                if robot_map.get_cell(nx, ny) != Cell.FREE:
                    continue

                next_cell = (nx, ny)

                if next_cell in visited:
                    continue

                visited.add(next_cell)
                queue.append(next_cell)

        return None