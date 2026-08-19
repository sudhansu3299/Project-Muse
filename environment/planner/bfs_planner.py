from collections import deque
from models.constants import Cell

class BFSPlanner:
    def __init__(self):
        self.nodes_expanded = 0

    def find_path(
            self,
            start,
            goal,
            robot_map
    ):

        self.nodes_expanded = 0
        queue = deque([start])

        came_from = {
            start: None
        }

        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ]

        while queue:

            current = queue.popleft()
            self.nodes_expanded += 1

            if current == goal:
                break

            x, y = current

            for dx, dy in directions:

                nx = x + dx
                ny = y + dy

                next_cell = (nx, ny)

                if not robot_map.is_inside(nx, ny):
                    continue

                # Navigate ONLY through known free space
                if (
                        robot_map.get_cell(nx, ny)
                        != Cell.FREE
                ):
                    continue

                if next_cell in came_from:
                    continue

                came_from[next_cell] = current

                queue.append(next_cell)

        # Goal wasn't reachable
        if goal not in came_from:
            return None

        path = []

        current = goal

        while current is not None:

            path.append(current)

            current = came_from[current]

        path.reverse()

        return path

    def reconstruct_path(
            self,
            came_from,
            goal
    ):

        path = []

        current = goal

        while current is not None:

            path.append(current)

            current = came_from[current]

        path.reverse()

        return path

    def path_length(
            self,
            drone,
            goal,
            robot_map,
    ):
        path = self.find_path(
            start=(drone.x, drone.y),
            goal=goal,
            robot_map=robot_map,
        )

        if path is None:
            return float("inf")

        return len(path) - 1