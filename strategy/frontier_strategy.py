from strategy.exploration_strategy import ExplorationStrategy
from models.constants import Cell
from models.action import Action
from collections import deque

class FrontierStrategy(ExplorationStrategy):
    def __init__(self, communication_radius=10):
        super().__init__("Frontier")

    def detect_frontiers(self, robot_map):
        """
        Detect frontier cells.

        A frontier is a known FREE cell that has at least
        one UNEXPLORED 4-connected neighbour.
        """
        frontiers = []

        for y in range(robot_map.height):
            for x in range(robot_map.width):

                if robot_map.get_cell(x, y) != Cell.FREE:
                    continue

                if self.has_unexplored_neighbour(x, y, robot_map):
                    frontiers.append((x, y))

        return frontiers

    def has_unexplored_neighbour(self, x, y, robot_map):

        for action in [
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]:
            dx, dy = action.delta()

            new_x = x + dx
            new_y = y + dy

            if not robot_map.is_inside(new_x, new_y):
                continue

            if robot_map.get_cell(new_x, new_y) == Cell.UNEXPLORED:
                return True

        return False

    def choose_action(
            self,
            drone,
            robot_map,
            true_map,
            nearby_agents,
    ):

        result = self.find_nearest_frontier(
            drone,
            robot_map
        )

        if result is None:
            print(
                f"Drone {drone.id} at ({drone.x}, {drone.y}): "
                f"NO REACHABLE FRONTIER"
            )
            return Action.STAY

        target, came_from = result

        path = self.reconstruct_path(
            came_from,
            target
        )

        action = self.path_to_action(
            drone,
            path
        )

        print(
            f"Drone {drone.id}: "
            f"pos=({drone.x},{drone.y}), "
            f"target={target}, "
            f"path={path}, "
            f"action={action}"
        )

        return action

    def path_to_action(
            self,
            drone,
            path
    ):

        if path is None or len(path) < 2:
            return Action.STAY

        next_x, next_y = path[1]

        dx = next_x - drone.x
        dy = next_y - drone.y

        if dx == 1:
            return Action.RIGHT

        if dx == -1:
            return Action.LEFT

        if dy == 1:
            return Action.DOWN

        if dy == -1:
            return Action.UP

        return Action.STAY


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

    def find_nearest_frontier(
            self,
            drone,
            robot_map
    ):

        start = (
            drone.x,
            drone.y
        )

        frontiers = set(
            self.detect_frontiers(robot_map)
        )

        print("\n--- FRONTIER DEBUG ---")
        print("Drone:", start)
        print("Drone cell:", robot_map.get_cell(*start))
        print("Frontiers:", frontiers)

        for y in range(5):
            row = []

            for x in range(5):
                row.append(
                    str(robot_map.get_cell(x, y))
                )

            print(y, row)

        # Current position is not a useful target
        frontiers.discard(start)

        if not frontiers:
            return None

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

            print("BFS visiting:", current)

            if current in frontiers:
                return current, came_from

            x, y = current

            for dx, dy in directions:

                nx = x + dx
                ny = y + dy

                next_cell = (nx, ny)

                if not robot_map.is_inside(nx, ny):
                    continue

                if (
                        robot_map.get_cell(nx, ny)
                        != Cell.FREE
                ):
                    continue

                if next_cell in came_from:
                    continue

                came_from[next_cell] = current
                queue.append(next_cell)

        return None

    def find_path(
            self,
            start,
            goal,
            robot_map
    ):

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