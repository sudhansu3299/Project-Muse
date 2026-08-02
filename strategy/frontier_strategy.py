from environment.planner.bfs_planner import BFSPlanner
from environment.planner.frontier_detector import FrontierDetector
from environment.planner.path_utils import PathUtils
from strategy.exploration_strategy import ExplorationStrategy
from models.constants import Cell
from models.action import Action
from collections import deque

class FrontierStrategy(ExplorationStrategy):
    def __init__(self, communication_radius=10):
        super().__init__("Frontier")

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

        bfs_planner = BFSPlanner()
        path_utils = PathUtils()

        path = bfs_planner.reconstruct_path(
            came_from,
            target
        )

        action = path_utils.path_to_action(
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

    def find_nearest_frontier(
            self,
            drone,
            robot_map
    ):

        start = (
            drone.x,
            drone.y
        )

        frontier_detector = FrontierDetector()

        frontiers = set(
            frontier_detector.detect_frontiers(robot_map)
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