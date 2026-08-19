import heapq
from models.constants import Cell

'''
Started with Manhattan distance as heuristic as BFS does the same
'''
#TODO: Use Euclidean and octile distance as heuristic

class AStarPlanner:

    def __init__(self):
        self.nodes_expanded = 0

    def find_path(
            self,
            start,
            goal,
            robot_map,
    ):

        nodes_before = self.nodes_expanded

        if start == goal:
            return [start]

        # Priority queue:
        # (f_score, g_score, cell)
        open_set = []

        heapq.heappush(
            open_set,
            (
                self._heuristic(start, goal),
                0,
                start,
            )
        )

        came_from = {
            start: None
        }

        g_score = {
            start: 0
        }

        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ]

        while open_set:

            _, current_g, current = heapq.heappop(
                open_set
            )

            # Ignore stale queue entries
            if current_g > g_score[current]:
                continue

            self.nodes_expanded += 1

        # Goal reached
            if current == goal:

                nodes_this_search = (
                        self.nodes_expanded
                        - nodes_before
                )

                # print(
                #     f"A*: {start} -> {goal}, "
                #     f"nodes expanded = {nodes_this_search}"
                # )

                return self._reconstruct_path(
                    came_from,
                    current,
                )

            x, y = current

            for dx, dy in directions:

                nx = x + dx
                ny = y + dy

                if not robot_map.is_inside(nx, ny):
                    continue

                if robot_map.get_cell(nx, ny) != Cell.FREE:
                    continue

                neighbor = (nx, ny)

                tentative_g = (
                        g_score[current] + 1
                )

                if (
                        neighbor not in g_score
                        or tentative_g < g_score[neighbor]
                ):

                    came_from[neighbor] = current

                    g_score[neighbor] = tentative_g

                    h = self._heuristic(
                        neighbor,
                        goal,
                    )

                    f = tentative_g + h

                    heapq.heappush(
                        open_set,
                        (
                            f,
                            tentative_g,
                            neighbor,
                        )
                    )

        # No path exists
        return None

    def _heuristic(
            self,
            current,
            goal,
    ):

        return (
                abs(current[0] - goal[0])
                +
                abs(current[1] - goal[1])
        )

    def _reconstruct_path(
            self,
            came_from,
            goal,
    ):

        path = []

        current = goal

        while current is not None:

            path.append(current)

            current = came_from[current]

        path.reverse()

        return path