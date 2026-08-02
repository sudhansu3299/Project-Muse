from models.constants import Cell
from models.action import Action

class FrontierDetector:

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
