from models.action import Action


class PathUtils:

    @staticmethod
    def path_to_action(
            drone,
            path,
            path_index,
    ):

        if path is None:
            return Action.STAY

        if path_index >= len(path):
            return Action.STAY

        current_position = (
            drone.x,
            drone.y,
        )

        # We want the next waypoint after the drone's
        # current position.
        next_index = path_index

        # If the current waypoint is already where the drone is,
        # move toward the following waypoint.
        if path[path_index] == current_position:
            next_index += 1

        if next_index >= len(path):
            return Action.STAY

        next_x, next_y = path[next_index]

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