from models.action import Action


class PathUtils:

    def path_to_action(
            self,
            drone,
            path,
            path_index,
    ):

        if path is None:
            return Action.STAY

        if path_index >= len(path) - 1:
            return Action.STAY

        # -------- Safety check --------
        if path[path_index] != (drone.x, drone.y):
            return Action.STAY

        current_x, current_y = path[path_index]
        next_x, next_y = path[path_index + 1]

        dx = next_x - current_x
        dy = next_y - current_y

        if dx == 1:
            return Action.RIGHT

        if dx == -1:
            return Action.LEFT

        if dy == 1:
            return Action.DOWN

        if dy == -1:
            return Action.UP

        return Action.STAY