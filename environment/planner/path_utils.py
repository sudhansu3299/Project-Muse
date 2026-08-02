from models.action import Action

class PathUtils:
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
