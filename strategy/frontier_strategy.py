from environment.utils.path_utils import PathUtils
from strategy.exploration_strategy import ExplorationStrategy
from models.action import Action


class FrontierStrategy(ExplorationStrategy):

    def __init__(self, frontier_assigner):
        super().__init__("Frontier")

        self.frontier_assigner = frontier_assigner
        self.assignments = {}

    def prepare_step(
            self,
            drones,
            robot_map,
    ):

        # --------------------------------------------------
        # First assignment
        # --------------------------------------------------

        if not self.assignments:

            new_assignments = self.frontier_assigner.assign(
                drones,
                robot_map,
            )

            self.assignments.update(
                new_assignments
            )

        # --------------------------------------------------
        # Reassign only drones whose current path is invalid
        # or finished
        # --------------------------------------------------

        drones_to_reassign = []

        for drone in drones:

            assignment = self.assignments.get(
                drone.id
            )

            if assignment is None:
                continue

            path = assignment["path"]

            # No path -> try again
            if path is None:
                drones_to_reassign.append(drone)
                continue

            index = assignment["path_index"]

            # Path completed -> get a new frontier
            if index >= len(path) - 1:
                drones_to_reassign.append(drone)

        # Batch reassign all drones that need new paths
        if drones_to_reassign:
            new_assignments = self.frontier_assigner.assign(
                drones_to_reassign,
                robot_map,
            )

            self.assignments.update(new_assignments)

    def choose_action(
            self,
            drone,
            robot_map,
            true_map,
            nearby_agents,
    ):


        assignment = self.assignments.get(drone.id)

        if assignment is None:
            return Action.STAY

        path = assignment["path"]

        if path is None or len(path) < 2:
            return Action.STAY

        index = assignment["path_index"]

        if index >= len(path) - 1:
            return Action.STAY

        next_cell = path[index + 1]

        dx = next_cell[0] - drone.x
        dy = next_cell[1] - drone.y

        if dx == 1:
            self.assignments[drone.id]["path_index"] += 1
            return Action.RIGHT

        if dx == -1:
            self.assignments[drone.id]["path_index"] += 1
            return Action.LEFT

        if dy == 1:
            self.assignments[drone.id]["path_index"] += 1
            return Action.DOWN

        if dy == -1:
            self.assignments[drone.id]["path_index"] += 1
            return Action.UP

        return Action.STAY