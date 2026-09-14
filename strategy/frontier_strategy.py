from environment.utils.path_utils import PathUtils
from strategy.exploration_strategy import ExplorationStrategy
from models.action import Action
from environment.coordination.nearest_frontier_assigner import NearestFrontierAssigner


class FrontierStrategy(ExplorationStrategy):

    def __init__(self, frontier_assigner):
        super().__init__("Frontier")

        self.frontier_assigner = frontier_assigner
        self.assignments = {}
        self.failed_reassignment_count = {}  # Track consecutive failures per drone
        self.greedy_fallback = NearestFrontierAssigner(self.frontier_assigner.planner)

    def reset(self):

        self.assignments = {}
        self.failed_reassignment_count = {}

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

            for drone_id, assignment in new_assignments.items():
                if assignment["path"] is not None:
                    self.assignments[drone_id] = assignment
                    self.failed_reassignment_count[drone_id] = 0  # Reset failure counter on success
                else:
                    # If no valid new assignment, set path to None to force immediate reassignment next step
                    # This prevents drones from staying stuck with completed paths
                    if drone_id in self.assignments:
                        self.assignments[drone_id]["path"] = None
                        self.assignments[drone_id]["path_index"] = 0
                        self.failed_reassignment_count[drone_id] = self.failed_reassignment_count.get(drone_id, 0) + 1

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
            # Fallback: use greedy nearest frontier assignment
            return self._get_greedy_fallback_assignment(drone, robot_map)

        index = assignment["path_index"]

        if index >= len(path) - 1:
            # Path completed - immediately try to get a new assignment
            # instead of staying idle for a step
            new_assignments = self.frontier_assigner.assign(
                [drone],
                robot_map,
            )
            if new_assignments.get(drone.id) and new_assignments[drone.id]["path"] is not None:
                self.assignments[drone.id] = new_assignments[drone.id]
                self.failed_reassignment_count[drone.id] = 0
                # Try to execute the first step of the new path
                new_path = self.assignments[drone.id]["path"]
                if len(new_path) >= 2:
                    next_cell = new_path[1]
                    dx = next_cell[0] - drone.x
                    dy = next_cell[1] - drone.y
                    self.assignments[drone.id]["path_index"] = 1
                    if dx == 1:
                        return Action.RIGHT
                    if dx == -1:
                        return Action.LEFT
                    if dy == 1:
                        return Action.DOWN
                    if dy == -1:
                        return Action.UP
            # If reassignment failed, use greedy fallback
            return self._get_greedy_fallback_assignment(drone, robot_map)

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

    def _get_greedy_fallback_assignment(self, drone, robot_map):
        """
        Use greedy nearest frontier as fallback when main assigner fails.
        This is exploration-aware and reduces redundancy compared to random movement.
        """
        greedy_assignment = self.greedy_fallback.assign([drone], robot_map)
        if greedy_assignment.get(drone.id) and greedy_assignment[drone.id]["path"] is not None:
            self.assignments[drone.id] = greedy_assignment[drone.id]
            self.failed_reassignment_count[drone.id] = 0
            # Execute first step of the greedy path
            path = self.assignments[drone.id]["path"]
            if len(path) >= 2:
                next_cell = path[1]
                dx = next_cell[0] - drone.x
                dy = next_cell[1] - drone.y
                self.assignments[drone.id]["path_index"] = 1
                if dx == 1:
                    return Action.RIGHT
                if dx == -1:
                    return Action.LEFT
                if dy == 1:
                    return Action.DOWN
                if dy == -1:
                    return Action.UP
        return Action.STAY

    def get_metrics(self):
        """
        Returns metrics from the frontier assigner if available.
        """
        if hasattr(self.frontier_assigner, 'get_metrics'):
            return self.frontier_assigner.get_metrics()
        return {}