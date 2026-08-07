from environment.planner.bfs_planner import BFSPlanner
from environment.utils.path_utils import PathUtils
from strategy.exploration_strategy import ExplorationStrategy
from models.action import Action

class FrontierStrategy(ExplorationStrategy):
    def __init__(self, frontier_assigner, communication_radius=10):
        super().__init__("Frontier")

        self.frontier_assigner = frontier_assigner
        self.assignments = {}
        # self.paths = {}  # Cache paths for each drone

    def prepare_step(
            self,
            drones,
            robot_map,
    ):
        """
        Only assign clusters when drones need new assignments:
        - First time (no assignments)
        - Drone has reached its assigned centroid
        """
        
        # Check if any drone needs reassignment
        needs_reassignment = False
        
        for drone in drones:
            # No assignment yet
            if drone.id not in self.assignments:
                needs_reassignment = True
                break

            assignment = self.assignments[drone.id]

            if assignment is None:
                needs_reassignment = True
                break

            # target = assignment["target"]
            #
            # if target is None:
            #     needs_reassignment = True
            #     break
            #
            # if (drone.x, drone.y) == target:
            #     needs_reassignment = True
            #     break

            path = assignment["path"]

            if path is None:
                needs_reassignment = True
                break

            if assignment["path_index"] >= len(path) - 1:
                needs_reassignment = True
                break

        if needs_reassignment:
            self.assignments = self.frontier_assigner.assign(
                drones,
                robot_map,
            )
            
            # # Rebuild paths for new assignments
            # bfs_planner = BFSPlanner()
            # self.paths = {}
            #
            # for drone in drones:
            #     cluster = self.assignments.get(drone.id)
            #     if cluster is not None:
            #         path = bfs_planner.find_path(
            #             start=(drone.x, drone.y),
            #             goal=cluster.centroid,
            #             robot_map=robot_map,
            #         )
            #         self.paths[drone.id] = path

    def choose_action(
            self,
            drone,
            robot_map,
            true_map,
            nearby_agents,
    ):

        assignment = self.assignments.get(
            drone.id
        )

        if assignment is None:
            return Action.STAY

        path = assignment["path"]

        if path is None:
            return Action.STAY

        index = assignment["path_index"]

        if index >= len(path) - 1:
            return Action.STAY

        if (
                index < len(path) - 1
                and
                (drone.x, drone.y) == path[index + 1]
        ):
            assignment["path_index"] += 1
            index += 1

        path_utils = PathUtils()

        return path_utils.path_to_action(
            drone,
            path,
            index,
        )