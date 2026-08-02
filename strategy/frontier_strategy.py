from environment.planner.bfs_planner import BFSPlanner
from environment.planner.path_utils import PathUtils
from strategy.exploration_strategy import ExplorationStrategy
from models.action import Action

class FrontierStrategy(ExplorationStrategy):
    def __init__(self, frontier_assigner, communication_radius=10):
        super().__init__("Frontier")

        self.frontier_assigner = frontier_assigner
        self.assignments = {}

    def prepare_step(
            self,
            drones,
            robot_map,
    ):
        """
        Default implementation.

        Independent frontier simply asks the assigner
        to assign a frontier to every drone independently.
        """

        self.assignments = self.frontier_assigner.assign(
            drones,
            robot_map,
        )

    def choose_action(
            self,
            drone,
            robot_map,
            true_map,
            nearby_agents,
    ):

        target = self.assignments.get(
            drone.id
        )

        if target is None:
            return Action.STAY

        bfs_planner = BFSPlanner()

        path = bfs_planner.find_path(
            start=(drone.x, drone.y),
            goal=target,
            robot_map=robot_map,
        )

        if path is None:
            return Action.STAY

        path_utils = PathUtils()

        action = path_utils.path_to_action(
            drone,
            path,
        )

        print(
            f"Drone {drone.id}: "
            f"pos=({drone.x},{drone.y}), "
            f"target={target}, "
            f"path={path}, "
            f"action={action}"
        )

        return action