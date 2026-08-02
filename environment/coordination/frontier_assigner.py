from abc import ABC, abstractmethod

from environment.planner.frontier_detector import FrontierDetector
from environment.planner.bfs_planner import BFSPlanner


class FrontierAssigner(ABC):

    def __init__(self):

        self.frontier_detector = FrontierDetector()
        self.bfs_planner = BFSPlanner()

    @abstractmethod
    def assign(
            self,
            drones,
            robot_map,
    ):
        """
        Returns

        {
            drone_id : frontier_cell
        }
        """
        pass