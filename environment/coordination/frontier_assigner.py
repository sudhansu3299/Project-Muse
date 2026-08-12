from abc import ABC, abstractmethod

from environment.utils.frontier_detector import FrontierDetector
from environment.planner.bfs_planner import BFSPlanner
from environment.planner.a_star_planner import AStarPlanner



class FrontierAssigner(ABC):

    def __init__(self, planner):

        self.frontier_detector = FrontierDetector()
        self.bfs_planner = BFSPlanner()
        self.planner = planner

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