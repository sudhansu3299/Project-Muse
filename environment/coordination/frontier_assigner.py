from abc import ABC, abstractmethod


class FrontierAssigner(ABC):

    @abstractmethod
    def assign(
            self,
            drones,
            frontiers,
            robot_map,
    ):
        """
        Returns

        {
            drone_id : frontier_cell
        }
        """
        pass