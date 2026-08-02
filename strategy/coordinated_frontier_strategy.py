from strategy.frontier_strategy import FrontierStrategy


class CoordinatedFrontierStrategy(FrontierStrategy):
    def __init__(self):
        super().__init__()
        self.assignments = {}

    def assign_frontiers(
            self,
            drones,
            robot_map
    ):
        frontiers = self.detect_frontiers(robot_map)


    def prepare_step(
            self,
            drones,
            robot_map
    ):
        self.assign_frontiers(
            drones,
            robot_map
        )

