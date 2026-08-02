from strategy.frontier_strategy import FrontierStrategy

class CoordinatedFrontierStrategy(
    FrontierStrategy
):

    def __init__(self, frontier_assigner):
        super().__init__(frontier_assigner)
