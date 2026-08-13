from environment.coordination.cluster_frontier_assigner import ClusterFrontierAssigner
from models.constants import Cell
from environment.coordination.frontier_assigner import FrontierAssigner
from environment.utils.frontier_clusterer import FrontierClusterer


class ClusterFrontierUtilityAssigner(ClusterFrontierAssigner):

    def __init__(
            self,
            planner,
            utility,
    ):
        super().__init__(planner)

        self.utility = utility

    def _cluster_utility(
            self,
            drone,
            cluster,
            cost_matrix,
    ):

        entry = cost_matrix[
            drone.id
        ][
            cluster.id
        ]

        cost = entry["cost"]
        ig = entry["ig"]

        if cost == float("inf"):
            return float("-inf")

        load = self.cluster_assignment_counts[
            cluster.id
        ]

        return self.utility.calculate(
            information_gain=ig,
            path_cost=cost,
            cluster_load=load,
        )