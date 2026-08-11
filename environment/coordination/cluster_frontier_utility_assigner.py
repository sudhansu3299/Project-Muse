#U=α⋅Information Gain−β⋅Path Cost−γ⋅Redundancy [+δ⋅Cluster Size] (opt.)
from environment.coordination.cluster_frontier_assigner import ClusterFrontierAssigner
from models.constants import Cell
from environment.coordination.frontier_assigner import FrontierAssigner
from environment.utils.frontier_clusterer import FrontierClusterer


class ClusterFrontierUtilityAssigner(ClusterFrontierAssigner):

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

        # TODO: normalize these later
        alpha = 1.0
        beta = 0.5
        gamma = 0.5

        return (
                alpha * ig
                - beta * cost
                - gamma * load
        )