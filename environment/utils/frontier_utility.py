
'''
U=α⋅Information Gain−β⋅Path Cost−γ⋅Redundancy [+δ⋅Cluster Size] (opt.)
'''

class FrontierUtility:

    def __init__(
            self,
            alpha=1.0,
            beta=0.5,
            gamma=0.5,
            delta=0.1,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def set_weights(
            self,
            alpha,
            beta,
            gamma,
            delta,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    @staticmethod
    def normalize_min_max(values):
        """
        Normalize a list of values using min-max normalization to [0, 1].
        Returns a list of normalized values.
        """
        if not values:
            return values

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return [0.0] * len(values)

        return [
            (v - min_val) / (max_val - min_val)
            for v in values
        ]

    def calculate(
            self,
            base_information_gain,
            path_cost,
            redundancy,
            cluster_size,
    ):
        """
        Calculate utility using normalized metrics.
        All input values should be normalized to [0, 1] range using normalize_min_max.
        """
        if path_cost == float("inf"):
            return float("-inf")

        utility = (
                self.alpha * base_information_gain
                - self.beta * path_cost
                - self.gamma * redundancy
                + self.delta * cluster_size
        )

        # print(
        #     f"alpha={self.alpha}, "
        #     f"beta={self.beta}, "
        #     f"gamma={self.gamma}, "
        #     f"IG={base_information_gain:.2f}, "
        #     f"cost={path_cost}, "
        #     f"redundancy={redundancy}, "
        #     f"utility={utility:.2f}"
        # )

        return utility