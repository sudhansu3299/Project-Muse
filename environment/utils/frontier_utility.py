
'''
U=α⋅Information Gain−β⋅Path Cost−γ⋅Redundancy [+δ⋅Cluster Size] (opt.)
'''

class FrontierUtility:

    COST_SCALE = 200.0
    SIZE_SCALE = 50.0

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

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, value))

    @classmethod
    def normalize_information_gain(cls, ig):
        return cls._clip01(ig)

    @classmethod
    def normalize_cost(cls, cost):
        return cls._clip01(
            cost / cls.COST_SCALE
        )

    @classmethod
    def normalize_redundancy(cls, redundancy):
        return cls._clip01(
            redundancy / 100.0
        )

    @classmethod
    def normalize_cluster_size(cls, size):
        return cls._clip01(
            size / cls.SIZE_SCALE
        )

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