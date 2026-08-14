
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

    def calculate(
            self,
            base_information_gain,
            path_cost,
            redundancy,
            cluster_size,
    ):

        if path_cost == float("inf"):
            return float("-inf")

        utility = (
                self.alpha * base_information_gain
                - self.beta * path_cost
                - self.gamma * redundancy
                + self.delta * cluster_size
        )

        print(
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"gamma={self.gamma}, "
            f"IG={base_information_gain:.2f}, "
            f"cost={path_cost}, "
            f"load={redundancy}, "
            f"utility={utility:.2f}"
        )

        return utility