
'''
U=α⋅Information Gain−β⋅Path Cost−γ⋅Redundancy [+δ⋅Cluster Size] (opt.)
'''

class FrontierUtility:

    def __init__(
            self,
            alpha=1.0,
            beta=0.5,
            gamma=0.5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def set_weights(
            self,
            alpha,
            beta,
            gamma,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate(
            self,
            information_gain,
            path_cost,
            cluster_load,
    ):

        if path_cost == float("inf"):
            return float("-inf")

        utility = (
                self.alpha * information_gain
                - self.beta * path_cost
                - self.gamma * cluster_load
        )

        print(
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"gamma={self.gamma}, "
            f"IG={information_gain:.2f}, "
            f"cost={path_cost}, "
            f"load={cluster_load}, "
            f"utility={utility:.2f}"
        )

        return utility