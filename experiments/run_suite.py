from environment.coordination.hungarian_frontier_assigner import HungarianFrontierAssigner
from environment.planner.bfs_planner import BFSPlanner
from environment.planner.a_star_planner import AStarPlanner

from experiments.run_experiment import run_experiment
from strategy.random_strategy import RandomStrategy
from strategy.frontier_strategy import FrontierStrategy
# from strategy.gso_strategy import GSOStrategy
# from strategy.ppo_strategy import PPOStrategy

from environment.coordination.nearest_frontier_assigner import NearestFrontierAssigner
from environment.coordination.greedy_frontier_assigner import GreedyFrontierAssigner
from environment.coordination.cluster_frontier_assigner import ClusterFrontierAssigner
from environment.coordination.cluster_frontier_utility_assigner  import ClusterFrontierUtilityAssigner

from environment.utils.frontier_utility import FrontierUtility


NUM_RUNS = 2

#U=1.0 * IG − 0.5*Cost − 0.5*Cluster Load
utility = FrontierUtility(
    alpha=1.0,
    beta=0.5,
    gamma=0.5,
)

strategies = {
    # "random": lambda: RandomStrategy(),
    #
    # "nearest_frontier": lambda: FrontierStrategy(
    #     NearestFrontierAssigner()
    # ),
    #
    # "greedy_frontier": lambda: FrontierStrategy(
    #     GreedyFrontierAssigner()
    # ),
    #
    # "cluster_frontier": lambda: FrontierStrategy(
    #     ClusterFrontierAssigner(BFSPlanner())
    # ),
    #
    # "cluster_utility_frontier": lambda: FrontierStrategy(
    #     ClusterFrontierUtilityAssigner(BFSPlanner())
    # ),

    # "hungarian_bfs_frontier": lambda: FrontierStrategy(
    #     HungarianFrontierAssigner(planner=BFSPlanner(), utility=utility)
    # ),
    # "hungarian_astar_frontier": lambda: FrontierStrategy(
    #     HungarianFrontierAssigner(planner=AStarPlanner(), utility=utility)
    # ),

    "hungarian_a": lambda: FrontierStrategy(
        HungarianFrontierAssigner(
            planner=BFSPlanner(),
            utility=FrontierUtility(
                alpha=1.0,
                beta=0.5,
                gamma=0,
                delta=0
            ),
        )
    ),

    "hungarian_b": lambda: FrontierStrategy(
        HungarianFrontierAssigner(
            planner=BFSPlanner(),
            utility=FrontierUtility(
                alpha=1.0,
                beta=0.5,
                gamma=0.5,
                delta=0
            ),
        )
    ),

    "hungarian_c": lambda: FrontierStrategy(
        HungarianFrontierAssigner(
            planner=BFSPlanner(),
            utility=FrontierUtility(
                alpha=1.0,
                beta=0.5,
                gamma=0,
                delta=0.1
            ),
        )
    ),

    "hungarian_d": lambda: FrontierStrategy(
        HungarianFrontierAssigner(
            planner=BFSPlanner(),
            utility=FrontierUtility(
                alpha=1.0,
                beta=0.5,
                gamma=0.5,
                delta=0.1
            ),
        )
    ),
}


for run_id in range(1, NUM_RUNS + 1):

    # One seed = one experimental environment
    map_seed = run_id

    print(f"\n===== RUN {run_id} | SEED {map_seed} =====")

    for strategy_name, strategy_factory in strategies.items():

        strategy = strategy_factory()

        run_experiment(
            strategy=strategy,
            strategy_name=strategy_name,
            run_id=run_id,
            map_seed=map_seed,
        )