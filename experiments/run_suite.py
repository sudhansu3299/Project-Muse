from experiments.run_experiment import run_experiment

from strategy.random_strategy import RandomStrategy
# from strategy.frontier_strategy import FrontierStrategy
# from strategy.gso_strategy import GSOStrategy
# from strategy.ppo_strategy import PPOStrategy


NUM_RUNS = 5

strategies = {
    "random": RandomStrategy,
    # "frontier": FrontierStrategy,
    # "gso": GSOStrategy,
    # "ppo": PPOStrategy,
}


for run_id in range(1, NUM_RUNS + 1):

    # One seed = one experimental environment
    map_seed = run_id

    print(f"\n===== RUN {run_id} | SEED {map_seed} =====")

    for strategy_name, StrategyClass in strategies.items():

        strategy = StrategyClass()

        run_experiment(
            strategy=strategy,
            strategy_name=strategy_name,
            run_id=run_id,
            map_seed=map_seed,
        )