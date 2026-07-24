from experiments.run_experiment import run_experiment
from strategy.random_strategy import RandomStrategy

'''
This is just for testing random algo run, 
by using run_experiment method
'''
NUM_RUNS = 5

for run_id in range(1, NUM_RUNS + 1):

    strategy = RandomStrategy()

    summary = run_experiment(
        strategy=strategy,
        strategy_name="random",
        run_id=run_id,
        map_seed=run_id,
    )

    print(summary)
