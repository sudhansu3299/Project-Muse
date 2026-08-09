from environment.simulator import Simulator
from metrics.metrics_collector import MetricsCollector

'''
This class is for running ONE algorithm on ONE SEED
'''
# ---------------- Experiment Configuration ----------------

GRID_WIDTH = 100
GRID_HEIGHT = 100

NUM_DRONES = 5
OBSTACLE_PERCENTAGE = 0.80
COMMUNICATION_RADIUS = 10

MAX_STEPS = 10000
TARGET_COVERAGE = 90.0


def run_experiment(
        strategy,
        strategy_name,
        run_id,
        map_seed=None,
):
    """
    Runs one headless simulation experiment.

    No Pygame rendering is used.

    The simulation stops when:
    1. Target coverage is reached
    OR
    2. Maximum number of steps is reached
    """

    # ---------------- Create Simulator ----------------

    simulator = Simulator(
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        num_drones=NUM_DRONES,
        obstacle_percentage=OBSTACLE_PERCENTAGE,
        strategy=strategy,
        communication_radius=COMMUNICATION_RADIUS,
        map_seed=map_seed,
    )

    # ---------------- Create Metrics Collector ----------------

    metrics_collector = MetricsCollector(
        strategy_name=strategy_name,
        run_id=run_id,
        map_seed=map_seed,
    )

    print(
        f"Starting experiment: "
        f"strategy={strategy_name}, "
        f"run={run_id}, "
        f"seed={map_seed}"
    )

    # ---------------- Run Simulation ----------------

    while simulator.timestep < MAX_STEPS:

        simulator.step()

        coverage = simulator.get_coverage()

        total_distance = (
            simulator.get_total_distance()
        )

        sensing_redundancy = (
            simulator.get_sensing_redundancy()
        )

        visit_overlap = (
            simulator.get_visit_overlap_percentage()
        )

        # Get cluster and active drone metrics
        num_clusters = 0
        num_active_drones = 0
        
        if hasattr(simulator.strategy, 'get_metrics'):
            metrics = simulator.strategy.get_metrics()
            num_clusters = metrics.get('num_clusters', 0)
            num_active_drones = metrics.get('num_assigned_drones', 0)

        # Get exploration efficiency
        exploration_efficiency = simulator.get_exploration_efficiency()

        # Get mean pairwise distance
        mean_pairwise_distance = simulator.get_mean_pairwise_distance()

        # Get movement efficiency
        movement_efficiency = simulator.get_movement_efficiency()

        # Record this timestep
        metrics_collector.record(
            timestep=simulator.timestep,
            coverage=coverage,
            total_distance=total_distance,
            sensing_redundancy=sensing_redundancy,
            visit_overlap=visit_overlap,
            num_clusters=num_clusters,
            num_active_drones=num_active_drones,
            exploration_efficiency=exploration_efficiency,
            mean_pairwise_distance=mean_pairwise_distance,
            movement_efficiency=movement_efficiency
        )

        # Stop if target coverage reached
        if coverage >= TARGET_COVERAGE:
            break

    # ---------------- Save Results ----------------

    metrics_collector.save_csv()

    # ---------------- Print Summary ----------------

    reached_target = (
            coverage >= TARGET_COVERAGE
    )

    print(
        f"Finished experiment: "
        f"strategy={strategy_name}, "
        f"run={run_id}"
    )

    print(
        f"Steps: {simulator.timestep}"
    )

    print(
        f"Final coverage: {coverage:.2f}%"
    )

    print(
        f"Total distance: {total_distance}"
    )

    print(
        f"Reached target: {reached_target}"
    )

    print("-" * 50)

    # Return summary so another script can use it later
    return {
        "strategy": strategy_name,
        "run_id": run_id,
        "map_seed": map_seed,
        "total_steps": simulator.timestep,
        "final_coverage": coverage,
        "total_distance": total_distance,
        "reached_target": reached_target,
    }