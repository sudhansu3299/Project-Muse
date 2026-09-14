import csv
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from environment.simulator import Simulator
from strategy.frontier_strategy import FrontierStrategy
from environment.coordination.cluster_frontier_utility_assigner import ClusterFrontierUtilityAssigner
from environment.coordination.hungarian_frontier_assigner import HungarianFrontierAssigner
from environment.planner.bfs_planner import BFSPlanner
from environment.utils.frontier_utility import FrontierUtility


# ---------------- Experiment Configuration ----------------
GRID_WIDTH = 100
GRID_HEIGHT = 100
NUM_DRONES = 5
OBSTACLE_PERCENTAGE = 0.20
COMMUNICATION_RADIUS = 10
MAX_STEPS = 10000
TARGET_COVERAGE = 90.0

# Seed range
START_SEED = 105
END_SEED = 115


# ---------------- Utility Ablation Configurations ----------------
UTILITY_CONFIGS = {
    "ig_only": FrontierUtility(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0),
    "ig_cost": FrontierUtility(alpha=1.0, beta=0.5, gamma=0.0, delta=0.0),
    "ig_cost_redundancy": FrontierUtility(alpha=1.0, beta=0.5, gamma=0.5, delta=0.0),
    "ig_cost_redundancy_cluster": FrontierUtility(alpha=1.0, beta=0.5, gamma=0.5, delta=0.1),
}


class UtilityAblation:
    """
    Runs utility function ablation study on multiple seeds.
    Tests incremental addition of utility components.
    """

    def __init__(self):
        self.raw_results = []
        self.aggregated_results = {}

    def get_strategies(self):
        """
        Returns dictionary of strategy name to strategy factory.
        Tests each utility configuration with both cluster and Hungarian assigners.
        """
        strategies = {}
        
        for config_name, utility in UTILITY_CONFIGS.items():
            # Test with cluster utility assigner
            strategies[f"cluster_{config_name}"] = lambda u=utility: FrontierStrategy(
                ClusterFrontierUtilityAssigner(BFSPlanner(), u)
            )
            # Test with Hungarian utility assigner
            strategies[f"hungarian_{config_name}"] = lambda u=utility: FrontierStrategy(
                HungarianFrontierAssigner(planner=BFSPlanner(), utility=u)
            )
        
        return strategies

    def run_single_experiment(
        self,
        strategy,
        strategy_name,
        map_seed,
    ):
        """
        Runs one experiment and extracts metrics at 90% coverage.
        Returns dictionary with metrics or None if 90% coverage not reached.
        """
        # Create simulator
        simulator = Simulator(
            grid_width=GRID_WIDTH,
            grid_height=GRID_HEIGHT,
            num_drones=NUM_DRONES,
            obstacle_percentage=OBSTACLE_PERCENTAGE,
            strategy=strategy,
            communication_radius=COMMUNICATION_RADIUS,
            map_seed=map_seed,
        )

        # Track metrics at 90% coverage
        metrics_at_90 = None

        # Run simulation
        while simulator.timestep < MAX_STEPS:
            simulator.step()

            coverage = simulator.get_coverage()

            # Check if we just reached or passed 90% coverage
            if coverage >= TARGET_COVERAGE and metrics_at_90 is None:
                total_distance = simulator.get_total_distance()
                sensing_redundancy = simulator.get_sensing_redundancy()
                visit_overlap = simulator.get_visit_overlap_percentage()
                movement_efficiency = simulator.get_movement_efficiency()

                # Get nodes expanded from strategy if available
                nodes_expanded = 0
                if hasattr(simulator.strategy, 'get_metrics'):
                    metrics = simulator.strategy.get_metrics()
                    nodes_expanded = metrics.get('nodes_expanded', 0)

                metrics_at_90 = {
                    "time_to_90": simulator.timestep,
                    "distance": total_distance,
                    "sensing_redundancy": sensing_redundancy,
                    "visit_overlap": visit_overlap,
                    "movement_efficiency": movement_efficiency,
                    "nodes_expanded": nodes_expanded,
                }

            # Stop if we're well past 90% coverage to save time
            if coverage >= TARGET_COVERAGE + 5:
                break

        return metrics_at_90

    def run_benchmark(self):
        """
        Runs all strategies on all seeds and collects results.
        """
        strategies = self.get_strategies()

        print(f"Running utility ablation on seeds {START_SEED}-{END_SEED}")
        print(f"Utility configs: {list(UTILITY_CONFIGS.keys())}")
        print(f"Total strategies: {len(strategies)}")
        print("-" * 60)

        for seed in range(START_SEED, END_SEED + 1):
            print(f"\n===== SEED {seed} =====")

            for strategy_name, strategy_factory in strategies.items():
                print(f"  Running {strategy_name}...")

                strategy = strategy_factory()
                metrics = self.run_single_experiment(
                    strategy=strategy,
                    strategy_name=strategy_name,
                    map_seed=seed,
                )

                if metrics is not None:
                    # Store raw result
                    self.raw_results.append({
                        "strategy": strategy_name,
                        "seed": seed,
                        **metrics,
                    })
                    print(f"    ✓ Reached 90% at step {metrics['time_to_90']}")
                else:
                    print(f"    ✗ Did not reach 90% coverage")
                    # Still record with None values for consistency
                    self.raw_results.append({
                        "strategy": strategy_name,
                        "seed": seed,
                        "time_to_90": None,
                        "distance": None,
                        "sensing_redundancy": None,
                        "visit_overlap": None,
                        "movement_efficiency": None,
                        "nodes_expanded": None,
                    })

    def compute_aggregated_stats(self):
        """
        Computes mean ± std for each strategy across all seeds.
        """
        # Group results by strategy
        strategy_results = {}
        for result in self.raw_results:
            strategy_name = result["strategy"]
            if strategy_name not in strategy_results:
                strategy_results[strategy_name] = []

            # Only include successful runs
            if result["time_to_90"] is not None:
                strategy_results[strategy_name].append(result)

        # Compute stats for each strategy
        for strategy_name, results in strategy_results.items():
            if not results:
                continue

            # Extract metric lists
            times = [r["time_to_90"] for r in results]
            distances = [r["distance"] for r in results]
            redundancies = [r["sensing_redundancy"] for r in results]
            overlaps = [r["visit_overlap"] for r in results]
            efficiencies = [r["movement_efficiency"] for r in results]
            nodes = [r["nodes_expanded"] for r in results]

            # Compute mean and std
            import numpy as np

            self.aggregated_results[strategy_name] = {
                "time_to_90_mean": np.mean(times),
                "time_to_90_std": np.std(times),
                "distance_mean": np.mean(distances),
                "distance_std": np.std(distances),
                "sensing_redundancy_mean": np.mean(redundancies),
                "sensing_redundancy_std": np.std(redundancies),
                "visit_overlap_mean": np.mean(overlaps),
                "visit_overlap_std": np.std(overlaps),
                "movement_efficiency_mean": np.mean(efficiencies),
                "movement_efficiency_std": np.std(efficiencies),
                "nodes_expanded_mean": np.mean(nodes),
                "nodes_expanded_std": np.std(nodes),
                "num_successful_runs": len(results),
            }

    def save_raw_csv(self):
        """
        Saves raw results to CSV.
        """
        output_dir = os.path.join("results", "utility_ablation")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, "raw_results.csv")

        fieldnames = [
            "strategy",
            "seed",
            "time_to_90",
            "distance",
            "sensing_redundancy",
            "visit_overlap",
            "movement_efficiency",
            "nodes_expanded",
        ]

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.raw_results)

        print(f"\n✓ Raw results saved to {filename}")

    def save_aggregated_csv(self):
        """
        Saves aggregated results (mean ± std) to CSV.
        """
        output_dir = os.path.join("results", "utility_ablation")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, "aggregated_results.csv")

        fieldnames = [
            "strategy",
            "time_to_90",
            "distance",
            "sensing_redundancy",
            "visit_overlap",
            "movement_efficiency",
            "nodes_expanded",
            "num_successful_runs",
        ]

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for strategy_name, stats in self.aggregated_results.items():
                # Format as "mean ± std"
                row = {
                    "strategy": strategy_name,
                    "time_to_90": f"{stats['time_to_90_mean']:.0f} ± {stats['time_to_90_std']:.0f}",
                    "distance": f"{stats['distance_mean']:.0f} ± {stats['distance_std']:.0f}",
                    "sensing_redundancy": f"{stats['sensing_redundancy_mean']:.1f}% ± {stats['sensing_redundancy_std']:.1f}%",
                    "visit_overlap": f"{stats['visit_overlap_mean']:.1f}% ± {stats['visit_overlap_std']:.1f}%",
                    "movement_efficiency": f"{stats['movement_efficiency_mean']:.3f} ± {stats['movement_efficiency_std']:.3f}",
                    "nodes_expanded": f"{stats['nodes_expanded_mean']:.0f} ± {stats['nodes_expanded_std']:.0f}",
                    "num_successful_runs": stats['num_successful_runs'],
                }
                writer.writerow(row)

        print(f"✓ Aggregated results saved to {filename}")

    def print_aggregated_results(self):
        """
        Prints aggregated results in a formatted table.
        """
        print("\n" + "=" * 120)
        print("UTILITY ABLATION RESULTS (Mean ± Std)")
        print("=" * 120)

        # Print header
        header = f"{'Strategy':<30} | {'Time to 90%':<12} | {'Distance':<12} | {'Sensing Redundancy':<18} | {'Visit Overlap':<14} | {'Movement Efficiency':<18}"
        print(header)
        print("-" * 120)

        # Group by assigner type for better comparison
        cluster_strategies = {k: v for k, v in self.aggregated_results.items() if k.startswith("cluster_")}
        hungarian_strategies = {k: v for k, v in self.aggregated_results.items() if k.startswith("hungarian_")}

        print("\n--- Cluster Frontier Assigner ---")
        for strategy_name in ["cluster_ig_only", "cluster_ig_cost", "cluster_ig_cost_redundancy", "cluster_ig_cost_redundancy_cluster"]:
            if strategy_name in cluster_strategies:
                stats = cluster_strategies[strategy_name]
                row = (
                    f"{strategy_name:<30} | "
                    f"{stats['time_to_90_mean']:.0f} ± {stats['time_to_90_std']:.0f:<8} | "
                    f"{stats['distance_mean']:.0f} ± {stats['distance_std']:.0f:<8} | "
                    f"{stats['sensing_redundancy_mean']:.1f}% ± {stats['sensing_redundancy_std']:.1f}%<{10} | "
                    f"{stats['visit_overlap_mean']:.1f}% ± {stats['visit_overlap_std']:.1f}%<{6} | "
                    f"{stats['movement_efficiency_mean']:.3f} ± {stats['movement_efficiency_std']:.3f}<{10}"
                )
                print(row)

        print("\n--- Hungarian Frontier Assigner ---")
        for strategy_name in ["hungarian_ig_only", "hungarian_ig_cost", "hungarian_ig_cost_redundancy", "hungarian_ig_cost_redundancy_cluster"]:
            if strategy_name in hungarian_strategies:
                stats = hungarian_strategies[strategy_name]
                row = (
                    f"{strategy_name:<30} | "
                    f"{stats['time_to_90_mean']:.0f} ± {stats['time_to_90_std']:.0f:<8} | "
                    f"{stats['distance_mean']:.0f} ± {stats['distance_std']:.0f:<8} | "
                    f"{stats['sensing_redundancy_mean']:.1f}% ± {stats['sensing_redundancy_std']:.1f}%<{10} | "
                    f"{stats['visit_overlap_mean']:.1f}% ± {stats['visit_overlap_std']:.1f}%<{6} | "
                    f"{stats['movement_efficiency_mean']:.3f} ± {stats['movement_efficiency_std']:.3f}<{10}"
                )
                print(row)

        print("=" * 120)

    def run(self):
        """
        Runs the complete benchmark pipeline.
        """
        self.run_benchmark()
        self.compute_aggregated_stats()
        self.save_raw_csv()
        self.save_aggregated_csv()
        self.print_aggregated_results()


if __name__ == "__main__":
    benchmark = UtilityAblation()
    benchmark.run()
