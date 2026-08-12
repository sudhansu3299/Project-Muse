import csv
import os


class MetricsCollector:

    def __init__(
            self,
            strategy_name,
            run_id,
            map_seed=None,
    ):
        self.strategy_name = strategy_name
        self.run_id = run_id
        self.map_seed = map_seed
        self.saved = False

        self.records = []

    def record(
            self,
            timestep,
            coverage,
            total_distance,
            sensing_redundancy,
            visit_overlap,
            num_clusters=0,
            num_active_drones=0,
            exploration_efficiency=0.0,
            mean_pairwise_distance=0.0,
            movement_efficiency=0.0
    ):

        """
        Record metrics for one simulation timestep.
        """

        self.records.append({
            "strategy": self.strategy_name,
            "run_id": self.run_id,
            "map_seed": self.map_seed,
            "timestep": timestep,
            "coverage": coverage,
            "total_distance": total_distance,
            "sensing_redundancy": sensing_redundancy,
            "visit_overlap": visit_overlap,
            "num_clusters": num_clusters,
            "num_active_drones": num_active_drones,
            "exploration_efficiency": exploration_efficiency,
            "mean_pairwise_distance": mean_pairwise_distance,
            "movement_efficiency": movement_efficiency
        })

    def save_csv(self):

        if self.saved:
            print("Already saved. Skipping.")
            return

        if not self.records:
            print("No records available. Skipping.")
            return

        # Each run gets its own directory
        output_dir = os.path.join(
            "results",
            "raw",
            f"run_{self.run_id:03d}"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # Each strategy gets its own CSV inside the run
        filename = os.path.join(
            output_dir,
            f"{self.strategy_name}.csv"
        )

        print(
            f"Saving CSV to: {os.path.abspath(filename)}"
        )

        with open(
                filename,
                "w",
                newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.records[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                self.records
            )

        self.saved = True

        print(f"Metrics saved to {filename}")