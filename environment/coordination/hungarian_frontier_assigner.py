from scipy.optimize import linear_sum_assignment

from environment.coordination.cluster_frontier_utility_assigner import (
    ClusterFrontierUtilityAssigner,
)


class HungarianFrontierAssigner(
    ClusterFrontierUtilityAssigner
):

    def __init__(self, planner, utility):
        super().__init__(planner, utility)

        self.total_nodes_expanded = 0

    def _build_cost_matrix(
            self,
            drones,
            clusters,
            robot_map,
    ):
        cost_matrix = {}

        for drone in drones:
            cost_matrix[drone.id] = {}

            # Calculate euclidean distances to all clusters
            cluster_distances = []
            for cluster in clusters:
                euclidean_dist = ((drone.x - cluster.centroid[0]) ** 2 +
                                  (drone.y - cluster.centroid[1]) ** 2)

                cluster_distances.append((euclidean_dist, cluster))

            # Sort by distance and take nearest 3
            cluster_distances.sort(key=lambda x: x[0])
            # nearest_clusters = cluster_distances[:3]

            # Run BFS only on nearest 3 clusters
            for _, cluster in cluster_distances:
                path = self.planner.find_path(
                    start=(drone.x, drone.y),
                    goal=cluster.centroid,
                    robot_map=robot_map,
                )
                self.total_nodes_expanded += self.planner.nodes_expanded

                cost_matrix[drone.id][cluster.id] = {
                    "cluster": cluster,
                    "path": path,
                    "cost": len(path) if path else float("inf"),
                    "ig": cluster.information_gain,
                }

        return cost_matrix

    def _build_utility_matrix(
            self,
            drones,
            clusters,
            cost_matrix,
            robot_map
    ):
        """
        Build a drone x cluster utility matrix.

        Rows    -> drones
        Columns -> clusters
        """

        # --------------------------------------------------
        # First pass: collect all metrics for normalization
        # --------------------------------------------------

        all_ig = []
        all_cost = []
        all_redundancy = []
        all_cluster_size = []

        metrics_matrix = []

        for drone in drones:
            row_metrics = []
            for cluster in clusters:
                entry = cost_matrix[drone.id][cluster.id]
                ig = entry["ig"]
                cost = entry["cost"]

                predicted_redundancy = self._estimate_redundancy(
                    drone,
                    cluster.centroid,
                    robot_map
                )
                cluster_size = len(cluster.cells)

                all_ig.append(ig)
                all_cost.append(cost)
                all_redundancy.append(predicted_redundancy)
                all_cluster_size.append(cluster_size)

                row_metrics.append({
                    "ig": ig,
                    "cost": cost,
                    "redundancy": predicted_redundancy,
                    "cluster_size": cluster_size,
                })

            metrics_matrix.append(row_metrics)

        # --------------------------------------------------
        # Use raw metrics (normalization disabled)
        # --------------------------------------------------

        # Use raw values instead of normalized values
        raw_ig = all_ig
        raw_cost = all_cost
        raw_redundancy = all_redundancy
        raw_cluster_size = all_cluster_size

        # --------------------------------------------------
        # Second pass: calculate utilities with raw values
        # --------------------------------------------------

        utility_matrix = []
        idx = 0

        for drone_idx, drone in enumerate(drones):
            row = []
            for cluster_idx, cluster in enumerate(clusters):
                metrics = metrics_matrix[drone_idx][cluster_idx]

                utility = self.utility.calculate(
                    base_information_gain=raw_ig[idx],
                    path_cost=raw_cost[idx],
                    redundancy=raw_redundancy[idx],
                    cluster_size=raw_cluster_size[idx],
                )

                # print(
                #     f"Drone={drone.id}, "
                #     f"Cluster={cluster.id}, "
                #     f"IG={metrics['ig']:.2f} (norm={normalized_ig[idx]:.2f}), "
                #     f"Cost={metrics['cost']} (norm={normalized_cost[idx]:.2f}), "
                #     f"Redundancy={metrics['redundancy']:.2f} (norm={normalized_redundancy[idx]:.2f}), "
                #     f"ClusterSize={metrics['cluster_size']} (norm={normalized_cluster_size[idx]:.2f}), "
                #     f"Utility={utility:.2f}"
                # )

                row.append(utility)
                idx += 1

            utility_matrix.append(row)

        return utility_matrix

    def assign(
            self,
            drones,
            robot_map,
    ):

        # --------------------------------------------------
        # 1. Detect frontiers
        # --------------------------------------------------

        frontiers = self.frontier_detector.detect_frontiers(
            robot_map
        )

        if not frontiers:
            return {
                drone.id: {
                    "target": None,
                    "cluster": None,
                    "path": None,
                    "path_index": 0,
                    "cost": float("inf"),
                    "information_gain": None,
                }
                for drone in drones
            }

        # --------------------------------------------------
        # 2. Cluster frontiers
        # --------------------------------------------------

        clusters = self.frontier_clusterer.cluster_frontiers(
            frontiers,
            robot_map,
        )

        self.num_clusters = len(clusters)

        self.cluster_assigned_cells = {
            cluster.id: set()
            for cluster in clusters
        }

        # --------------------------------------------------
        # 3. Build drone x cluster path/IG matrix
        # --------------------------------------------------

        cost_matrix = self._build_cost_matrix(
            drones,
            clusters,
            robot_map,
        )

        # --------------------------------------------------
        # 4. Build utility matrix
        # --------------------------------------------------

        utility_matrix = self._build_utility_matrix(
            drones,
            clusters,
            cost_matrix,
            robot_map
        )

        # --------------------------------------------------
        # 5. Convert utility maximization into
        #    cost minimization
        # --------------------------------------------------

        assignment_cost_matrix = []

        for row in utility_matrix:

            assignment_cost_matrix.append([
                -utility
                if utility != float("-inf")
                else 1e9
                for utility in row
            ])

        # --------------------------------------------------
        # 6. Hungarian assignment
        # --------------------------------------------------

        row_indices, col_indices = linear_sum_assignment(
            assignment_cost_matrix
        )

        # --------------------------------------------------
        # 7. Convert result into assignments
        # --------------------------------------------------

        assignments = {}

        # Initially no drone is assigned
        for drone in drones:

            assignments[drone.id] = {
                "target": None,
                "cluster": None,
                "path": None,
                "path_index": 0,
                "cost": float("inf"),
                "information_gain": None,
            }

        assigned_count = 0

        # --------------------------------------------------
        # 8. Process Hungarian pairs
        # --------------------------------------------------

        for row, col in zip(
                row_indices,
                col_indices,
        ):

            drone = drones[row]
            cluster = clusters[col]

            entry = cost_matrix[
                drone.id
            ][
                cluster.id
            ]

            # No reachable path
            if entry["path"] is None:
                continue

            # --------------------------------------------------
            # Select actual frontier cell inside cluster
            # --------------------------------------------------

            target_cell, path, path_cost = (
                self._assign_frontier_cell_in_cluster(
                    drone,
                    cluster,
                    robot_map,
                )
            )

            if target_cell is None or path is None:
                continue

            assignments[drone.id] = {
                "target": target_cell,
                "cluster": cluster,
                "path": path,
                "path_index": 0,
                "cost": path_cost,
                "information_gain":
                    cluster.information_gain,
            }

            self.cluster_assigned_cells[
                cluster.id
            ].add(target_cell)

            assigned_count += 1

        self.num_assigned_drones = assigned_count

        return assignments