from models.constants import Cell
from environment.coordination.frontier_assigner import FrontierAssigner
from environment.utils.frontier_clusterer import FrontierClusterer


class ClusterFrontierAssigner(FrontierAssigner):

    def __init__(self):
        super().__init__()

        self.frontier_clusterer = FrontierClusterer()
        self.num_clusters = 0
        self.num_assigned_drones = 0
        self.cluster_assignment_counts = {}
        self.cluster_assigned_cells = {}

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
            nearest_clusters = cluster_distances[:3]
            
            # Run BFS only on nearest 3 clusters
            for _, cluster in nearest_clusters:
                path = self.bfs_planner.find_path(
                    start=(drone.x, drone.y),
                    goal=cluster.centroid,
                    robot_map=robot_map,
                )

                cost_matrix[drone.id][cluster.id] = {
                    "cluster": cluster,
                    "path": path,
                    "cost": len(path) if path else float("inf"),
                    "ig": cluster.information_gain,
                }
            
            # Set infinite cost for remaining clusters
            for _, cluster in cluster_distances[3:]:
                cost_matrix[drone.id][cluster.id] = {
                    "cluster": cluster,
                    "path": None,
                    "cost": float("inf"),
                    "ig": cluster.information_gain,
                }
        
        return cost_matrix

    def _assign_frontier_cell_in_cluster(
            self,
            drone,
            cluster,
            robot_map,
    ):

        if not cluster.cells:
            return None, None, float("inf")

        best_cell = None
        best_path = None
        best_score = float("-inf")
        best_cost = float("inf")

        for cell in cluster.cells:

            if cell in self.cluster_assigned_cells[cluster.id]:
                continue

            # self.cluster_assigned_cells[cluster.id].add(best_cell)

            path = self.bfs_planner.find_path(
                start=(drone.x, drone.y),
                goal=cell,
                robot_map=robot_map,
            )

            if path is None:
                continue

            cost = len(path)

            # Count unique unexplored cells around this frontier cell
            unexplored = 0

            for dx, dy in [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
            ]:

                nx = cell[0] + dx
                ny = cell[1] + dy

                if not robot_map.is_inside(nx, ny):
                    continue

                if robot_map.get_cell(nx, ny) == Cell.UNEXPLORED:
                    unexplored += 1

            # Information gain per unit travel
            score = unexplored / (cost + 1)

            if score > best_score:
                best_score = score
                best_cell = cell
                best_path = path
                best_cost = cost

        if best_cell is not None:
            self.cluster_assigned_cells[
                cluster.id
            ].add(best_cell)

        return best_cell, best_path, best_cost

    def get_metrics(self):
        """
        Returns the current assignment metrics.
        """
        return {
            "num_clusters": self.num_clusters,
            "num_assigned_drones": self.num_assigned_drones,
        }
    def _cluster_utility(
            self,
            drone,
            cluster,
            cost_matrix,
    ):

        entry = cost_matrix[drone.id][cluster.id]

        cost = entry["cost"]
        ig = entry["ig"]

        if cost == float("inf"):
            return float("-inf")

        load = self.cluster_assignment_counts[cluster.id]

        return (
                ig
                /
                (cost + 1)
                /
                (1 + load)
        )

    def assign(
            self,
            drones,
            robot_map,
    ):

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

        clusters = self.frontier_clusterer.cluster_frontiers(
            frontiers,
            robot_map,
        )

        self.cluster_assigned_cells = {
            cluster.id: set()
            for cluster in clusters
        }
        
        self.num_clusters = len(clusters)
        
        # Reset cluster assignment counts
        self.cluster_assignment_counts = {cluster.id: 0 for cluster in clusters}

        cost_matrix = self._build_cost_matrix(
            drones,
            clusters,
            robot_map,
        )

        assignments = {}
        assigned_count = 0

        # If we have enough clusters for each drone, use original logic
        if len(clusters) >= len(drones):
            available_clusters = clusters.copy()
            
            for drone in drones:
                if not available_clusters:
                    assignments[drone.id] = {
                        "target": None,
                        "cluster": None,
                        "path": None,
                        "path_index": 0,
                        "cost": float("inf"),
                        "information_gain": None,
                    }
                    continue

                valid_clusters = [
                    cluster
                    for cluster in available_clusters
                    if cost_matrix[drone.id][cluster.id]["path"] is not None
                ]

                if not valid_clusters:
                    assignments[drone.id] = {
                        "target": None,
                        "cluster": None,
                        "path": None,
                        "path_index": 0,
                        "cost": float("inf"),
                        "information_gain": None,
                    }
                    continue

                best_cluster = max(
                    valid_clusters,
                    key=lambda cluster:
                    self._cluster_utility(
                        drone,
                        cluster,
                        cost_matrix,
                    )
                )

                assignments[drone.id] = {
                    "target": best_cluster.centroid,
                    "cluster": best_cluster,
                    "path": cost_matrix[drone.id][best_cluster.id]["path"],
                    "path_index": 0,
                    "cost": cost_matrix[drone.id][best_cluster.id]["cost"],
                    "information_gain": best_cluster.information_gain,
                }

                available_clusters.remove(best_cluster)
                assigned_count += 1
        else:
            # Not enough clusters, assign multiple drones to same cluster
            for drone in drones:
                valid_clusters = [
                    cluster
                    for cluster in clusters
                    if cost_matrix[drone.id][cluster.id]["path"] is not None
                ]

                if not valid_clusters:
                    assignments[drone.id] = {
                        "target": None,
                        "cluster": None,
                        "path": None,
                        "path_index": 0,
                        "cost": float("inf"),
                        "information_gain": None,
                    }
                    continue

                # Select best cluster based on information gain,
                # path cost, and current cluster load
                best_cluster = max(
                    valid_clusters,
                    key=lambda cluster:
                    self._cluster_utility(
                        drone,
                        cluster,
                        cost_matrix,
                    )
                )

                # Assign a specific frontier cell within the cluster
                target_cell, path, cost = self._assign_frontier_cell_in_cluster(
                    drone,
                    best_cluster,
                    robot_map,
                )

                if target_cell is None or path is None:
                    assignments[drone.id] = {
                        "target": None,
                        "cluster": None,
                        "path": None,
                        "path_index": 0,
                        "cost": float("inf"),
                        "information_gain": None,
                    }
                    continue

                assignments[drone.id] = {
                    "target": target_cell,
                    "cluster": best_cluster,
                    "path": path,
                    "path_index": 0,
                    "cost": cost,
                    "information_gain": best_cluster.information_gain,
                }

                # Increment assignment count for this cluster
                self.cluster_assignment_counts[best_cluster.id] += 1
                assigned_count += 1

        self.num_assigned_drones = assigned_count
        return assignments