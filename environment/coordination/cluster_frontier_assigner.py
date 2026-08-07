import math
from environment.coordination.frontier_assigner import FrontierAssigner
from environment.utils.frontier_clusterer import FrontierClusterer


class ClusterFrontierAssigner(FrontierAssigner):

    def __init__(self):
        super().__init__()

        self.frontier_clusterer = FrontierClusterer()

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
                drone.id: None
                for drone in drones
            }

        clusters = self.frontier_clusterer.cluster_frontiers(
            frontiers,
            robot_map,
        )

        cost_matrix = self._build_cost_matrix(
            drones,
            clusters,
            robot_map,
        )

        assignments = {}

        available_clusters = clusters.copy()

        for drone in drones:

            if not available_clusters:
                assignments[drone.id] = None
                continue

            valid_clusters = [
                cluster
                for cluster in available_clusters
                if cost_matrix[drone.id][cluster.id]["path"] is not None
            ]

            if not valid_clusters:
                assignments[drone.id] = None
                continue

            best_cluster = min(
                available_clusters,
                key=lambda cluster:
                cost_matrix[drone.id][cluster.id]["cost"]
            )

            assignments[drone.id] = {
                "cluster": best_cluster,
                "path": cost_matrix[drone.id][best_cluster.id]["path"],
                "path_index": 0,
            }

            available_clusters.remove(
                best_cluster
            )

        return assignments