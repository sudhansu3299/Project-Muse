from collections import deque

from sympy import centroid

from models.constants import Cell
from models.action import Action


class FrontierCluster:

    def __init__(self, cluster_id, cells):

        self.id = cluster_id
        self.cells = cells

        self.centroid = self._calculate_centroid()
        self.size = len(cells)

        self.information_gain = 0.0

    def _calculate_centroid(self):

        if not self.cells:
            return (0, 0)

        avg_x = sum(x for x, _ in self.cells) / len(self.cells)
        avg_y = sum(y for _, y in self.cells) / len(self.cells)

        centroid_cell =  (
            int(round(avg_x)),
            int(round(avg_y))
        )

        representative_cell = min(
            self.cells,
            key=lambda cell: (
                (cell[0] - centroid_cell[0]) ** 2 + #there is a possibility that the centroid is not a free cell
                (cell[1] - centroid_cell[1]) ** 2
            )
        )

        return representative_cell


class FrontierClusterer:

    def __init__(self):

        self.next_cluster_id = 0

    def cluster_frontiers(
            self,
            frontiers,
            robot_map,
    ):
        """
        Groups connected frontier cells into clusters.
        """
        self.next_cluster_id = 0

        if not frontiers:
            return []

        clusters = self._connected_components(
            frontiers
        )

        # print(f"Frontiers: {len(frontiers)}")
        # print(f"Clusters: {len(clusters)}")
        #
        # for cluster in clusters:
        #     print(
        #         f"id={cluster.id}, "
        #         f"size={cluster.size}, "
        #         f"centroid={cluster.centroid}"
        #     )

        for cluster in clusters:

            cluster.information_gain = (
                self._calculate_information_gain(
                    cluster,
                    robot_map,
                )
            )

        return sorted(
            clusters,
            key=lambda c: c.information_gain,
            reverse=True,
        )

    def _connected_components(
            self,
            frontiers,
    ):

        frontier_set = set(frontiers)

        visited = set()

        clusters = []

        for frontier in frontier_set:

            if frontier in visited:
                continue

            cells = []

            queue = deque([frontier])

            visited.add(frontier)

            while queue:

                current = queue.popleft()

                cells.append(current)

                for neighbour in self._get_neighbors(current):

                    if (
                            neighbour in frontier_set
                            and neighbour not in visited
                    ):

                        visited.add(neighbour)

                        queue.append(neighbour)

            clusters.append(
                FrontierCluster(
                    self.next_cluster_id,
                    cells,
                )
            )

            self.next_cluster_id += 1

        return clusters

    def _get_neighbors(
            self,
            cell,
    ):

        x, y = cell

        neighbours = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                neighbours.append(
                    (
                        x + dx,
                        y + dy,
                    )
                )

        return neighbours

    def _calculate_information_gain(
            self,
            cluster,
            robot_map,
    ):

        unexplored_cells = set()
        total_cells = set()

        for x, y in cluster.cells:

            for action in [
                Action.UP,
                Action.DOWN,
                Action.LEFT,
                Action.RIGHT,
            ]:

                dx, dy = action.delta()

                nx = x + dx
                ny = y + dy

                if not robot_map.is_inside(nx, ny):
                    continue

                cell = (nx, ny)

                total_cells.add(cell)

                if (
                        robot_map.get_cell(nx, ny)
                        == Cell.UNEXPLORED
                ):
                    unexplored_cells.add(cell)

        base_gain = len(unexplored_cells) / max(len(total_cells), 1)
        #This is because we shouldn't double count the unexplored cells!
        #Asks how many unique neigbouring cells around this cluster are unexplored?

        # size_bonus = 0.1 * cluster.size

        return round(
            base_gain,
            4,
            )
    #IG= unexplored/total + 0.1 * S (0.1 being the hyperparameter)
    '''
    Now, IG=U/T+λS and then multiplying with α, would still give αλS
    Thus separated the λS from IG to make a separate parameter to make this δS
    '''

'''
Detect Frontiers
        ↓
Cluster nearby frontiers
        ↓
Compute Information Gain for each cluster
        ↓
Sort clusters by Information Gain
        ↓
Take top N clusters
        ↓
Assign Drone i → Cluster i
'''