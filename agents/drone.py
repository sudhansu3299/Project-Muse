from models.constants import Cell

class Drone:
    def __init__(self, x, y, strategy, sensor_radius=3, heading=0):
        self.x = x
        self.y = y
        self.heading = heading
        self.visited_cells = set()
        self.strategy = strategy

        self.sensor_radius = sensor_radius

        # Useful later
        self.total_distance = 0
        self.id = None

        self.total_sensed_cells = 0
        self.redundant_sensed_cells = 0

    def execute(self, action, true_map):

        dx, dy = action.delta()

        new_x = self.x + dx
        new_y = self.y + dy

        if true_map.is_valid(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.total_distance += 1

    def step(self, true_map, robot_map, nearby_agents):
        """
        One simulation step.
        """

        action = self.strategy.choose_action(
            self,
            robot_map,
            true_map,
            nearby_agents,
        )

        # print("Action called with:", action)

        self.execute(action, true_map)

        # Sense again after movement
        self.sense(
            true_map,
            robot_map
        )

        self.update_map(true_map, robot_map)

#Sense basically updates the sensor of the drone with LIDAR/sensor that can
#sense at once rather than one cell at a time
    def sense(self, true_map, robot_map):

        for dy in range(-self.sensor_radius, self.sensor_radius + 1):
            for dx in range(-self.sensor_radius, self.sensor_radius + 1):

                new_x = self.x + dx
                new_y = self.y + dy

                # Circular sensor radius
                distance_squared = dx * dx + dy * dy

                if distance_squared > self.sensor_radius ** 2:
                    continue

                # Make sure cell is inside map
                if not true_map.is_inside(new_x, new_y):
                    continue

                self.total_sensed_cells += 1

                # Was this cell already discovered?
                if (
                        robot_map.get_cell(new_x, new_y)
                        != Cell.UNEXPLORED
                ):
                    self.redundant_sensed_cells += 1

                cell = true_map.get_cell(
                    new_x,
                    new_y
                )

                robot_map.set_cell(
                    new_x,
                    new_y,
                    cell
                )

    def update_map(self, true_map, robot_map):
        cell = true_map.get_cell(self.x, self.y)
        robot_map.set_cell(self.x, self.y, cell)

        self.visited_cells.add((self.x, self.y))

    def __repr__(self):
        return f"Drone(x={self.x}, y={self.y}, heading={self.heading})"