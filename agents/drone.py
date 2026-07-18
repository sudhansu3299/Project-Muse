
class Drone:
    def __init__(self, x, y, strategy, heading=0):
        self.x = x
        self.y = y
        self.heading = heading
        self.visited_cells = set()
        self.strategy = strategy

        # Useful later
        self.total_distance = 0
        self.id = None

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

        self.execute(action, true_map)

        self.update_map(true_map, robot_map)

    def sense(self):
        pass

    def update_map(self, true_map, robot_map):
        cell = true_map.get_cell(self.x, self.y)
        robot_map.set_cell(self.x, self.y, cell)

        self.visited_cells.add((self.x, self.y))

    def __repr__(self):
        return f"Drone(x={self.x}, y={self.y}, heading={self.heading})"