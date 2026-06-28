import random

class Drone:
    def __init__(self, x, y, heading=0):
        self.x = x
        self.y = y
        self.heading = heading
        self.visited_cells = set()

    def move(self, grid):
        directions = [
            (0, 1),
            (0, -1),
            (-1, 0),
            (1, 0)
        ]

        random.shuffle(directions)

        for dx, dy in directions:
            newX = self.x + dx
            newY = self.y + dy

            if grid.is_valid(newX, newY):
                self.x = newX
                self.y = newY
                return

    def sense(self):
        pass

    def update_map(self, true_map, robot_map):
        cell = true_map.get_cell(self.x, self.y)
        robot_map.set_cell(self.x, self.y, cell)
        self.visited_cells.add((self.x, self.y))

    def __repr__(self):
        return f"Drone(x={self.x}, y={self.y}, heading={self.heading})"