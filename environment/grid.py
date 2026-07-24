import random
from models.constants import Cell

class OccupancyGrid:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def reset(self):
        self.grid = [[-1 for _ in range(self.width)] for _ in range(self.height)]

    def get_cell(self, x, y):
        return self.grid[y][x]

    def set_cell(self, x, y, value):
        self.grid[y][x] = value

    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_inside(self, x, y):
        return (
                0 <= x < self.width
                and
                0 <= y < self.height
        )

    def is_obstacle(self, x, y):
        return self.grid[y][x] == 1

    def randomize_obstacles(self, obstacle_percentage, seed=None):
        rng = random.Random(seed) #To keep the map intact for all strategies

        for y in range(self.height):
            for x in range(self.width):
                if rng.random() < obstacle_percentage:
                    self.set_cell(x, y, Cell.OBSTACLE)
                else:
                    self.set_cell(x, y, Cell.FREE)


#[y][x] because the grid is stored as x and y cartesian coordinates.

"""    
Column (x)
    0   1   2
+---+---+---+
Row 0 | 1 | 2 | 3 |
+---+---+---+
Row 1 | 4 | 5 | 6 |
+---+---+---+
Row 2 | 7 | 8 | 9 |
+---+---+---+
"""