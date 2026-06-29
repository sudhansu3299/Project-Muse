import pygame
from environment.grid import OccupancyGrid
from models.constants import Cell, Color
from agents.drone import Drone


# ---------------- Constants ----------------

CELL_SIZE = 8

GRID_HEIGHT = 100
GRID_WIDTH = 100

TOP_MARGIN = 40
PADDING = 20

PANEL_WIDTH = GRID_WIDTH * CELL_SIZE

WIDTH = PANEL_WIDTH * 2 + PADDING
HEIGHT = GRID_HEIGHT * CELL_SIZE + TOP_MARGIN

GRID_COLORS = {
    Cell.UNEXPLORED: Color.GRAY,
    Cell.FREE: Color.WHITE,
    Cell.OBSTACLE: Color.BLACK,
}

#Draw the grid for true and robot map
def draw_grid(screen, grid, offset_x=0, offset_y=0):
    for y in range(grid.height):
        for x in range(grid.width):
            color = GRID_COLORS[grid.get_cell(x, y)]

            pygame.draw.rect(
                screen,
                color,
                (
                    offset_x + x * CELL_SIZE,
                    offset_y + y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ),
            )

            #grid lines separation with black lines
            pygame.draw.rect(
                screen,
                (180, 180, 180),
                (
                    offset_x + x * CELL_SIZE,
                    offset_y + y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ),
                1,
            )



# ---------------- Initialization ----------------

true_map = OccupancyGrid(GRID_HEIGHT, GRID_WIDTH) #true map of the world
robot_map = OccupancyGrid(GRID_HEIGHT, GRID_WIDTH) #what the robot perceives as of now

true_map.randomize_obstacles(0.10) #randomize the obstacle generation
robot_map.reset()

drones = [ #lets assume to start at 0,0 for all the drones
    Drone(0, 0),
    Drone(0, 0),
    Drone(0, 0),
    Drone(0, 0),
    Drone(0, 0),
]

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Occupancy Grid")

font = pygame.font.SysFont("Arial", 24, bold=True)

running = True

# ---------------- Main Loop ----------------
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    # ----- Titles -----
    screen.blit(font.render("Ground Truth", True, (0, 0, 0)), (20, 8))
    screen.blit(
        font.render("Robot Map", True, (0, 0, 0)),
        (PANEL_WIDTH + PADDING + 20, 8),
    )

    # ----- Draw the two grids -----
    draw_grid(screen, true_map, 0, TOP_MARGIN)
    draw_grid(screen, robot_map, PANEL_WIDTH + PADDING, TOP_MARGIN)

    divider_x = PANEL_WIDTH + PADDING // 2 #divider between the two grids

    pygame.draw.line(
        screen,
        (0, 0, 0),
        (divider_x, 0),
        (divider_x, HEIGHT),
        3,
    )

    #Drone visualization
    DRONE_COLOR = Color.BLUE

    #move the drones
    for drone in drones:
        drone.update_map(true_map, robot_map)
        drone.move(true_map)

    #draw the drones
    for drone in drones:
        pygame.draw.circle(
            screen,
            DRONE_COLOR,
            (
                PANEL_WIDTH + PADDING + drone.x * CELL_SIZE + CELL_SIZE // 2,
                TOP_MARGIN + drone.y * CELL_SIZE + CELL_SIZE // 2,
            ),
            CELL_SIZE // 2,
            )

    pygame.display.flip()

pygame.quit()