import pygame

from environment.simulator import Simulator
from models.constants import Cell, Color
from strategy.random_strategy import RandomStrategy


# ---------------- Constants ----------------

CELL_SIZE = 8

GRID_HEIGHT = 100
GRID_WIDTH = 100

NUM_DRONES = 5
OBSTACLE_PERCENTAGE = 0.10
COMMUNICATION_RADIUS = 10

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

DRONE_COLOR = Color.BLUE


# ---------------- Rendering ----------------

def draw_grid(screen, grid, offset_x=0, offset_y=0):

    for y in range(grid.height):

        for x in range(grid.width):

            color = GRID_COLORS[
                grid.get_cell(x, y)
            ]

            # Draw cell
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

            # Draw grid lines
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


def draw_drones(screen, drones):

    for drone in drones:

        pygame.draw.circle(
            screen,
            DRONE_COLOR,
            (
                PANEL_WIDTH
                + PADDING
                + drone.x * CELL_SIZE
                + CELL_SIZE // 2,

                TOP_MARGIN
                + drone.y * CELL_SIZE
                + CELL_SIZE // 2,
            ),
            CELL_SIZE // 2,
            )


# ---------------- Simulator Initialization ----------------

strategy = RandomStrategy()

simulator = Simulator(
    grid_width=GRID_WIDTH,
    grid_height=GRID_HEIGHT,
    num_drones=NUM_DRONES,
    obstacle_percentage=OBSTACLE_PERCENTAGE,
    strategy=strategy,
    communication_radius=COMMUNICATION_RADIUS,
)


# ---------------- Pygame Initialization ----------------

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Multi-UAV Exploration Simulator"
)

font = pygame.font.SysFont(
    "Arial",
    24,
    bold=True
)

clock = pygame.time.Clock()

running = True


# ---------------- Main Loop ----------------

while running:

    # ----- Handle Events -----

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False


    # ----- Advance Simulation -----

    simulator.step()


    # ----- Clear Screen -----

    screen.fill(
        (255, 255, 255)
    )


    # ----- Titles -----

    screen.blit(
        font.render(
            "Ground Truth",
            True,
            (0, 0, 0)
        ),
        (20, 8)
    )

    screen.blit(
        font.render(
            "Robot Map",
            True,
            (0, 0, 0)
        ),
        (
            PANEL_WIDTH
            + PADDING
            + 20,
            8
        ),
    )


    # ----- Draw Maps -----

    draw_grid(
        screen,
        simulator.true_map,
        0,
        TOP_MARGIN
    )

    draw_grid(
        screen,
        simulator.robot_map,
        PANEL_WIDTH + PADDING,
        TOP_MARGIN
    )


    # ----- Divider -----

    divider_x = (
            PANEL_WIDTH
            + PADDING // 2
    )

    pygame.draw.line(
        screen,
        (0, 0, 0),
        (divider_x, 0),
        (divider_x, HEIGHT),
        3,
    )


    # ----- Draw Drones -----

    draw_drones(
        screen,
        simulator.drones
    )


    # ----- Update Display -----

    pygame.display.flip()

    # Limit simulation speed
    clock.tick(30)

pygame.quit()