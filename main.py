import pygame

from environment.simulator import Simulator
from models.constants import Cell, Color, FontSize

from strategy.random_strategy import RandomStrategy
from strategy.frontier_strategy import FrontierStrategy

from environment.coordination.nearest_frontier_assigner import NearestFrontierAssigner
from environment.coordination.greedy_frontier_assigner import GreedyFrontierAssigner

from metrics.metrics_collector import MetricsCollector

# ---------------- Constants ----------------

CELL_SIZE = 8

GRID_HEIGHT = 100
GRID_WIDTH = 100

NUM_DRONES = 5
OBSTACLE_PERCENTAGE = 0.10
COMMUNICATION_RADIUS = 10

TOP_MARGIN = 100
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

MAX_STEPS = 10000
TARGET_COVERAGE = 90.0
MAP_SEED = 42

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

def draw_frontiers(
        screen,
        frontiers
):
    for x, y in frontiers:

        pygame.draw.circle(
            screen,
            (255, 0, 255),
            (
                PANEL_WIDTH
                + PADDING
                + x * CELL_SIZE
                + CELL_SIZE // 2,

                TOP_MARGIN
                + y * CELL_SIZE
                + CELL_SIZE // 2,
            ),
            2
        )

# ---------------- Simulator Initialization ----------------

# strategy = RandomStrategy()
#
# simulator = Simulator(
#     grid_width=GRID_WIDTH,
#     grid_height=GRID_HEIGHT,
#     num_drones=NUM_DRONES,
#     obstacle_percentage=OBSTACLE_PERCENTAGE,
#     strategy=strategy,
#     communication_radius=COMMUNICATION_RADIUS,
#     map_seed=MAP_SEED,
# )

strategy = FrontierStrategy(NearestFrontierAssigner())

simulator = Simulator(
    grid_width=GRID_WIDTH,
    grid_height=GRID_HEIGHT,
    num_drones=NUM_DRONES,
    obstacle_percentage=OBSTACLE_PERCENTAGE,
    strategy=strategy,
    communication_radius=COMMUNICATION_RADIUS,
    map_seed=MAP_SEED,
)


# ---------------- Pygame Initialization ----------------

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Multi-UAV Exploration Simulator"
)

title_font = pygame.font.SysFont(
    "Arial",
    FontSize.TITLE,
    bold=True
)

metrics_font = pygame.font.SysFont(
    "Arial",
    FontSize.METRICS,
    bold=True
)

button_font = pygame.font.SysFont(
    "Arial",
    FontSize.BUTTON,
    bold=True
)

clock = pygame.time.Clock()
elapsed_time = 0.0

running = True

simulation_running = True

pause_button = pygame.Rect(
    WIDTH // 2 - 50,
    10,
    100,
    28
)

metrics_collector = MetricsCollector(
    strategy_name="frontier",
    run_id=1,
    map_seed=MAP_SEED,
)


# ---------------- Main Loop ----------------

while running:

    dt = clock.tick(30) / 1000.0

    if simulation_running:
        elapsed_time += dt

    # ----- Handle Events -----

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if pause_button.collidepoint(event.pos):
                simulation_running = not simulation_running


    # ----- Advance Simulation -----

    if simulation_running:

        simulator.step()

        coverage = simulator.get_coverage()
        total_distance = (
            simulator.get_total_distance()
        )

        overlap_percentage = simulator.get_overlap_percentage()

        metrics_collector.record(
            timestep=simulator.timestep,
            coverage=coverage,
            total_distance=total_distance,
            overlap_percentage=overlap_percentage,
        )

        if (
                coverage >= TARGET_COVERAGE
                or
                simulator.timestep >= MAX_STEPS
        ):

            simulation_running = False
            metrics_collector.save_csv()

    # ----- Clear Screen -----

    screen.fill(
        (255, 255, 255)
    )

    # ----- Titles -----

    # Ground Truth title - centered over left grid
    ground_truth_text = title_font.render(
        "Ground Truth",
        True,
        (0, 0, 0)
    )

    ground_truth_rect = ground_truth_text.get_rect(
        center=(
            PANEL_WIDTH // 2,
            18
        )
    )

    screen.blit(
        ground_truth_text,
        ground_truth_rect
    )


    # Robot Map title - centered over right grid
    robot_map_text = title_font.render(
        "Robot Map",
        True,
        (0, 0, 0)
    )

    robot_map_rect = robot_map_text.get_rect(
        center=(
            PANEL_WIDTH
            + PADDING
            + PANEL_WIDTH // 2,
            18
        )
    )

    screen.blit(
        robot_map_text,
        robot_map_rect
    )


    # ----- Metrics -----

    coverage = simulator.get_coverage()

    metrics_text = metrics_font.render(
        f"Step: {simulator.timestep}  |  "
        f"Coverage: {coverage:.2f}%  |  "
        f"Overlap: {simulator.get_overlap_percentage():.2f}%  |  "
        f"Time: {elapsed_time:.1f}s",
        True,
        (40, 90, 160)
    )

    metrics_rect = metrics_text.get_rect(
        center=(
            WIDTH // 2,
            65
        )
    )

    screen.blit(
        metrics_text,
        metrics_rect
    )


    # ----- Pause / Resume Button -----

    pygame.draw.rect(
        screen,
        (200, 50, 50),
        pause_button
    )

    button_label = (
        "PAUSE"
        if simulation_running
        else "RESUME"
    )

    button_text = button_font.render(
        button_label,
        True,
        (255, 255, 255)
    )

    button_rect = button_text.get_rect(
        center=pause_button.center
    )

    screen.blit(
        button_text,
        button_rect
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

    frontiers = strategy.detect_frontiers(
        simulator.robot_map
    )

    draw_frontiers(
        screen,
        frontiers
    )


    # ----- Divider -----

    divider_x = (
            PANEL_WIDTH
            + PADDING // 2
    )

    pygame.draw.line(
        screen,
        (0, 0, 0),
        (divider_x, TOP_MARGIN),
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

metrics_collector.save_csv()

pygame.quit()