import numpy as np
import pygame

from rl.exploration_env import ExplorationEnv

from models.constants import Cell, Color, FontSize
from environment.utils.frontier_detector import FrontierDetector
from environment.utils.frontier_clusterer import FrontierClusterer


# ============================================================
# Configuration
# ============================================================

CELL_SIZE = 8

GRID_HEIGHT = 25
GRID_WIDTH = 25

NUM_DRONES = 5
OBSTACLE_PERCENTAGE = 0.20
COMMUNICATION_RADIUS = 10

TOP_MARGIN = 100
PADDING = 20

PANEL_WIDTH = GRID_WIDTH * CELL_SIZE

WIDTH = PANEL_WIDTH * 2 + PADDING
HEIGHT = GRID_HEIGHT * CELL_SIZE + TOP_MARGIN

MAX_STEPS = 1000
NUM_EPISODES = 3
MAP_SEED = 42


# ============================================================
# Rendering
# ============================================================

CLUSTER_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 165, 0),
    (128, 0, 255),
]


GRID_COLORS = {
    Cell.UNEXPLORED: Color.GRAY,
    Cell.FREE: Color.WHITE,
    Cell.OBSTACLE: Color.BLACK,
}

DRONE_COLOR = Color.BLUE


def draw_grid(screen, grid, offset_x=0, offset_y=0):

    for y in range(grid.height):
        for x in range(grid.width):

            color = GRID_COLORS[
                grid.get_cell(x, y)
            ]

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


def draw_clusters(screen, clusters):

    for cluster in clusters:

        color = CLUSTER_COLORS[
            cluster.id % len(CLUSTER_COLORS)
            ]

        for x, y in cluster.cells:

            pygame.draw.circle(
                screen,
                color,
                (
                    PANEL_WIDTH
                    + PADDING
                    + x * CELL_SIZE
                    + CELL_SIZE // 2,

                    TOP_MARGIN
                    + y * CELL_SIZE
                    + CELL_SIZE // 2,
                ),
                2,
            )


# ============================================================
# Environment
# ============================================================

env = ExplorationEnv(
    grid_width=GRID_WIDTH,
    grid_height=GRID_HEIGHT,
    num_drones=NUM_DRONES,
    obstacle_percentage=OBSTACLE_PERCENTAGE,
    communication_radius=COMMUNICATION_RADIUS,
    max_steps=MAX_STEPS,
)


frontier_detector = FrontierDetector()
frontier_clusterer = FrontierClusterer()


# ============================================================
# Pygame
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "PPO Exploration Visualization"
)

title_font = pygame.font.SysFont(
    "Arial",
    FontSize.TITLE,
    bold=True,
)

metrics_font = pygame.font.SysFont(
    "Arial",
    FontSize.METRICS,
    bold=True,
)

clock = pygame.time.Clock()


# ============================================================
# Episode loop
# ============================================================

running = True

for episode in range(NUM_EPISODES):

    if not running:
        break

    print(
        f"\n========== EPISODE {episode} =========="
    )

    # --------------------------------------------------------
    # IMPORTANT: reset environment
    # --------------------------------------------------------

    state, info = env.reset(
        seed=MAP_SEED
    )

    episode_done = False

    total_reward = 0.0

    while running and not episode_done:

        # ----------------------------------------------------
        # Handle pygame events
        # ----------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        # ----------------------------------------------------
        # Fixed weights for now
        #
        # We are NOT testing PPO yet.
        # We are testing the environment + reset behavior.
        # ----------------------------------------------------

        action = np.array(
            [1.0, 0.5, 0.5, 0.1],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # One environment step
        # ----------------------------------------------------

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        episode_done = (
                terminated or truncated
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        coverage = info["coverage"]
        redundancy = info["redundancy"]
        assigned_drones = info["assigned_drones"]

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        screen.fill(
            (255, 255, 255)
        )

        # Ground truth title

        ground_truth_text = title_font.render(
            "Ground Truth",
            True,
            (0, 0, 0),
        )

        ground_truth_rect = (
            ground_truth_text.get_rect(
                center=(
                    PANEL_WIDTH // 2,
                    18,
                )
            )
        )

        screen.blit(
            ground_truth_text,
            ground_truth_rect,
        )

        # Robot map title

        robot_map_text = title_font.render(
            "Robot Map",
            True,
            (0, 0, 0),
        )

        robot_map_rect = (
            robot_map_text.get_rect(
                center=(
                    PANEL_WIDTH
                    + PADDING
                    + PANEL_WIDTH // 2,
                    18,
                )
            )
        )

        screen.blit(
            robot_map_text,
            robot_map_rect,
        )

        # ----------------------------------------------------
        # Metrics text
        # ----------------------------------------------------

        metrics_text = metrics_font.render(
            f"Episode: {episode}  |  "
            f"Step: {env.simulator.timestep}  |  "
            f"Coverage: {coverage:.2f}%  |  "
            f"Redundancy: {redundancy:.2f}%  |  "
            f"Assigned: {assigned_drones}",
            True,
            (40, 90, 160),
        )

        metrics_rect = (
            metrics_text.get_rect(
                center=(
                    WIDTH // 2,
                    65,
                )
            )
        )

        screen.blit(
            metrics_text,
            metrics_rect,
        )

        # ----------------------------------------------------
        # Draw maps
        # ----------------------------------------------------

        draw_grid(
            screen,
            env.simulator.true_map,
            0,
            TOP_MARGIN,
        )

        draw_grid(
            screen,
            env.simulator.robot_map,
            PANEL_WIDTH + PADDING,
            TOP_MARGIN,
            )

        # ----------------------------------------------------
        # Draw frontier clusters
        # ----------------------------------------------------

        frontiers = (
            frontier_detector.detect_frontiers(
                env.simulator.robot_map
            )
        )

        clusters = (
            frontier_clusterer.cluster_frontiers(
                frontiers,
                env.simulator.robot_map,
            )
        )

        draw_clusters(
            screen,
            clusters,
        )

        # ----------------------------------------------------
        # Divider
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Drones
        # ----------------------------------------------------

        draw_drones(
            screen,
            env.simulator.drones,
        )

        pygame.display.flip()

        # Control visualization speed
        clock.tick(30)

    # ========================================================
    # Episode finished
    # ========================================================

    print(
        f"\nEpisode {episode} finished"
    )

    print(
        f"Steps: "
        f"{env.simulator.timestep}"
    )

    print(
        f"Final coverage: "
        f"{coverage:.2f}%"
    )

    print(
        f"Final redundancy: "
        f"{redundancy:.2f}%"
    )

    print(
        f"Total reward: "
        f"{total_reward:.3f}"
    )

    print(
        "Press SPACE for next episode"
    )

    # --------------------------------------------------------
    # Wait before resetting to next episode
    # --------------------------------------------------------

    waiting = True

    while waiting and running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False
                waiting = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    waiting = False

        clock.tick(30)


pygame.quit()