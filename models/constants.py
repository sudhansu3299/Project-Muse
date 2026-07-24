from enum import IntEnum
import pygame

class Cell(IntEnum):
    UNEXPLORED = -1
    FREE = 0
    OBSTACLE = 1

class Color:
    GRAY   = (120, 120, 120)
    WHITE  = (255, 255, 255)
    BLACK  = (0, 0, 0)
    BLUE   = (50, 120, 255)
    GREEN  = (0, 220, 0)
    YELLOW = (255, 220, 0)
    ORANGE = (255, 140, 0)
    PURPLE = (180, 0, 255)

class FontSize:
    TITLE = 24
    METRICS = 16
    BUTTON = 16