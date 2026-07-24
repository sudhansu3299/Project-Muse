#ABC: abstract base classes
#Interface for strategies

from abc import ABC, abstractmethod
from models.action import Action
import math

class ExplorationStrategy(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_action(
            self,
            agent,
            robot_map,
            environment,
            nearby_agents,
    ):
        pass

    def valid_actions(self, agent, environment):
        """
        Return actions that do not hit obstacles or leave the map.
        """
        actions = []

        for action in [
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]:
            if environment.is_valid_move(agent, action):
                actions.append(action)

        return actions

    def get_neighbors(self, x, y):
        return [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )