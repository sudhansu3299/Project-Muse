import random

from strategy.exploration_strategy import ExplorationStrategy
from models.action import Action

class RandomStrategy(ExplorationStrategy):

    def __init__(self, communication_radius=10):
        super().__init__("Random")

    def choose_action(
            self,
            agent,
            robot_map,
            true_map,
            nearby_agents,
    ):

        actions = []

        for action in Action:

            dx, dy = action.delta()

            nx = agent.x + dx
            ny = agent.y + dy

            if true_map.is_valid(nx, ny):
                actions.append(action)

        return random.choice(actions)