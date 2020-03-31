from ..game import Game, action_to_mov_command
from ..influence import get_influence_map
from .eat_humans import glouton, transfo
from .eat_enemy import eat_enemy_action

class Player():
    def __init__(self, faction=None, return_actions=False):
        self.faction = faction
        self.return_actions = return_actions
    
    def play(self, game: Game, faction=None):
        faction = faction or self.faction
        if faction is None:
            raise ValueError("No value for faction (argument or member)")
        influence = get_influence_map(game, self.faction)
        actions = transfo(game.size(), *glouton(game, self.faction), influence=influence)
        if not actions:
            actions = [eat_enemy_action(game, self.faction, influence)]
        if self.return_actions:
            return actions
        return action_to_mov_command(actions)