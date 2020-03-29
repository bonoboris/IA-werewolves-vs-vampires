from abc import ABC, abstractmethod
import random as rd
from typing import Sequence

from .game import Game, Action
from .utils import get_adj_case

class BasePlayer(ABC):
    """Abstract player."""
    def __init__(self, faction: int):
        self.faction = faction
    
    @abstractmethod
    def play(self, game: Game) -> Sequence[Action]:
        pass


class TestPlayer(BasePlayer):
    """Player which returns random valid actions."""

    def play(self, game: Game) -> Sequence[Action]:
        actions = list( )
        for unit in game.units(self.faction):
            start, num = unit[:2], unit[2]
            dest = rd.choice(get_adj_case(game.size(), start))
            actions.append(Action(*start, *dest, num))
        return actions
