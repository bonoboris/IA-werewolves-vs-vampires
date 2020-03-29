import copy
from collections import defaultdict
from numbers import Integral
from typing import Tuple, Dict, Iterator, NamedTuple, Optional, Union, List, Mapping, Optional
from time import sleep, time

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from .str_utils import SimpleRepr
from .utils import dist, Coords


class Action(NamedTuple, SimpleRepr):
    start_i: int
    start_j: int
    dest_i: int
    dest_j: int
    num: int

    def start(self):
        return self.start_i, self.start_j
    
    def dest(self):
        return self.dest_i, self.dest_j


class Game(dict):
    """Game state and usefull methods.

    Usage
    -----
    g = Game(10, 5)
    """

    Vampire = 0
    Werewolf = 1
    Human = 2

    # ------ Public Methods ------

    def __init__(self, m: int, n: int, initial_pop : Dict = dict()):
        """
        Args
        ----
            m: Number of rows
            n: Number of cols
            initial_pop : a dictionnary in the form {index_population : [[(coord),number]]
                                                     
        """
        self.m = m
        self.n = n

        for kind, initial_data in initial_pop.items():
            for group in initial_data:
                coords = group[0]
                n = group[1]
                self.__setitem__(coords, (kind,n))

    def size(self) -> Coords:
        """Grid size."""
        return self.m, self.n

    def populations(self) -> Dict[int, int]:
        """Populations dictionnary."""
        dct = {self.Vampire: 0, self.Werewolf: 0, self.Human: 0}
        for ty, num in self.values():
            dct[ty] += num
        return dct

    def human_pop(self) -> int:
        """Total human population."""
        return sum((num for _,_,num in self.humans()))
    
    def werewolf_pop(self) -> int:
        """Total werewolf population."""
        return sum((num for _,_,num in self.werewolves()))
    
    def vampire_pop(self) -> int:
        """Total vampire population."""
        return sum((num for _,_,num in self.vampires()))

    def humans(self) -> Iterator[Tuple[int, int, int]]:
        """Human units iterator."""
        return ((*key, val[1]) for key, val in self.items() if val[0]==self.Human)

    def werewolves(self) -> Iterator[Tuple[int, int, int]]:
        """Werewolf units iterator."""
        return ((*key, val[1]) for key, val in self.items() if val[0]==self.Werewolf)

    def vampires(self) -> Iterator[Tuple[int, int, int]]:
        """Vampires units iterator."""
        return ((*key, val[1]) for key, val in self.items() if val[0]==self.Vampire)

    def units(self, kind:int):
        if kind == Game.Human:
            return self.humans()
        elif kind == Game.Werewolf:
            return self.werewolves()
        elif kind == Game.Vampire:
            return self.vampires() 

    def winner(self):
        ret = None
        if self.vampire_pop() == 0: ret = Game.Werewolf
        elif self.werewolf_pop() == 0: ret = Game.Vampire
        return ret

    @staticmethod
    def enemy_faction(ally_faction: int) ->  int:
        return Game.Vampire if ally_faction == Game.Werewolf else Game.Werewolf
    # ------------

    def to_matrix(self) -> np.ndarray:
        mat = np.zeros((self.m, self.n, 3), dtype=int)
        for (i,j), (ty, nb) in self.items():
            mat[i,j,ty] = nb
        return mat

    def to_matrix_CHW(self) -> np.ndarray:
        mat = np.zeros((3,self.m, self.n), dtype=int)
        for (i, j), (ty, nb) in self.items():
            mat[ty, i, j] = nb
        return mat

    @classmethod
    def from_matrix(cls, mat: np.ndarray):
        m, n, _ = mat.shape
        inst = cls(m, n)
        for i in range(m):
            for j in range(n):
                has_units = False
                for ty in range(3):
                    if mat[i,j,ty] > 0:
                        if not has_units:
                            inst[i,j] = (ty, mat[i,j,ty])
                            has_units = True
                        else:
                            raise ValueError("Cannot have multiple units types aat the same coords.")
    # ------------

    @staticmethod
    def fight_nonhuman(num_att, num_def):
        r = num_att / num_def
        if r > 1.5:
            return True, num_att
        elif r > 1:
            p = np.random.random()
            if p < r - 0.5:
                return True, np.random.binomial(num_att, r - 0.5)
            else:
                return False, np.random.binomial(num_def, 1.5-r)
        else:
            p = np.random.random()
            if p < r / 2.:
                return True, np.random.binomial(num_att, r /2.)
            else:
                return False, np.random.binomial(num_def, 1 - r/2.) 

    @staticmethod
    def fight_human(num_att, num_hum):
        r = num_att / num_hum
        if r > 1:
            return True, num_att + num_hum
        else:
            p = np.random.random()
            if p < r / 2.:
                return True, np.random.binomial(num_att, r /2.)
            else:
                return False, np.random.binomial(num_hum, 1 - r/2.)         


    def register_actions(self, kind: int, actions:List[Action]):
        if not actions:
            raise ValueError("At least one action must be provided!")
        self.check_actions(kind, actions)
        grouped_dests = defaultdict(int)

        for action in actions:
            self[action.start()] = (kind, self[action.start()][1] - action.num)
            grouped_dests[action.dest()] += action.num

        for dest, num in grouped_dests.items():
            dest_val = self[dest] 
            if dest_val:
                dest_ty, dest_num = dest_val
                if dest_ty == self.Human:
                    iswin, new_num = self.fight_human(num, dest_num)
                elif dest_ty == kind:
                    iswin, new_num = True, dest_num + num  # hackish
                else:
                    iswin, new_num = self.fight_nonhuman(num, dest_num)
                new_kind = kind if iswin else dest_ty
                self[dest] = (new_kind, new_num)
            else:
                self[dest] = (kind, num)


    def check_actions(self, kind:int, actions: List[Action]) -> bool:
        starts = set()
        dests = set()
        for action in actions:
            self.check_action(action, kind)
            if action.start() in dests or action.dest() in starts:
                raise ValueError("Invalid actions: a square cannot send and receive units in the same turn")
            if action.start() in starts:
                raise ValueError("A square cannot send units to multiple square at the same turn")
            starts.add(action.start())
            dests.add(action.dest())

    def check_action(self, action: Action, kind: Optional[int] = None) -> bool:
        """Check if an action is valid and raise ValueError if not. If `kind` is specified also check if the right type of units in the starting coords."""
        start = action.start()
        dest = action.dest()
        num = action.num
        
        try:
            self._check_coords(*start)
            self._check_coords(*dest)
        except IndexError:
            raise ValueError(f"Invalid action {action}: start or/and destination outside of the board ({self.m}x{self.n})")
        

        if not self[start]:
            raise ValueError(f"Invalid action {action}: square {start} is empty.")
        start_ty, start_num = self[start] 
        if start_num < num:
            raise ValueError(f"Invalid action {action}: square {start} contains only {from_num} units.")
        if num == 0:
            raise ValueError(f"Invalid action {action}: Must move at least 1 unit.")

        if kind and start_ty != kind:
            raise ValueError(f"Invalid kind: expected {start_ty} but received {kind}")

        if dist(start, dest) != 1:
            raise ValueError(f"Invalid action {action}: cannot move units from squares {start} to {dest}.")
        
    def add_create(self, coords, value):
        if not self[coords]:
            return self.__setitem__(coords, value)
        else:
            cur_ty, cur_num = self[coords]
            ty, diff = self._pair_int(value)
            if cur_ty != ty:
                raise(f"Value error mismatch between current units type {cur_ty} at coords {coords} and `value` parameter {value}")
            else:
                self[coords] = (ty, cur_num + diff)


    # ------ List method ------

    def __getitem__(self, key: Coords):
        """Returns (type, number) at coords `key`=(i, j) or None of the case is empty."""
        i, j = self._pair_int(key)
        self._check_coords(i, j)
        if key in super().keys():
            return super().__getitem__(key)
        else:
            return None
    
    def __setitem__(self, key: Coords, value: Tuple[int, int]):
        """Set the `value`=(type, number) at coords `key`=(i,j)."""
        # Validate args
        i, j = self._pair_int(key)
        self._check_coords(i, j)

        ty, nb = self._pair_int(value)
        if ty not in [0,1,2]:
            raise ValueError(f"Invalid type {ty}, must be one of {{0, 1 , 2}}.")
        if nb < 0:
            raise ValueError("Numbers must be striclty superior to 0.")


        if nb == 0:
            # Del value
            return super().__delitem__(key)

        # Set value
        return super().__setitem__(key, value)
    
    def __delitem__(self, key: Coords):
        """Delete the units at coords `key`=(i,j) if any."""
        i, j = self._pair_int(key)
        self._check_coords(i, j)
        if key in super().keys():
            super().__delitem__(key)

    def __repr__(self):
        return f'Game(m={self.m}, n={self.n})\n{super().__repr__()}'

    # ------ Helpers ------

    def _check_coords(self, i ,j):
        if i >= self.m or i < 0:
            raise IndexError(f"Grid x-index {i} outside of range [0, {self.m})")
        if j >= self.n or j < 0:
            raise IndexError(f"Grid y-index {j} outside of range [0, {self.n})")

    @staticmethod
    def _pair_int(v, err_type=ValueError):
        i, j = v
        if not isinstance(i, Integral) or not isinstance(j, Integral):
            raise err_type(f"Expected a pair of int, got: {v}")
        return i, j

class GamePlotter():
    def __init__(self, game: Game = None, influence:Optional = None):
        plt.ion()
        self.m, self.n = game.size()
        self.fig: plt.Figure = None
        self.ax: plt.Axes = None
        self.fig, self.ax = plt.subplots()
        self.im: Optional[mpl.image.AxesImage] = None
        self.fig.tight_layout()
        self._first(game, influence)

    def update(self, game: Game, influence: Optional = None) -> None:
        if self.im is None:
            self._first(game, influence)
        else:
            self._redraw(game, influence)
        self.fig.canvas.draw()
        plt.pause(0.001)

    def _first(self, game: Game, influence=None):
        if influence is None:
            self.im = self.ax.imshow(self.get_game_rgb(game))
            for (i, j), (_,num) in game.items():
                text = self.ax.text(j, i, num, ha="center", va="center", color="w")
        else:
            self.im = self.ax.imshow(influence)
            self.ax.texts.clear()
            for i, j, num in game.humans():
                text = self.ax.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor='b', alpha=0.5))
            for i, j, num in game.vampires():
                text = self.ax.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor='r', alpha=0.5))
            for i, j, num in game.werewolves():
                text = self.ax.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor='g', alpha=0.5))            

    def _redraw(self, game: Game, influence=None):
        if influence is None:
            self.im.set_data(self.get_game_rgb(game))
            self.ax.texts.clear()
            for (i, j), (_,num) in game.items():
                text = self.ax.text(j, i, num, ha="center", va="center", color="w")
        
        else:
            self.im.set_data(influence)
            self.ax.texts.clear()
            for i, j, num in game.humans():
                text = self.ax.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor='b', alpha=0.5))
            for i, j, num in game.vampires():
                text = self.ax.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor='r', alpha=0.5))
            for i, j, num in game.werewolves():
                text = self.ax.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor='g', alpha=0.5))
    
    @staticmethod
    def get_game_rgb(game: Game):
        return np.where(game.to_matrix() > 0, 122, 0)



if __name__ == "__main__":
    g = Game(10, 10)
    g[2, 7] = (Game.Vampire, 10)
    g[8, 1] = (Game.Werewolf, 10)
    g[5, 4] = (Game.Human, 5)
    
    gp = GamePlotter(g)
    sleep(1)
    a = Action(2,7,2,8,2)
    print(a)
    g.register_actions(Game.Vampire, [a])
    # del g[2,7]
    # g[2,8] = (Game.Vampire, 5)
    gp.update(g)
    input()