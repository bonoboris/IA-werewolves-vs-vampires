import copy
from numbers import Integral
from typing import Tuple, Dict, Iterator


class Game(dict):
    """Game state and usefull methods.

    Usage
    -----
    g = Game(10, 5)
    """

    Human = 0
    Werewolf = 1
    Vampire = 2

    # ------ Public Methods ------

    def __init__(self, m: int, n: int, initial_pop : dict):
        """
        Args
        ----
            m: Number of rows
            n: Number of cols
            initial_pop : a dictionnary in the form {index_population : [[(coord),number]]
                                                     
        """
        self.m = m
        self.n = n

        self._populations = {0:0, 1:0, 2:0}
        for kind, initial_data in initial_pop.items():
            for group in initial_data:
                coords = group[0]
                n = group[1]
                self.__setitem__(coords, (kind,n))

    def size(self) -> Tuple[int, int]:
        """Grid size."""
        return self.m, self.n

    def populations(self) -> Dict[int, int]:
        """Populations dictionnary."""
        return copy.copy(self._populations)

    def human_pop(self) -> int:
        """Total human population."""
        return self._populations[self.Human]
    
    def werewolf_pop(self) -> int:
        """Total werewolf population."""
        return self._populations[self.Werewolf]
    
    def vampire_pop(self) -> int:
        """Total vampire population."""
        return self._populations[self.Vampire]

    def humans(self) -> Iterator[Tuple[int, int, int]]:
        """Human units iterator."""
        return {(*key, val[1]) for key, val in self.items() if val[0]==self.Human}

    def werewolves(self) -> Iterator[Tuple[int, int, int]]:
        """Werewolf units iterator."""
        return {(*key, val[1]) for key, val in self.items() if val[0]==self.Werewolf}

    def vampires(self) -> Iterator[Tuple[int, int, int]]:
        """Vampires units iterator."""
        return {(*key, val[1]) for key, val in self.items() if val[0]==self.Vampire}

    # ------ List method ------

    def __getitem__(self, key: Tuple[int, int]):
        i, j = self._pair_int(key)
        self._check_coords(i, j)
        if key in super().keys():
            return super().__getitem__(key)
        else:
            return None
    
    def __setitem__(self, key: Tuple[int, int], value: Tuple[int, int]):
        # Validate args
        i, j = self._pair_int(key)
        self._check_coords(i, j)

        ty, nb = self._pair_int(value)
        if ty not in [0,1,2]:
            raise ValueError(f"Invalid type {ty}, must be one of {{0, 1 , 2}}.")
        if nb < 1:
            raise ValueError("Numbers must be striclty superior to 0.")
        
        # Actualize populations
        if self[i, j]:
            ty_, nb_ = self[i, j]
            self._populations[ty_] -= nb_
        self._populations[ty] += nb

        # Set value
        return super().__setitem__(key, value)
    
    def __delitem__(self, key: Tuple[int, int]):
        i, j = self._pair_int(key)
        self._check_coords(i, j)
        if key in super().keys():
            ty, nb = super().pop(key)
            self._populations[ty] -= nb

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

if __name__ == "__main__":
    g = Game(10, 5)
    g[2, 3] = (Game.Vampire, 10)
    print(g)
    print(g.vampire_pop())
    del g[2, 3]
    print(g.vampire_pop())
    print(g[2, 3])
    print(g[3, 3])
    print(g[2, 3])
    # print(g[11, 3]) --> should raise IndexError