from typing import Sequence, Iterable, Dict, Tuple, Optional, Set, List
from itertools import product

import numpy as np

from ..utils import Coords
from ..game import Game
from ..plotting import TextAnnot
from .subset_iter import SubsetIter

IVec3D = Tuple[int, int, int]

def dist(c1: Coords, c2: Coords) -> int:
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))

def dist_inf(c1: Coords, c2: Coords, val) -> int:
    return (c1[0] >= c2[0] - val and 
            c1[0] <= c2[0] + val and 
            c1[1] >= c2[1] - val and
            c1[1] <= c2[1] + val)

def iter_except(it: Iterable, idx: int):
    return (val for i, val in enumerate(it) if idx != i)

class LazyJoiningMat:
    def __init__(self, game:Game, faction: int) -> None:
        self.vals: Dict[IVec3D, int] = dict()
        self.game = game
        self.faction = faction
        self.other_faction = Game.enemy_faction(faction)
        self.funits = list(self.game.units(self.faction))
        self.ounits = list(self.game.units(self.other_faction))
        self.hunits = list(self.game.humans())
        self.faction_human_dist = [[dist(f[:2], h[:2]) for h in self.hunits] for f in self.funits]

    def __getitem__(self, idx: IVec3D) -> int:
        if idx not in self.vals:
            self._compute_human(idx)
        return self.vals[idx]

    def _compute(self, idx: IVec3D) -> int:
        idx = tuple(idx)
        s, i, j = idx
        val = 0
        for ie, je, ne in self.game.units(self.faction):
            if i >= ie - s and i <= ie + s and j >= je - s and j <= je + s:
                val += ne
        self.vals[idx] = val
        return val
    
    def _compute_human(self, idx: IVec3D) -> int:
        """With `idx`=(s,i,j), compute the maximum number of `self.faction` units which can go to square (i,j) in s steps accounting for eaten humans on the way."""
        idx = tuple(idx)
        s, i, j = idx
        
        # distance between humains and (i,j)
        Hc_dist = [dist((hi, hj), (i, j)) for hi, hj, _ in self.hunits]

        # self.faction units at s step or less from (i,j)
        F = [k for k, (fi, fj, _) in enumerate(self.funits) if dist_inf((fi, fj), (i,j), s)]

        # humans at s step or less from (i,j)
        H = [k for k, (hi, hj, _) in enumerate(self.hunits) if Hc_dist[k] <= s]

        # dict:
        #   key: a human at s step or less
        #   value: set of self.faction units which can go to key coords and then to (i,j) in s step or less
        HF = {}
        for h in H:
            HF[h] = {f for f in F if self.faction_human_dist[f][h] + Hc_dist[h] <= s}

        # dict:
        #   key: a human at s step or less
        #   value: set of subset of self.faction units which can go to key coords and then to (i,j) in s step or less and are enough to eat the humans
        HPF = {}
        for h in H:
            HPFh = [set()]
            hnum = self.hunits[h][2]
            it = SubsetIter(HF[h])
            for comb in it:
                tot_fnum = sum((self.funits[f][2] for f in comb))
                # print("*", comb, tot_fnum)
                if tot_fnum > hnum:
                    HPFh.append(comb)
                    it.exclude()
            HPF[h] = HPFh

        # Assign subsets of self.faction units to eat as many humans possible before going to (i,j) in s step or less 
        max_hum = 0
        max_assign = None
        for PF_list in product(*(HPF[h] for h in H)):
            union = set().union(*PF_list)
            if len(union) < sum(map(len, PF_list)):
                continue
            else:
                non_empty = (h for k, h in enumerate(H) if PF_list[k])
                hum = sum((self.hunits[h][2] for h in non_empty))
                if hum > max_hum:
                    max_hum = hum
                    max_assign = PF_list
        
        # Final value all self.faction units at s step or less plus the eaten humans
        if max_hum > 0:
            assign = {}
            for k, h in enumerate(H):
                if max_assign[k]:
                    assign[self.hunits[h]] = [self.funits[f] for f in max_assign[k]]
            # print("human accounted !")
            # print(f"s,i,j = {s},{i},{j}")
            # print("assign", assign)
        val = max_hum + sum((self.funits[f][2] for f in F))
        self.vals[idx] = val
        return val


    def can_eat_in(self, steps, target):
        """With `idx`=(s,i,j), compute the maximum number of `self.faction` units which can go to square (i,j) in s steps accounting for eaten humans on the way."""
        s = steps
        i,j = self.hunits[target][:2]
        
        # distance between humains and (i,j)
        Hc_dist = [dist((hi, hj), (i, j)) for hi, hj, _ in self.hunits]

        # self.faction units at s step or less from (i,j)
        F = [k for k, (fi, fj, _) in enumerate(self.funits) if dist_inf((fi, fj), (i,j), s)]

        # humans at s step or less from (i,j)
        H = [k for k, (hi, hj, _) in enumerate(self.hunits) if Hc_dist[k] <= s and k != target]

        # dict:
        #   key: a human at s step or less
        #   value: set of self.faction units which can go to key coords and then to (i,j) in s step or less
        HF = {}
        for h in H:
            HF[h] = {f for f in F if self.faction_human_dist[f][h] + Hc_dist[h] <= s}


        # dict:
        #   key: a human at s step or less
        #   value: set of subset of self.faction units which can go to key coords and then to (i,j) in s step or less and are enough to eat the humans
        HPF = {}
        for h in H:
            HPFh = [set()]
            hnum = self.hunits[h][2]
            it = SubsetIter(HF[h])
            for comb in it:
                tot_fnum = sum((self.funits[f][2] for f in comb))
                # print("*", comb, tot_fnum)
                if tot_fnum > hnum:
                    HPFh.append(comb)
                    it.exclude()
            HPF[h] = HPFh


        # Assign subsets of self.faction units to eat as many humans possible before going to (i,j) in s step or less 
        max_hum = 0
        max_assign = None
        for PF_list in product(*(HPF[h] for h in H)):
            union = set().union(*PF_list)
            if len(union) < sum(map(len, PF_list)):
                continue
            else:
                non_empty = (h for k, h in enumerate(H) if PF_list[k])
                hum = sum((self.hunits[h][2] for h in non_empty))
                if hum > max_hum:
                    max_hum = hum
                    max_assign = PF_list
        
        # Final value all self.faction units at s step or less plus the eaten humans
        if max_hum > 0:
            assign = {}
            for k, h in enumerate(H):
                if max_assign[k]:
                    assign[self.hunits[h]] = [self.funits[f] for f in max_assign[k]]
            # print("human accounted !")
            # print(f"s,i,j = {s},{i},{j}")
            # print("assign", assign)
        val = max_hum + sum((self.funits[f][2] for f in F))
        return val




    def keys(self, step: Optional[int] = None) -> Iterable[IVec3D]:
        if step is None:
            return self.vals.keys()
        else:
            return (k for k in self.vals.keys() if k[0] == step)

    def computed_at(self, i: int, j: int) -> Sequence[Tuple[int, int]]:
        return ((s, v) for k, v in self.vals.items() if k[1:] == (i,j))

    def computed_steps(self) -> Iterable[int]:
        return list(sorted(set((s for s, _, _ in self.keys()))))

    def computed_at_step(self, step) -> Iterable[Coords]:
        return sorted(((i,j) for s, i, j in self.keys() if s == step))
    
    def step_mat(self, num: int) -> np.ndarray:
        mat = np.empty(self.game.size(), dtype=int)
        mat.fill(-1)
        for s in reversed(self.computed_steps()):
            for i, j in self.computed_at_step(s):
                if 1.5 * self[s,i,j] > num:
                    mat[i, j] = s
                else:
                    mat[i,j] = 0
        return mat
    
    def step_im_mat(self, num: int) -> Tuple[np.ndarray, List[TextAnnot]]:
        im_mat = np.zeros(self.game.size() + (3,), dtype=int)
        mat = self.step_mat(num)
        text_list = []
        for i in range(self.game.m):
            for j in range(self.game.n):
                if mat[i,j] > 0:
                    pval = round(125*(mat[i,j] + 1) / max(self.game.m, self.game.n))
                    im_mat[i,j] = [pval, 0, pval]
                    text_list.append((i,j, mat[i,j]))
                elif mat[i,j] == 0:
                    im_mat[i,j] = [0, 50, 0]
        return im_mat, text_list


from ..runner import BaseRunner
from ..plotting import ButtonGamePlotter, set_text
from matplotlib import pyplot as plt

class Runner(BaseRunner):
    def __init__(self, game: Game, indices, num):
        super().__init__()
        self.game = game
        self.indices_it = iter(indices)
        self.jm = LazyJoiningMat(game, Game.Werewolf)
        self.num = num

    def __call__(self, game=None):
        idx = next(self.indices_it)
        print(f"Computing val at {idx}")
        print(f"Value at {idx}: {self.jm[idx]}")
        return (self.game,) + self.jm.step_im_mat(self.num) 


class Plotter(ButtonGamePlotter):
    def update(self, runner_ret):
        self.influence_mode = "None"
        game, mat, text_annots = runner_ret
        self.ax.texts.clear()                    
        self._update_data(mat)
        set_text(self.ax, text_annots)
        set_text(self.ax, game.humans(), bbox_col='b')
        set_text(self.ax, game.vampires(), bbox_col='g')
        set_text(self.ax, game.werewolves(), bbox_col='r')
        plt.draw()


if __name__ == "__main__":
    from ..game_gen import GameGenerator
    # m = LazyJoiningMat(GameGenerator()(), Game.Vampire)
    
    game = Game(30, 30)
    game[28,6] = Game.Vampire, 50
    game[15,2] = Game.Human, 21
    game[19,18] = Game.Human, 11
    game[16,3] = Game.Werewolf, 31

    indices = [(8,20,10), (9,20,10), (100,20,10), (9,19,9)]
    Plotter(Runner(game, indices, 50))
