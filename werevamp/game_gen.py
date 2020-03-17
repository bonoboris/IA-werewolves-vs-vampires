import random as rd
import math
from time import time

from .game import Game, GamePlotter
from .str_utils import SimpleRepr
from scipy.special import binom
from tabulate import tabulate

import numpy as np


# def rd_fixed_sum(k, s):
#     class RoundUpDown:
#         def __init__(self):
#             self.last_up = True
#         def __call__(self, val):
#             res = math.floor(val) if self.last_up else math.ceil(val)
#             self.last_up = not self.last_up
#             return res
#     if k == 1:
#         return [s]
#     if k % 2 == 1:
#         s += 1
#     lower = (k-1) / s
#     ratios = [rd.uniform(lower, 1) for i in range(k)]
#     sum_ratios = sum(ratios)
#     as_float = [s*r/sum_ratios for r in ratios]
#     rounder = RoundUpDown()
#     vals = [rounder(v) for v in as_float]
#     return vals

_mem = dict()

def _gen_all2(k,s):
    if k == 1:
        return [[s]]
    if s == k:
        return [[1]*k]
    elif s < k:
        raise ValueError("Invalid k, s = ", k, s)
    elif (k,s) in _mem:
        return _mem[k,s]
    combs = list()
    for i in reversed(range(math.ceil(s / k), s - k + 2)):
        res = _gen_all2(k-1, s - i)
        combs.extend(([i, *els] for els in res))
        _mem[k, s] = combs
    return combs

def _gen_all(k,s):
    comb = [s-k+1] + [1]*(k-1)
    combs = [list(comb)]
    cnt = 1
    while True:
        i = 0
        found = False
        for j in range(i, k):
            if comb[j] < comb[i] - 1:
                comb[i] -= 1
                comb[j] += 1
                combs.append(list(comb))
                found = True
                i = j
                cnt += 1
        if not found:
            break
    return np.array(combs)

def uni(k,s):
    if k == 1:
        return [s]
    elif k==s:
        return [1] * k
    elif s < k:
        raise ValueError("s < k")
    elif k == 2:
        v1 = rd.randint(1, s-1)
        return [v1, s-v1]
    
    k1 = rd.randint(1, k-1)
    k2 = k - k1
    s1 = rd.randint(k1, s-k2)
    s2 = s - s1
    return uni(k1, s1) + uni(k2, s2)


def rd_fixed_sum(k, s, method="multinomial"):
    if k == s and k == 0:
        return []
    if s < k:
        raise ValueError("s must be superior to k")
    
    if k == 1:
        return [s]
    if method == "multinomial":
        return np.random.multinomial(s-k, [1/k]*k, size=1)[0] + 1
    elif method == "uniform":
        combs = _gen_all(k, s)
        return rd.choice(combs)
    elif method == "test":
        rem = s - k
        vals = []
        for i in range(k-1):
            v = rd.randint(0, rem)
            rem -= v
            vals.append(v+1)
        vals.append(s - sum(vals))
        return vals

    else: raise ValueError(f"Invalid method {method}")


def rd_coords(shape, k):
    m, n = shape
    idxs = rd.sample(range(m*n), k=k)
    return [(v // n, v % n) for v in idxs]


def transpose_game(game: Game) -> Game:
    game_t = Game(game.n, game.m)
    for (i,j), v in game.items():
        game_t[j, i] = v
    return game_t


class GameGenerator(SimpleRepr):
    def __init__(
        self,
        m:int=50,
        n:int=50,
        vamp_pop:int=100,
        were_pop:int=100,
        human_pop:int=100,
        vamp_spread:int=1,
        were_spread:int=1,
        human_spread:int=1,
    ) -> None:

        self.m = m
        self.n = n
        self.vamp_pop = vamp_pop
        self.were_pop = were_pop
        self.human_pop = human_pop
        self.vamp_spread = vamp_spread
        self.were_spread = were_spread
        self.human_spread = human_spread

        # if sym in ("x", "y") and (vamp_spread != 1 or were_spread != 1 or human_spread % 2 != 0 or human_pop % 2 != 0):
        #     raise ValueError("For `x` or `y` symmetry vamp_spread and were_spread must be equal to 1 and human_spread and human_pop must be multiple of 4.")
        # if sym in ("xy") and (vamp_spread != 1 or were_spread != 1 or human_spread % 4 != 0 or human_pop % 4 != 0):
        #     raise ValueError("For `x` or `y` symmetry vamp_spread and were_spread must be equal to 1 and human_spread and human_pop must be mutliple of 4.")
    
    def __call__(self) -> Game:
        coords = rd_coords((self.m, self.n), self.vamp_spread + self.were_spread + self.human_spread)

        vamp_nums = rd_fixed_sum(self.vamp_spread, self.vamp_pop)
        were_nums = rd_fixed_sum(self.were_spread, self.were_pop)
        human_nums = rd_fixed_sum(self.human_spread, self.human_pop)

        vals = []
        vals.extend(((Game.Vampire, num) for num in vamp_nums))
        vals.extend(((Game.Werewolf, num) for num in were_nums))
        vals.extend(((Game.Human, num) for num in human_nums))

        assert len(coords) == len(vals)
        assert len(vals) == self.vamp_spread + self.were_spread + self.human_spread

        g = Game(self.m, self.n)
        g.update(zip(coords, vals))

        assert g.vampire_pop() == self.vamp_pop
        assert g.werewolf_pop() == self.were_pop
        assert g.human_pop() == self.human_pop

        return g

class SymGameGenerator():
    def __init__(
        self,
        m:int=50,
        n:int=50,
        players_pop:int=100,
        human_pop:int=100,
        human_spread:int=1,
        sym:str="x",
    ) -> None:

        self.m = m
        self.n = n
        self.players_pop = players_pop
        self.human_pop = human_pop
        self.human_spread = human_spread
        self.sym = sym
    
    def __call__(self):
        g = Game(self.m, self.n)
        if self.sym == "x" or self.sym == "r":
            inv_y = True if self.sym == "r" else False
            if self.m % 2 == 0:
                m_half = self.m // 2
                last_row = "mirror"
            else:
                m_half = self.m // 2 + 1
                last_row = "double"              
            top_half = GameGenerator(
                m=m_half, n=self.n,
                vamp_pop=0, were_pop=0, vamp_spread=0, were_spread=0,
                human_spread=self.human_spread // 2, human_pop=self.human_pop //2
            )()
            
            g = self.mirror_x(top_half, last_row=last_row, inv_y=inv_y)
            # assert g.human_pop() == 2 * top_half.human_pop()

            pi, pj = rd.randint(0, self.m // 2 - 1), rd.randint(0, self.n -1)
            while g[pi, pj]:
                pi, pj = rd.randint(0, self.m // 2 - 1), rd.randint(0, self.n -1)

            g[pi, pj] = (Game.Vampire, self.players_pop)
            mpi = self.m - 1 - pi
            mpj = self.n - 1 - pj if inv_y else pj 
            g[mpi, mpj] = (Game.Werewolf, self.players_pop)
        
        elif self.sym == "y":
            gen_t = SymGameGenerator(**vars(self))
            gen_t.n, gen_t.m = gen_t.m, gen_t.n
            gen_t.sym = "x"
            game_t = gen_t()
            g = transpose_game(game_t)

        return g

    @staticmethod
    def mirror_x(game: Game, last_row:str="mirror", inv_y:bool=False) -> Game:
        if last_row == "mirror":
            mirrored = Game(2 * game.m, game.n)
            for (i, j), v in game.items():
                mi = mirrored.m - 1 - i
                mj = game.n - 1 - j if inv_y else j
                mirrored[i,j] = v
                mirrored[mi, mj] = v
        elif last_row == "double":
            mirrored = Game(2 * game.m - 1, game.n)
            for (i, j), v in game.items():
                mi = mirrored.m - 1 - i
                mj = game.n - 1 - j if inv_y else j
                if mi == i:
                    mirrored.add_create((i,j), v)
                    mirrored.add_create((mi, mj), v)
                else:
                    mirrored[i,j] = v
                    mirrored[mi, mj] = v
        else: raise ValueError(f"Unknown value {last_row} for last_row parameter.")
        return mirrored


if __name__ == "__main__":
    from time import sleep

    hp = 100
    pp = 100

    gg = SymGameGenerator(m= 15, n=21, human_spread=10, sym='r', human_pop=hp, players_pop=pp)
    # gg = GameGenerator(m= 15, n=21, human_spread=10, human_pop=hp, vamp_pop=pp, were_pop=pp)
    g = gg()
    
    gp = GamePlotter(g)
    if g.human_pop() != hp or g.vampire_pop() != pp or g.werewolf_pop() != pp:
        print(g.populations())        
        print("vamp")
        print(tabulate(list(g.vampires())))
        print("were")
        print(tabulate(list(g.werewolves())))
        print("humans")
        print(tabulate(list(g.humans())))
        input("Continue...")

    for i in range(10000):
        g = gg()
        gp.update(g)
        if g.human_pop() != hp or g.vampire_pop() != pp or g.werewolf_pop() != pp:
            print("iter", i)
            print(g.populations())        
            print("vamp")
            print(tabulate(list(g.vampires())))
            print("were")
            print(tabulate(list(g.werewolves())))
            print("humans")
            print(tabulate(list(g.humans())))
            input("Continue...")
        sleep(1)
