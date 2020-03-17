from typing import Tuple, Sequence

import numpy as np

from .game import Game, Action
from .influence import get_influence_map
from .utils import dist

def sort_dist(unit:Tuple, seq:Sequence, thresh = 1.):
    i0, j0, num0 = unit
    over = list()
    under = list()
    for i, j, num in seq:
        if num0 > thresh * num:
            over.append((dist(i0, j0, i, j), i, j, num))
        else:
            under.append((dist(i0, j0, i, j), i, j, num))
    return list(sorted(over)), list(sorted(under))

def closer(from_case, to_case, shape):
    adjs = get_adj_case(from_case, shape)
    d0 = dist(from_case, to_case)
    return [c for c in adjs if dist(c, to_case) < d0]

def unitize(val):
    if val > 0: return 1
    elif val == 0: return 0
    else: return -1

def next_case(from_case, to_case, influence, forbidden, shape):
    candidates = [c for c in closer(from_case[:2], to_case[:2], shape) if c not in forbidden]
    print(f"{from_case}: {candidates}")
    max_infl, ret = -np.inf, None
    for c in candidates:
        if influence[c] > max_infl:
            max_infl, ret = influence[c], c
    return ret

class Dummy():
    def __init__(self, player):
        self.player = player
        self.opponent = Game.Vampire if player == Game.Werewolf else Game.Werewolf
    
    def play(self, game: Game):
        nearest_humans = dict()
        nearest_enemies = dict()
        allies = set(game.units(self.player))
        enemies = set(game.units(self.opponent))
        ker_size = max(game.m, game.n)
        if ker_size % 2 == 0: ker_size += 1
        influence = get_influence_map(game, self.player, kernel_size=ker_size)

        for unit in allies:
            nearest_humans[unit] = sort_dist(unit, game.humans(), thresh=1.)
            nearest_enemies[unit] = sort_dist(unit, enemies, thresh=1.5)

        actions = list()

        for unit , (winnable, forbidden_humans) in nearest_humans.items():
            if winnable:
                forbidden = []
                for fopp in nearest_enemies[unit][1]:
                    if fopp[0] < 3: forbidden.extend(get_adj_case(fopp[1:3], game.size()))
                    else: break
                for fhum in forbidden_humans:
                    if fhum[0] < 3: forbidden.append(fhum[1:3])
                
                for target in winnable:
                    print(f"* target {target}  forbidden {forbidden}")
                    ncase = next_case(unit, target[1:], influence, forbidden, game.size())
                    if ncase:
                        actions.append(Action(*unit[:2], *ncase, unit[2]))
                        break
        
        return actions

if __name__ == "__main__":
    print(get_adj_case((11,11), (20,20)))
    exp = {
        (10,10), (11,10), (10,11), (11,12), (12,11), (12,12), (12,10), (10, 12)
    }
    print(closer((5, 12), (5, 11), (20, 20)))
    print(closer((10, 10), (11, 11), (20, 20)))
    print(closer((10, 10), (11, 10), (20, 20)))