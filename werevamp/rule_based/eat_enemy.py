from typing import Sequence
from math import inf

from ..game_gen import GameGenerator
from ..game import Game, Action
from ..utils import Coords, sign, sub_coords, add_coords, get_adj_case
from ..utils import dist, min_argmin, get_adj_case
from ..influence import get_influence_map

def nearest_corner(shape, coords):
    m,n = shape
    i,j = coords
    ri, ri = m-i-1, n-i-1
    return (sign(ri-i), sign(rj-j))

def direction(from_coords, to_coords):
    dif = sub_coords(to_coords, from_coords)
    udif = tuple(map(sign, dif))
    return 


def eat_enemy(shape, ally_coords, enemy_coords):
    dif = sub_coords(enemy_coords, ally_coords)
    udif = tuple(map(sign, dif))
    adif = tuple(map(abs, dif))
    if adif[0] > adif[1]:
        d = (sign(dif[0]), 0)
    elif adif[0] < adif[1]:
        d = (0, sign(dif[1]))
    else:
        d = udif
    return add_coords(ally_coords, d)


def eat_enemy_action(game: Game, faction:int, influence):
    allies = list(game.units(faction))
    assert len(allies) == 1
    ai, aj, anum = allies[0]
    enemies = list(game.units(game.enemy_faction(faction)))
    tot_num_enemy = sum((e[2] for e in enemies))

    eatable_enemies = [e for e in enemies if anum >= 1.5*e[2]]

    if eatable_enemies:
        dist_ene = [dist((ai, aj), e[:2]) for e in eatable_enemies]
        min_dist, min_indices = min_argmin(dist_ene)
        target = min_indices[0]
        target_coords = eatable_enemies[0]
        dest = eat_enemy(game.size(), (ai, aj), target_coords)
        # print("eat certain")
        return Action(ai, aj, *dest, anum)


    rand_eatable_enemies = [e for e in enemies if 1.5*anum > e[2] and anum < 1.5*e[2]]
    if anum < tot_num_enemy and rand_eatable_enemies:
        dist_ene = [dist((ai, aj), e[:2]) for e in rand_eatable_enemies]
        min_dist, min_indices = min_argmin(dist_ene)
        target = min_indices[0]
        target_coords = rand_eatable_enemies[0]
        dest = eat_enemy(game.size(), (ai, aj), target_coords)
        # print("eat random")
        return Action(ai, aj, *dest, anum)

    else:
        # print("flee")
        dest = max_inluence_or_rd(get_adj_case(game.size(), (ai, aj)), influence)
        return Action(ai, aj, *dest, anum)


    
def max_inluence_or_rd(seq: Sequence[Coords], influence=None):
    if len(seq) == 0:
        raise ValueError("Cannot select point from an empty sequence")
    
    if influence is not None:
        max_infl, ret = -inf, None
        for c in seq:
            infl = influence[c]
            if infl > max_infl:
                max_infl, ret = infl, c
    else:
        ret = rd.choice(seq)
    return ret


if __name__ == "__main__":
    from ..runner import CallableRuner
    from ..plotting import ButtonGamePlotter
    import random as rd

    class Eat():
        def __init__(self):
            self.alt = 0

        def __call__(self, game: Game):
            if game.werewolf_pop() == 0: return None
            ai, aj, anum = next(game.vampires())
            ei, ej, enum = next(game.werewolves())
            
            nac = (ai, aj) if self.alt else eat_enemy(game.size(), (ai, aj), (ei,ej))
            nec = (ei, ej) if not self.alt else rd.choice(get_adj_case(game.size(), (ei, ej)))

            ngame = Game(*game.size())
            ngame[nec] = Game.Werewolf, enum
            ngame[nac] = Game.Vampire, anum

            self.alt = 1 - self.alt
            return ngame
    
    gg = GameGenerator(30,30, vamp_pop=200, human_pop=0, human_spread=0)
    g = gg()

    cr = CallableRuner(Eat(), g)
    ButtonGamePlotter(cr)