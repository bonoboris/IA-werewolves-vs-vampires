from typing import Sequence, Mapping, Tuple, Any, List
from collections import defaultdict
import random as rd
import math
from time import time, sleep

import numpy as np
from matplotlib import pyplot as plt

from ..game import Game, Action
from ..utils import dist, get_adj_case, get_rectangle, clamp, sign, Coords, min_argmin
from ..influence import get_influence_map

TUnit = Tuple[int, int, int]   # i, j, num
TDistUnit = Tuple[int, int, int]  # dist, i, j, num

def sort_dist(unit:TUnit, seq:Sequence[TUnit], thresh:float= 1.) -> Tuple[Sequence[TDistUnit], Sequence[TDistUnit]]:
    i0, j0, num0 = unit
    over = list()
    under = list()
    for i, j, num in seq:
        if num0 > thresh * num:
            over.append((dist(i0, j0, i, j), i, j, num))
        else:
            under.append((dist(i0, j0, i, j), i, j, num))
    return list(sorted(over)), list(sorted(under))


def closer(shape: Coords, from_case: Coords, to_case: Coords, forbidden=set()):
    adjs = [c for c in get_adj_case(from_case, shape) if c not in forbidden]
    adjs_dist = [dist(c, to_case) for c in adjs]
    return [adjs[i] for i in min_argmin(adjs_dist)[1]]


def next_case(shape: Coords, from_case: Coords, to_case: Coords, influence = None, forbidden:Sequence[Coords] = set()):
    if from_case == to_case:
        return to_case
    return max_inluence_or_rd(closer(shape, from_case, to_case, forbidden), influence)


def max_inluence_or_rd(seq: Sequence[Coords], influence=None):
    if len(seq) == 0:
        raise ValueError("Cannot select point from an empty sequence")
    
    if influence is not None:
        max_infl, ret = -np.inf, None
        for c in seq:
            infl = influence[c]
            if infl > max_infl:
                max_infl, ret = infl, c
    else:
        print("rd")
        ret = rd.choice(seq)
    return ret


# ---- n->1 merging heuristic ----

def get_closest_merge_points(shape: Coords, c1: Coords, c2: Coords, forbidden:Sequence[Coords]=set()) -> Tuple[int, Sequence[Coords]]:
    ret = None
    i1, j1 = c1
    i2, j2 = c2
    di, dj = i2 - i1, j2 - j1
    sdi, sdj = sign(di, zero_val=1), sign(dj, zero_val=1)
    d_merge = math.ceil(max(abs(di), abs(dj)) / 2)
    corner1 = i1 + sdi*d_merge, j1 + sdj*d_merge
    corner2 = i2 - sdi*d_merge, j2 - sdj*d_merge
    corner1 = clamp(corner1, shape)
    corner2 = clamp(corner2, shape)
    rect = get_rectangle(corner1, corner2)
    return (d_merge, [pt for pt in rect if pt not in forbidden])


def merger(shape, units: Sequence[TUnit], dest:TUnit, influence=None, forbidden=set()) -> List[Action]:
    n_units = len(units)
    dest_coords = tuple(dest[:2])
    unit_coords = [tuple(unit[:2]) for unit in units]
    
    dists_to_dest = [dist(uc, dest_coords) for uc in unit_coords]
    max_dist = max(dists_to_dest)
    print("max dist", max_dist)
    forbidden_wo_dest = forbidden.difference({dest})
    # print(forbidden_wo_dest)
    sorted_pairs_dist = sorted(
        [(get_closest_merge_points(shape, unit_coords[i], unit_coords[j], forbidden_wo_dest), i, j)
         for j in range(n_units) for i in range(j) if i != j]
    )
    units_dest = dict()
    
    # Try pairing units to merge them if it does not delay the 
    for (d_merge, merge_pts), i, j in sorted_pairs_dist:
        if i not in units_dest and j not in units_dest:
            _offset = 0  # correction term if 
            if dest in merge_pts:
                _offset = 1
                merge_pts.remove(dest)
            dists_merge_pts_to_dest = [dist(p, dest_coords) for p in merge_pts]
            merge_min_dst, merge_min_idxs = min_argmin(dists_merge_pts_to_dest)
            if merge_min_dst + d_merge <= max_dist + _offset:
                candidates = [merge_pts[idx] for idx in merge_min_idxs]
                unit_dst = max_inluence_or_rd(candidates, influence)
                units_dest[i] = unit_dst
                units_dest[j] = unit_dst
        if len(units_dest) >= n_units - 1:
            break
    
    try:
        assert len(units_dest) >= n_units - 1
    except AssertionError:
        print("--err--")
        print(len(units_dest), n_units)
        print(units, dest, forbidden)
        raise
    actions = list()
    for u_idx, unit in enumerate(units):
        u_coords = unit[:2]
        u_num = unit[2]
        if u_idx in units_dest:
            u_dest = units_dest.get(u_idx, dest)
        else:
            u_dest = min([(dist(u_coords, mpt), mpt) for mpt in units_dest.values()])[1]
        if u_coords != u_dest:
            u_next = next_case(shape, u_coords, u_dest, influence, forbidden)
            actions.append(Action(*u_coords, *u_next, u_num))

    return actions


def goals_to_actions(
        allies: Sequence[TUnit],
        enemies: Sequence[TUnit],
        humans: Sequence[TUnit],
        inv_goal_humans: Sequence[Sequence[int]],
        inv_goal_enemies: Sequence[Sequence[int]],
        enemy_ptype = Game.Werewolf
    ) -> Sequence[Action]:

    ally_goals = len(allies) * [[]]

    for hum_idx, ally_idx_seq in enumerate(inv_goal_humans):
        for ally_idx in ally_idx_seq:
            ally_goals[ally_idx].append((Game.Human, hum_idx))

    # for ene_idx, ally_idx_seq in enumerate(inv_goal_enemies):
    #     for ally_idx in ally_idx_seq:
    #         ally_goals[ally_idx].append((enemy_ptype, hum_idx))

    actions = list()

    for human_idx, ally_idx_seq in enumerate(inv_goal_humans):
        if len(ally_idx_seq) == 1:
            pass


def plot_mat(mat) -> Tuple[plt.Axes, Any]:
    m, n = mat.shape[:2]
    plt.figure()
    im = plt.imshow(mat, aspect='equal')

    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, m, max(1, m//10)))
    ax.set_yticks(np.arange(0, n, max(1, m//10)))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, m + 1, max(1, m//10)))
    ax.set_yticklabels(np.arange(1, n + 1, max(1, m//10)))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    return ax, im


if __name__ == "__main__":
    # merger((20,20), [(12,5, 17), (3,6,20)], (8, 4), forbidden={(8, 4)})
    # exit()
    from ..game_gen import rd_coords, GamePlotter, GameGenerator

    m = 20
    n = 20
    gg = GameGenerator(m, n, vamp_spread=5, were_spread=4, human_pop=99)    
    g = gg()
    forbidden = set()
    forbidden.update((u[:2] for u in g.humans()))
    forbidden.update((u[:2] for u in g.humans()))
    for u in g.werewolves():
        forbidden.update(get_adj_case(u[:2], g.size()))
    print(forbidden)
    def next_game_state(game: Game) -> Game:
        influence = get_influence_map(g, Game.Vampire, kernel_type="square", kernel_size=41)
        dest = next(g.humans())[:2]
        units = list(g.vampires())
        if len(units) > 1:
            actions = merger(g.size(), units, dest, influence=influence, forbidden=forbidden)
        elif len(units) == 1:
            u_coords, u_num = units[0][:2], units[0][2]
            dist_dest = dist(u_coords, dest)
            print("dist", dist_dest)
            if dist_dest == 1:
                forbidden.remove(dest)
            actions = [Action(*u_coords, *next_case(g.size(), u_coords, dest, influence=influence, forbidden=forbidden), u_num)]
        g.register_actions(Game.Vampire, actions)
        return g

    delay = 0.5
    gp = GamePlotter(g, get_influence_map(g, Game.Vampire, kernel_type="square", kernel_size=41))
    for _ in range(100):
        sleep(delay)
        g = next_game_state(g)
        gp.update(g, get_influence_map(g, Game.Vampire, kernel_type="square", kernel_size=41))
        if g.human_pop() == 0:
            break
    exit()
    t0 = time()
    actions = merger(shape, units, dest)
    print(f"Merger in {time()-t0:.3f} s")
    print(actions)

    # dest_unit = defaultdict(list)
    # for u, d in unit_dest.items():
    #     dest_unit[d].append(u)

    # mat = np.zeros(shape + (3,))
    # mat[dest][1] = 255
    
    # for u in units:
    #     mat[u][0] = 255
    # for u in dest_unit:
    #     mat[u][2] = 255
    
    # ax, im = plot_mat(mat)
    # for cnt, (mpt, idxs) in enumerate(dest_unit.items()):
    #     ax.text(mpt[1], mpt[0], cnt, ha="center", va="center", color="w")
    #     pt = units[idxs[0]]
    #     ax.text(pt[1], pt[0], cnt, ha="center", va="center", color="w")
    #     pt = units[idxs[1]]
    #     ax.text(pt[1], pt[0], cnt, ha="center", va="center", color="w")
    
    # plt.show(block=True)
    
