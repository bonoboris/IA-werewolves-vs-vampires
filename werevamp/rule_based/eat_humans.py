from typing import Sequence, Tuple, Set, Iterable, Dict, Union, List, Optional
from collections import defaultdict, namedtuple
from itertools import chain
from pprint import pprint
from math import inf
from functools import total_ordering
import copy as cp
from time import time
import random as rd

import numpy as np

from ..game import Game, Action
from ..runner import GameHistory
from ..plotting import ButtonGamePlotter
from ..utils import Coords, min_argmin, find, get_adj_case
from ..str_utils import SimpleRepr
from .simplepath import simple_path
from .joining_mat import LazyJoiningMat


TUnit = Tuple[int, int, int]


def dist(c1: Coords, c2: Coords) -> int:
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))


def dist_inf(c1: Coords, c2: Coords, val) -> int:
    return (c1[0] >= c2[0] - val and 
            c1[0] <= c2[0] + val and 
            c1[1] >= c2[1] - val and
            c1[1] <= c2[1] + val)


def closer(shape: Coords, from_case: Coords, to_case: Coords, forbidden=set()):
    adjs = [c for c in get_adj_case(shape, from_case) if c not in forbidden]
    if not adjs:
        # print(shape, from_case, to_case, forbidden)
        return [from_case]
    adjs_dist = [dist(c, to_case) for c in adjs]
    return [adjs[i] for i in min_argmin(adjs_dist)[1]]


def next_case(shape: Coords, from_case: Coords, to_case: Coords, influence = None, forbidden:Sequence[Coords] = set()):
    # print(f"next_case(shape={shape}, from_case={from_case}, to_case={to_case}, influence={influence}, forbidden={forbidden}")
    if from_case == to_case:
        return to_case
    return max_inluence_or_rd(closer(shape, from_case, to_case, forbidden), influence)


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


class RUnit(SimpleRepr):
    def __init__(self, coords, num, r=0):
        self.coords = coords
        self.num = num
        # self.variable = None
        self.r = r
    
    @classmethod
    def from_unit(cls, unit: TUnit, r=0) -> "RUnit":
        coords, num = unit[:2], unit[2]
        return cls(coords, num, r)

    def dist_other(self, other: "RUnit") -> int:
        return max(0, dist(self.coords, other.coords) - self.r - other.r)

    def dist_coords(self, coords: Coords) -> int:
        return max(0, dist(self.coords, coords) - self.r)

    def dist_unit(self, unit: TUnit) -> int:
        return max(0, dist(self.coords, unit[:2]) - self.r)


class Shared():
    def __init__(self, sum:int, unit_sups: Dict[RUnit, int]):
        self.sum = sum
        self.sups = unit_sups
        for u in unit_sups:
            u.variable = self

    def assign(self, unit: RUnit, num:int) -> None:
        if unit not in self.sups:
            raise KeyError(f"Unit not in the relation")
        elif num < self.num:
            raise ValueError(f"Requested {num} but only self.num = {self.num} available")
        elif num < self.sups[unit]:
            raise ValueError(f"Requested {num} but unit limit is {self.sups[unit]}")
        self.sups[unit] -= num
        self.sum -= num
    
    def max_assignable(self, *units: RUnit) -> int:
        return min(sum((self.sups[unit] for unit in units)), self.sum)



# def _eat(runits: Sequence[RUnit], human: TUnit, r):
#     human_coords = human[:2]
#     human_num = human[2]
#     sum_units_num = sum((u.num for u in runits))
#     if sum_units_num == human_num:
#         return [RUnit(human_coords, 2*human_num, 0)]
#     else:
#         new_units_sups = {}
#         for u in runits:
#             nu = RUnit(u.coords, max(0, human_num - u.num), r)
#             nu_sup = u.num - nu.num
#             new_units_sups[nu] = nu_sup
#         Shared(sum_units_num - human_num, new_units_sups)
#         new_units = list(new_units_sups)
#         new_units.append(RUnit(human_coords, 2*human_num, 0))
#         return new_units


def next_step(runits: Sequence[RUnit], humans: Sequence[TUnit], joining_mat: LazyJoiningMat, cur_step:int):
    # print("cur step", cur_step)
    # print("humans:")
    # pprint(humans)
    # print("runits:")
    # pprint(runits)
    # distance human-units
    runits = list(runits)
    humans = list(humans)
    dist_hu = [[u.dist_unit(h) for u in runits] for h in humans]
    # pprint(dist_hu)
    # mat len(humans) * len(runits):
    #   row [h]: sorted len(runits) * 2-tuple (dist(h, u), u) where u is a unit index 
    hu_sorted = [
        # list(sorted(((d, u) for u, d in enumerate(dist_hrow))))
        # for dist_hrow in dist_hu
    ]

    for h, dist_hrow in enumerate(dist_hu):
        hu_sorted.append(list(sorted(
            ((d, u) for u, d in enumerate(dist_hrow) if joining_mat.can_eat_in(d+cur_step, h) < humans[h][2])
        )))

    # list: len(humans) * 2-tuple: 
    h_steps = []
    ret_none = True
    for h, row in enumerate(hu_sorted):
        val = None
        human_num = humans[h][2]
        cands = []
        cands_num = 0
        for (d, u) in row:
            cands.append(u)
            cands_num += runits[u].num
            if cands_num >= human_num:
                val = (d, cands)
                ret_none = False
                break
        h_steps.append(val)
    # print("h_steps")
    # print("*uneatable ", [humans[h] for h, val in enumerate(h_steps) if val is None])
    # pprint(h_steps)
    if ret_none:
        return None
    dist, attackers, target = inf, None, None
    # pprint(h_steps)
    for h, val in enumerate(h_steps):
        if val is not None:
            (d, att) = val
            if d < dist:
                dist = d
                attackers = att
                target = h

    # print(f"target {target} | attackers {attackers} | dist {dist}")
    tval = humans[target]
  
    new_humans = [hval for h, hval in enumerate(humans) if h != target]
  
    new_runits = []
    for u, uval in enumerate(runits):
        if u not in attackers:
            cop = cp.copy(uval)
            cop.r += dist
            new_runits.append(cop)
    # add remainder
    attackers_num = sum((runits[u].num for u in attackers))
    # print(f"target num {tval[2]} attackers num {attackers_num}")
    rem_num = attackers_num - tval[2]
    # print("has rem", rem_num > 0)
    if rem_num > 0:
        cop = cp.copy(runits[attackers[-1]])
        cop.r = dist - 1
        cop.num = rem_num
        new_runits.append(cop)
    # add new converted human unit
    new_runits.append(RUnit(tval[:2], 2*tval[2]))
    # print("new_humans")
    # pprint(new_humans)
    # print("new_runits")
    # pprint(new_runits)

    for u in attackers:
        try:
            assert runits[u].dist_unit(tval) <= dist
        except AssertionError:
            print("state")
            pprint(runits)
            pprint(humans)
            print("hu_sorted")
            pprint(hu_sorted)
            print("h_steps")
            pprint(h_steps)
            print("ret min_argmin & dist")
            pprint(dist)
            print("unit too far and human")
            pprint(runits[u])
            pprint(tval)
            raise
    return (new_runits, new_humans), (target, dist, attackers)


def make_rgame(shape, humans, enemy_faction, enemies):
    g = Game(*shape)
    for i,j,num in humans:
        g[i,j] = Game.Human, num
    for i,j,num in enemies:
        g[i,j] = enemy_faction, num
    return g


def glouton(game: Game, faction:int):
    """Increasingly find best eating strategy"""
    runits = [RUnit.from_unit(u) for u in game.units(faction)]
    humans = list(game.humans())

    state = (runits, humans)
    states = [state]
    actions = []
    sum_step = 0
    enemy_faction = Game.enemy_faction(faction)
    enemies = list(game.units(enemy_faction))
    it = 1
    while state[1]:
        # print("* Iter", it)
        rgame = make_rgame(game.size(), state[1], enemy_faction, enemies)
        joining_mat = LazyJoiningMat(rgame, enemy_faction)
        ret = next_step(*state, joining_mat=joining_mat, cur_step=sum_step)
        if ret is None:
            # print("uneatable humans ", state[1])
            break
        state, action = ret
        sum_step += action[1]
        states.append(state)
        actions.append(action)
        it +=1
    
    # pprint(states[0])
    # for a, s in zip(actions, states[1:]):
    #     print("action")
    #     pprint(a)
    #     print("new state")
    #     pprint(s)
    return states, actions


def find_rems(states, actions):
    n = len(actions)
    has_rems = []
    for i in range(n):
        if len(states[i][0]) - len(actions[i][2]) + 1 < len(states[i+1][0]):
            has_rems.append(i+1)
    return has_rems



def transfo(shape, states, actions, forbidden=set(), influence=None):
    state_has_rem = find_rems(states, actions)
    id_cnt = 0
    coords_id = dict()
    id_runit = dict()
    state_runit_ids = list()
    appear_state = dict()
    disappear_state = dict()
    dests = dict()
    rem_parents = dict()
    for i, (runits, _) in enumerate(states):
        if i in state_has_rem:
            rem_parents[id_cnt] = coords_id[runits[-2].coords]
            del coords_id[runits[-2].coords]
        runits_id = []
        for ru in runits:
            if ru.coords in coords_id:
                id_ = coords_id[ru.coords]
            else:
                id_ = id_cnt
                id_cnt += 1
                coords_id[ru.coords] = id_
                id_runit[id_] = ru
                appear_state[id_] = i
            runits_id.append(id_)
        if i > 0:
            dests[i] = runits[-1].coords
            for id_ in set(state_runit_ids[-1]).difference(runits_id):
                disappear_state[id_] = i
        state_runit_ids.append(runits_id)

    rem_to_remove = set(rem_parents).union(set(disappear_state))
    (id_runit, state_runit_ids, appear_state,
    disappear_state, rem_parents, state_has_rem) = filter_rem(id_runit, state_runit_ids, appear_state,
                                                              disappear_state, rem_parents, state_has_rem)

    state_steps = [0]
    for _, steps, _ in actions:
        state_steps.append(state_steps[-1] + steps)

    for i_state, (runits, _) in enumerate(states):
        for ru in runits:
            assert ru.r <= state_steps[i]

    # print(state_steps)
    num_steps = state_steps[-1] + 1
    # print("num_steps", num_steps)
    traces = dict()
    moving = dict()
    merge_dest = median_point([id_runit[id_].coords for id_ in state_runit_ids[0]])
    _forbidden = set(forbidden)
    for id_ in sorted(id_runit):
        appear_step = state_steps[appear_state[id_]]
        if id_ in rem_parents:
            if rem_parents[id_] not in traces:
                continue
            else:
                trace_par = traces[rem_parents[id_]]
                if trace_par[0] is None or len(trace_par) == 2:
                    continue
                appear_step = 1
        if appear_step > 1:
            continue
        trace = [None] * appear_step
        start =  id_runit[id_].coords
        num = id_runit[id_].num
        if appear_step == 1:
            # print(f"id {id_} appears")
            trace = [None, (start, num)]
        else:
            trace = [(start, num)]
            if id_ not in disappear_state: # move toward merge dest 
                # print(f"id {id_} merging toward {merge_dest}")
                if start == merge_dest:
                    trace.append((start, num))
                else:
                    moving[id_] = (start, merge_dest, num)
                    _forbidden.add(start)
            else:
                dest = dests[disappear_state[id_]]
                disappear_step = state_steps[disappear_state[id_]]
                num_start_steps = disappear_step - appear_step - dist(start, dest) + 1
                if num_start_steps < 1:
                    print(f"Error ? num_start_steps={num_start_steps}")
                elif num_start_steps > 1:
                    # print(f"id {id_} moving later")
                    trace.append((start, num))
                else:
                    # print(f"id {id_} moving now")
                    move_num = num
                    for id_rem, id_parent in rem_parents.items():
                        if id_ == id_parent:
                            move_num -= id_runit[id_rem].num
                    adj_case = get_adj_case(shape, start)
                    moving[id_] = (start, dest, move_num)
                    _forbidden.add(start)
        traces[id_] = trace
    for id_, (start, dest, move_num) in moving.items():
        ncase = next_case(shape, start, dest, influence, _forbidden)
        if ncase == start:
            print(shape, start, dest, [id_runit[id_].coords for id_ in state_runit_ids[0]])
        traces[id_].append((ncase, move_num))

    # print(traces)
    unit_eaten = {id_:state_runit_ids[state_num][-1] for id_, state_num in disappear_state.items()}
    return traces_to_actions(traces, unit_eaten, rem_parents)
    # yield from traces_to_game_gen(shape, states, state_steps, traces)


def max_inlf_coords(influence: np.ndarray):
    m,n = influence.shape
    midx = np.argmax(influence)
    mi, mj = midx // n, midx % n
    return mi, mj

def median_point(coords: Coords):
    return tuple(map(round, np.median(coords, axis=0).tolist()))


def filter_rem(id_runit, state_runit_ids, appear_state, disappear_state, rem_parents, state_has_rem):
    rem_to_remove = {id_ for id_ in rem_parents if id_ not in disappear_state}

    new_state_runit_ids = []
    new_id_runit = {id_: v for id_, v in id_runit.items() if id_ not in rem_to_remove}
    new_state_has_rem = cp.deepcopy(state_has_rem)
    new_appear_state = {id_: v for id_, v in appear_state.items() if id_ not in rem_to_remove}
    new_disappear_state = {id_: v for id_, v in disappear_state.items() if id_ not in rem_to_remove}
    new_rem_parents = {id_: v for id_, v in rem_parents.items() if id_ not in rem_to_remove}

    # print(state_has_rem)

    to_increase = defaultdict(int)
    for rem in rem_to_remove:
        rem_num = id_runit[rem].num
        state_num = appear_state[rem]
        # print(state_num)
        new_state_has_rem.remove(state_num)
        while True:
            combine_to = state_runit_ids[state_num][-1]
            to_increase[combine_to] += rem_num
            if combine_to in disappear_state:
                state_num = disappear_state[combine_to]
            else:
                break

    
    for id_, inc in to_increase.items():
        new_id_runit[id_].num += inc

    new_state_runit_ids = [[id_ for id_ in row if id_ not in rem_to_remove] for row in state_runit_ids]

    return new_id_runit, new_state_runit_ids, new_appear_state, new_disappear_state, new_rem_parents, new_state_has_rem


def traces_to_actions(traces, unit_eaten, rem_parents):
    actions = list()
    for id_, trace in traces.items():
        if trace[0] is not None:
            start_coords, start_num = trace[0]
            if trace[1] is not None:
                dest_coords, dest_num = trace[1]
                if dest_coords != start_coords:
                    actions.append(Action(*start_coords, *dest_coords, dest_num))
            else:
                dest_coords = traces[unit_eaten[id_]][1][0]
                child_num = 0
                for rem, parent in rem_parents.items():
                    if parent == id_:
                        if traces[rem][0] is None and traces[rem][1] is not None:
                            child_num = traces[rem][1][1]
                    
                actions.append(Action(*trace[0][0], *dest_coords, start_num - child_num))
    return actions



def traces_to_game_gen(shape, states, state_steps, traces):
    state_steps.append(inf)
    for s in range(state_steps[-2] + 1):
        gs = Game(*shape)
        ns = find(state_steps, lambda el: el > s)
        if ns > 0:
            for i,j,num in states[ns - 1][1]:
                if gs[i,j] is None:
                    gs[i,j] = Game.Human, num
                else:
                    raise ValueError("Override")

        for trace in traces.values():
            if trace[s] is not None:
                coords, num = trace[s]
                gs.add_create(coords, (Game.Vampire, num))
        yield gs


if __name__ == "__main__":
    from ..game_gen import GameGenerator
    gg = GameGenerator(30,30, vamp_spread=5, human_spread=5, human_pop=100)
    g = gg()
    g[0,0] = Game.Human, 1000
    t0 = time()
    states, actions = glouton(g, Game.Vampire)

    print(transfo(g.size(), states, actions))
    # gr = GameHistory(transfo(g.size(), states, actions))
    # t1 = time()
    # print(f"Elapsed time: {t1-t0:.5f} s")
    # ButtonGamePlotter(gr)
