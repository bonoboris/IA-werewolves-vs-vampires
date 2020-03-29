from typing import Sequence, Tuple, Set, Iterable, Dict, Union, List, Optional
from collections import defaultdict, namedtuple
from itertools import chain
from pprint import pprint
from math import inf
from functools import total_ordering
import copy as cp

from ..game import Game
from ..runner import GameHistory
from ..plotting import ButtonGamePlotter
from ..utils import Coords, min_argmin, find
from ..str_utils import SimpleRepr
from .simplepath import simple_path


TUnit = Tuple[int, int, int]


def dist(c1: Coords, c2: Coords) -> int:
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))


def dist_inf(c1: Coords, c2: Coords, val) -> int:
    return (c1[0] >= c2[0] - val and 
            c1[0] <= c2[0] + val and 
            c1[1] >= c2[1] - val and
            c1[1] <= c2[1] + val)


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



def _eat(runits: Sequence[RUnit], human: TUnit, r):
    human_coords = human[:2]
    human_num = human[2]
    sum_units_num = sum((u.num for u in runits))
    if sum_units_num == human_num:
        return [RUnit(human_coords, 2*human_num, 0)]
    else:
        new_units_sups = {}
        for u in runits:
            nu = RUnit(u.coords, max(0, human_num - u.num), r)
            nu_sup = u.num - nu.num
            new_units_sups[nu] = nu_sup
        Shared(sum_units_num - human_num, new_units_sups)
        new_units = list(new_units_sups)
        new_units.append(RUnit(human_coords, 2*human_num, 0))
        return new_units


def next_step(runits: Sequence[RUnit], humans: Sequence[TUnit]):
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
        list(sorted(((d, u) for u, d in enumerate(dist_hrow))))
        for dist_hrow in dist_hu
    ]
    # pprint(hu_sorted)
    # list: len(humans) * 2-tuple: 
    h_steps = []
    for h, row in enumerate(hu_sorted):
        human_num = humans[h][2]
        cands = []
        cands_num = 0
        for (d, u) in row:
            cands.append(u)
            cands_num += runits[u].num
            if cands_num >= human_num:
                h_steps.append((d, cands))
                break
    # print("h_steps")
    # pprint(h_steps)
    ret =  min_argmin(h_steps)
    (dist, attackers), targets = ret
    if len(targets) > 1: print("multiple targets ", targets)
    target = targets[0]
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
            print("h_steps")
            pprint(h_steps)
            print("ret min_argmin & dist")
            pprint(ret)
            pprint(dist)
            print("unit too far and human")
            pprint(runits[u])
            pprint(tval)
            raise
    return (new_runits, new_humans), (target, dist, attackers)


def glouton(units: Sequence[TUnit], humans: Sequence[TUnit]):
    """Increasingly find best eating strategy"""
    runits = [RUnit.from_unit(u) for u in units]

    state = (runits, humans)
    states = [state]
    actions = []

    while state[1]:
        state, action = next_step(*state)
        states.append(state)
        actions.append(action)
    
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



def transfo_ids(states, actions):
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


def transfo(shape, states, actions):
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

    state_steps = [0]
    for _, steps, _ in actions:
        state_steps.append(state_steps[-1] + steps)

    for i_state, (runits, _) in enumerate(states):
        for ru in runits:
            assert ru.r <= state_steps[i]

    # print(state_steps)
    num_steps = state_steps[-1] + 1
    # print("num_steps", num_steps)
    traces = []
    for id_ in range(id_cnt):
        appear_step = state_steps[appear_state[id_]]
        if id_ in rem_parents:
            trace_par = traces[rem_parents[id_]]
            start_par = find(trace_par, lambda el: el is not None)
            # print("*", start_par)
            appear_step = start_par + find(trace_par[start_par: ], lambda el: el != trace_par[start_par])
            # print("**", appear_step)
        trace = [None] * appear_step

        start =  id_runit[id_].coords
        dest = None
        num = id_runit[id_].num
        path_num = num
        for id_rem, id_par in rem_parents.items():
            if id_ == id_par:
                path_num -= id_runit[id_rem].num
        # print("start", start)
        if id_ not in disappear_state: # stay in place
            trace += [(start, num)] * (num_steps - len(trace))
        else:
            dest = dests[disappear_state[id_]]
            # print("dest", dest)
            disappear_step = state_steps[disappear_state[id_]]
            path = simple_path(start, dest)
            waiting_steps = disappear_step - appear_step - len(path) + 1
            if waiting_steps < 1:
                print(f"Error ? waiting_steps={waiting_steps}")
            trace += [(start, num)] * waiting_steps
            trace.extend(((coord, path_num) for coord in path[:-1]))
            trace += [None] * (num_steps - len(trace))
        if len(trace) != num_steps:
            print(f"len trace = {len(trace)} expected {num_steps}")
            print(f"is_rem {id_ in rem_parents} appear_state {appear_state[id_]}, disappear_state {disappear_state.get(id_, None)}, start {start}, dest {dest}")
            print(trace)
        traces.append(trace)
    # pprint(state_runit_ids)
    # pprint(state_steps)
    # pprint(appear_state)
    # pprint(disappear_state)
    # pprint(rem_parents)
    
    # pprint(traces)

    for s in range(state_steps[-1] + 1):
        gs = Game(*shape)
        ns = find(state_steps, lambda el: el > s)
        if ns > 0:
            for i,j,num in states[ns - 1][1]:
                if gs[i,j] is None:
                    gs[i,j] = Game.Human, num
                else:
                    raise ValueError("Override")

        for trace in traces:
            if trace[s] is not None:
                coords, num = trace[s]
                gs.add_create(coords, (Game.Vampire, num))
                # if gs[coords] is None:
                #     gs[coords] = Game.Vampire, num
                # else:
                #     print("overide")
                #     print(s)
                #     print(coords, num)
                #     print(gs[coords])
                #     gs[coords] = Game.Vampire, num
                # #     raise ValueError("Override")
        yield gs




def transform_states_actions_to_game_(states, actions):
    class UnitPath(SimpleRepr):
        # __slots__= ("start", "step", "dest", "num")
        def __init__(self, start:Coords=None, step:int=None, dest:Coords=None, num:int=None):
            self.start = start
            self.step = step
            self.dest = dest
            self.num = num
        
        @staticmethod
        def find_with_coords(coords, seq: Iterable["UnitPath"]) -> "UnitPath":
            for el in seq:
                if el.start == coords:
                    return el
            return None

    unit_paths: Set[UnitPath] = set() 
    states_with_rems = find_rems(states, actions)
    cur_step = 0
    for idx in range(len(actions)):
        state = states[idx]
        runits, humans = state
        next_state = states[idx+1]
        next_runits, _ = next_state
        action = actions[idx]
        target_idx, steps, attackers_indices = action
        dest = humans[target_idx]
        # set attackers path
        print('num new path', len(attackers_indices))
        for runit in (runits[u] for u in attackers_indices):
            unit_paths.add(UnitPath(
                start=runit.coords,
                step=cur_step - runit.r,
                dest=dest,
                num=runit.num
            ))
        cur_step += steps
        # handle reminder unit if any
        has_rem = bool(len(next_runits) - len(runits) - len(attackers_indices) + 1)
        if has_rem:
            print("has_rem", has_rem)
            rem_runit = next_runits[-2]
            # correct num
            att_up = UnitPath.find_with_coords(rem_runit.coords, unit_paths)
            att_up.num -= rem_runit.num
        
        # add unit path for converted humans unit
        pprint(unit_paths)
        print()


if __name__ == "__main__":
    from ..game_gen import GameGenerator
    gg = GameGenerator(30,30, vamp_spread=5, human_spread=5)
    g = gg()
    states, actions = glouton(list(g.vampires()), list(g.humans()))
    # print(find_rem(states, actions))
    gr = GameHistory(transfo(g.size(), states, actions))
    ButtonGamePlotter(gr)
    # transform_states_actions_to_game_(states, actions)
