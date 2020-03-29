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

class RUnitManager(SimpleRepr): 
    def __init__(self, unit_iterable: Iterable[TUnit]):
        self._id_gen_cnt = 0
        self.runits: Sequence[RUnit] = [RUnit(id=self._next_id(), coords=(i,j), num=num) for i,j,num in unit_iterable]
        self.cur_state_num = 0
        self.state_lenght = [0]
        self._actives = list(range(len(self.runits)))

    def create(self, coords:Coords, num:int, r:int=0):
        ru = RUnit(coords, num, r, created_at_state=self.cur_state_num)
        self.runits.append(ru)

    def __getitem__(self, idx:int):
        return self.runits[self._actives[idx]]

    def __iter__(self):
        return (self.runits[idx] for idx in self._actives)

    def _next_id(self) -> int:
        ret = self._id_gen_cnt
        self._id_gen_cnt += 1
        return ret

class RUnit(SimpleRepr):
    def __init__(self, id:int, coords:int, num:int, r:int=0, created_at_state:int=0, disappear_at_state:Optional[int]=None):
        self.id = id
        self.coords = coords
        self.num = num
        self.r = r
        self.created_at_state = created_at_state
    
    @classmethod
    def from_unit(cls, unit: TUnit, r=0) -> "RUnit":
        coords, num = unit[:2], unit[2]
        return cls(coords, num, r)

    def dist_unit(self, unit: TUnit) -> int:
        return max(0, dist(self.coords, unit[:2]) - self.r)


def next_step(rum: RUnitManager, humans: Sequence[TUnit]):
    # distance human-units
    humans = list(humans)
    dist_hu = [[ru.dist_unit(h) for ru in rum.actives()] for h in humans]

    hu_sorted = [
        list(sorted(((d, u) for u, d in enumerate(dist_hrow))))
        for dist_hrow in dist_hu
    ]

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

    ret =  min_argmin(h_steps)
    (dist, attackers), targets = ret
    if len(targets) > 1: print("multiple targets ", targets)
    target = targets[0]
    tval = humans[target]
  
    new_humans = [hval for h, hval in enumerate(humans) if h != target]
  
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