from typing import NamedTuple, Tuple, List, Optional, Sequence, Dict, Generator
import heapq
from time import time
from collections import defaultdict
from queue import SimpleQueue

import numpy as np
import matplotlib as plt

from ..game import Game
from ..utils import Coords, get_adj_case, clamp, valid_coords, sign, add_coords, scale_coords, srange
from ..str_utils import SimpleRepr
from ..runner import BaseRunner
from ..plotting import ButtonGamePlotter, set_text
from .. utils import sub_coords
from .joining_mat import LazyJoiningMat


def dist(c1: Coords, c2: Coords) -> int:
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))


def simple_path(start: Coords, dest:Coords):
    path = list()
    dif = sub_coords(dest, start)
    udif = (sign(dif[0]), sign(dif[1]))
    k = min(abs(dif[0]), abs(dif[1]))
    for i in range(k):
        path.append(add_coords(start, scale_coords(k+1, udif)))
    m = path[-1]
    if m[0] == dest[0]:
        for j in srange(m[1], dest[1], True, False):
            path.append((dest[0], j))
    elif m[1] == dest[1]:
        for i in srange(m[0], dest[0], True, False):
            path.append((i, dest[1]))
    return path


class Node(SimpleRepr):
    def __init__(self, f: Optional[int] = None, g: Optional[int] = None, coords: Optional[Coords] = None, parent: Optional["Node"] = None):
        super().__init__(exclude={"parent"})
        self.f = f
        self.g = g
        self.coords = coords
        self.parent = parent

    def find_coords_in(self, seq_nodes: Sequence["Node"]) -> int:
        for i, node in enumerate(seq_nodes):
            if self.coords == node.coords:
                return i
        return -1

    def __lt__(self, other: "Node") -> bool:
        return self.f < other.f


class PathFinder:
    def __init__(self, game: Game, ally_faction: int) -> None:
        self.game = game
        self.ally_faction = ally_faction
        self.enemy_faction = Game.enemy_faction(ally_faction)
        self.enemy_mats = LazyJoiningMat(self.game, self.enemy_faction)
        self.human_mat = np.zeros(self.game.size(), dtype=int)
        for i, j, num in self.game.humans():
            self.human_mat[i, j] = num

    def __call__(self, start: Coords, num: int, dest: Coords) -> List[Coords]:
        opened: List[Node] = [Node(dist(start, dest), 0, start, None)]
        closed = dict()
        while len(opened) > 0:
            cur = heapq.heappop(opened)
            if cur.coords == dest:
                return self.construct_path(cur)
            closed[cur.coords] = cur.g

            reheap = False
            for adj in self.neighbors(cur, num, dest):
                if adj.coords in closed:
                    if adj.g < closed[adj.coords]:
                        print("recomp")
                        del closed[adj.coords]
                    else:
                        continue
                adj_idx = adj.find_coords_in(opened)
                if adj_idx > -1:
                    if opened[adj_idx].g > adj.g:
                        dif = opened[adj_idx].g - adj.g
                        opened[adj_idx].g -= dif
                        opened[adj_idx].f -= dif
                        opened[adj_idx].parent = adj.parent
                        reheap = True
                else:
                    adj.f = adj.g + dist(adj.coords, dest)
                    opened.append(adj)
                    reheap = True
            if reheap:
                heapq.heapify(opened)
        return list()

    def iter_mat(self, start: Coords, num: int, dest: Coords) -> List[Coords]:
        opened: List[Node] = [Node(dist(start, dest), 0, start, None)]
        closed = dict()
        while len(opened) > 0:
            cur = heapq.heappop(opened)
            yield self.pfmat(cur, opened, closed, num)
            if cur.coords == dest:
                return self.construct_path(cur)
            closed[cur.coords] = cur.g

            reheap = False
            for adj in self.neighbors(cur, num, dest):
                if adj.coords in closed:
                    if adj.g < closed[adj.coords]:
                        print("recomp")
                        del closed[adj.coords]
                    else:
                        continue
                adj_idx = adj.find_coords_in(opened)
                if adj_idx > -1:
                    if opened[adj_idx].g > adj.g:
                        dif = opened[adj_idx].g - adj.g
                        opened[adj_idx].g -= dif
                        opened[adj_idx].f -= dif
                        opened[adj_idx].parent = adj.parent
                        reheap = True
                else:
                    adj.f = adj.g + dist(adj.coords, dest)
                    opened.append(adj)
                    reheap = True
            if reheap:
                heapq.heapify(opened)
            else:
                print("no reheap")
        return list()
        

    def pfmat(self, cur: Node, opened: List[Node], closed: Dict[Coords, int], num:int) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        im_mat = np.zeros(game.size() + (3,), dtype=int)
        mat = self.enemy_mats.step_mat(num)
        
        text_list = []
        for i in range(game.m):
            for j in range(game.n):
                if mat[i,j] > 0:
                    im_mat[i,j] = [round(125*(mat[i,j] + 1) / max(game.m, game.n)), 0, 0]
                    text_list.append((i,j, mat[i,j]))
                elif mat[i,j] == 0:
                    im_mat[i,j] = [0, 50, 0]

        for onode in opened:
            im_mat[onode.coords] = [0, 0, 150]
            text_list.append((*onode.coords, onode.f))

        for coords, g in closed.items():
            im_mat[coords] = [0, 0, 100]
            text_list.append((*coords, g))
        
        im_mat[cur.coords] = [100,100,100]
        text_list.append((*cur.coords, cur.g))
        return im_mat, text_list

    @staticmethod
    def construct_path(node: Node) -> List[Coords]:
        rpath = [node.coords]
        cur = node
        while cur.parent is not None:
            rpath.append(cur.parent.coords)
            cur = cur.parent
        return list(reversed(rpath))

    def neighbors(self, parent: Node, num, dest: Coords) -> List[Node]:
        neighs_coords = get_adj_case(self.game.size(), parent.coords)
        g_neighs = parent.g + 1
        neighs_nodes = list()
        for c in neighs_coords:
            if not self.is_obstacle(g_neighs, c, num):
                neighs_nodes.append(Node(g=g_neighs, coords=c, parent=parent))
        return neighs_nodes

    def is_obstacle(self, g: int, coords: Coords, num:int) -> bool:
        if self.human_mat[coords] > num:
            return True
        elif 1.5 * self.enemy_mats[(g,) + coords] > num:
            return True
        return False


class PFRunner(BaseRunner):
    def __init__(self, game, unit_coord, dest):
        super().__init__()
        faction, num = game[unit_coord]
        self.pf = PathFinder(game, faction)
        # self.gen = self.pf.iter_mat(start=unit_coord, num=num, dest=dest)
        self.gen = self.pf.explore_it(start=unit_coord, num=num)

    def __call__(self):
        try:
            return next(self.gen) + (self.pf.game,)
        except StopIteration:
            return None

class PFPlotter(ButtonGamePlotter):
    def update(self, pfrunner_ret):
        if pfrunner_ret is None:
            return
        mat, text_list, game = pfrunner_ret
        self._update_data(mat)
        self.ax.texts.clear()
        set_text(self.ax, text_list)
        set_text(self.ax, game.humans(), bbox_col="b")
        set_text(self.ax, game.vampires(), bbox_col="g")
        set_text(self.ax, game.werewolves(), bbox_col="r")
        plt.draw()


if __name__ == "__main__":
    from ..game_gen import GameGenerator
    from matplotlib import pyplot as plt
    from ..plotting import set_text, set_grid, ButtonGamePlotter
    from ..runner import BaseRunner
    from itertools import chain
    from time import time
    import random as rd

    m=30
    n=30
    gg = GameGenerator(m=m, n=n, vamp_spread=1, were_spread=5, human_spread=4, were_pop=200, human_pop=1000, vamp_pop=50)

    # game = gg()
    # unit_coord = next(game.vampires())[:2]
    # dest = (0, 0)
    # pfr = PFRunner(game, unit_coord, dest)
    # pfp = PFPlotter(pfr)
    # exit()

    class Runnner(BaseRunner):
        def __call__(self, game=None):
            game = gg()
            t0 = time()
            pf = PathFinder(game, Game.Vampire)
            i,j,num = next(game.vampires())
            bound = pf.explore((i,j), num)
            t1 = time()
            print(f"Time spent: {t1-t0} s")  # : dist = {max(i,j)} / len path = {len(path) - 1}
            # free8 = pf.find_8way_free((i,j), num)
            mat = pf.enemy_mats.step_mat(num)
            # for p in free8:
            #     mat[p] = -2
            return game, list(), mat


    class Plotter(ButtonGamePlotter):
        def update(self, runner_ret):
            self.influence_mode = "None"
            game, path, mat = runner_ret
            self.ax.texts.clear()
            im_mat = np.zeros(game.size() + (3,), dtype=int)
            for i in range(game.m):
                for j in range(game.n):
                    if mat[i,j] > 0:
                        im_mat[i,j] = [round(125*(mat[i,j] + 1) / max(game.m, game.n)), 0, 0]
                        self.ax.text(j,i, mat[i,j], ha="center", va="center", color="w")
                    elif mat[i,j] == 0:
                        im_mat[i,j] = [0, 50, 0]
                    elif mat[i,j] == -2:
                        im_mat[i,j] = [0, 200, 0]
            for coords in path:
                im_mat[coords] = [255, 255, 255]
                        
            self._update_data(im_mat)
            set_text(self.ax, ((i, j, k+1) for k,(i,j) in enumerate(path[1:])), textcol="k")
            set_text(self.ax, game.humans(), bbox_col='b')
            set_text(self.ax, game.vampires(), bbox_col='g')
            set_text(self.ax, game.werewolves(), bbox_col='r')
            plt.draw()

    Plotter(Runnner())

    # for i in range(1000):
    #     if (i+1) % 10 == 0:
    #         print(f"Iteration {i+1}")
    #     game = gg()
    #     dest = (rd.randint(0, m- 1), rd.randint(0, n-1))
    #     unit = next(game.vampires())
    #     start, num = unit[:2], unit[2]
    #     pf = PathFinder(game, Game.Vampire)
    #     path = pf(start, num, dest)
    #     if len(path) != 0 and len(path) - 1 != dist(start, dest):
    #         print(f"len path {len(path) - 1} dist {dist(start, dest)}")

