from typing import Sequence, List, Generator, Optional, Tuple, Set
from time import time

import numpy as np
from matplotlib import pyplot as plt

from ..game import Game
from ..utils import Coords, valid_coords, add_coords
from ..plotting import TextAnnot, ButtonGamePlotter, set_text
from ..runner import BaseRunner
from .joining_mat import LazyJoiningMat


def dist(c1: Coords, c2: Coords) -> int:
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))


class Explorer:
    def __init__(self, game: Game, ally_faction: int) -> None:
        self.game = game
        self.ally_faction = ally_faction
        self.enemy_faction = Game.enemy_faction(ally_faction)
        self.enemy_mats = LazyJoiningMat(self.game, self.enemy_faction)
        self.human_mat = np.zeros(self.game.size(), dtype=int)
        for i, j, num in self.game.humans():
            self.human_mat[i, j] = num

    def find_free_exp(self, start, num, dir_coords):
        i,j = start
        rstart = self.game.m - i - 1, self.game.n - j - 1
        def _k_lim(_val, _idx): return {-1: start[_idx], 0: np.inf, 1: rstart[_idx]}[_val]
        k_lim = min(_k_lim(dir_coords[0], 0), _k_lim(dir_coords[1], 1))
        def k_to_coord(k):
            return i + dir_coords[0] * k, j+dir_coords[1] * k

        if k_lim == 0 or not self.is_obstacle(k_lim, k_to_coord(k_lim), num):
            return k_to_coord(k_lim)
        else:
            free_k, obs_k = 0, k_lim
            while obs_k - free_k > 1:
                mid_k = (free_k + obs_k)//2
                if self.is_obstacle(mid_k, k_to_coord(mid_k), num):
                    obs_k = mid_k
                else:
                    free_k = mid_k
            return k_to_coord(free_k)

    def find_leftmost_free(self, start: Coords, num: int) -> Coords:
        i, j = start
        if j == 0 or not self.is_obstacle(j, (i, 0), num):
            return (i, 0)
        else:
            free_j, obs_j = j, 0
            while free_j - obs_j > 1:
                mid = (obs_j + free_j)//2
                if self.is_obstacle(j - mid, (i, mid), num):
                    obs_j = mid
                else:
                    free_j = mid
            return i, free_j

    def find_8way_free(self, start, num) -> Sequence[Coords]:
        def find_free(k_lim, k_to_coord):
            if k_lim == 0 or not self.is_obstacle(k_lim, k_to_coord(k_lim), num):
                return k_to_coord(k_lim)
            else:
                free_k, obs_k = 0, k_lim
                while obs_k - free_k > 1:
                    mid_k = (free_k + obs_k)//2
                    if self.is_obstacle(mid_k, k_to_coord(mid_k), num):
                        obs_k = mid_k
                    else:
                        free_k = mid_k
                return k_to_coord(free_k)

        i,j = start
        ri, rj = self.game.m - i - 1, self.game.n - j - 1
        return [
            find_free(j,            lambda k: (i, j-k)),    # left
            find_free(rj,           lambda k: (i, j+k)),    # right
            find_free(i,            lambda k: (i-k, j)),    # up
            find_free(ri,           lambda k: (i+k, j)),      # down
            find_free(min(i, j),    lambda k: (i-k, j-k)),  # up left
            find_free(min(i, rj),   lambda k: (i-k, j+k)),  # up right
            find_free(min(ri, rj),  lambda k: (i+k, j+k)),  # down right
            find_free(min(ri, j),   lambda k: (i+k, j-k)),  # down left
        ]

    def find_diag_non_diag_free(self, start, num) -> Tuple[Set[Coords], Set[Coords]]:
        return {
            self.find_free_exp(start, num, (-1,-1)),
            self.find_free_exp(start, num, (1,-1)),
            self.find_free_exp(start, num, (-1,1)),
            self.find_free_exp(start, num, (1,1)),
        }

    def explore(self, start: Coords, num:int) -> List[Coords]:
        p_coords = [
            (0,-1), (-1,-1), (-1,0), (-1, 1),
            (0,1), (1,1), (1,0), (1,-1)
        ]
        def add_coord(c1, c2):
            return c1[0] + c2[0], c1[1] + c2[1]
        def next_p(p):
            return (p + 1) % 8
        def new_p(p):
            return (p - (p%2) - 1) % 8

        bound_start = self.find_leftmost_free(start, num)
        bound = [bound_start]
        p = 1
        cur = bound_start 
        while True:
            for _ in range(8):
                cand = add_coord(cur, p_coords[p])
                if  valid_coords(self.game.size(), cand) and not self.is_obstacle(dist(cand, start), cand, num):
                    cur = cand
                    p = new_p(p)
                    break
                else:
                    p = next_p(p)
            if cur == bound_start: break
            else: bound.append(cur)
        return bound

    def explore_it(self, start: Coords, num: int) -> Generator[Tuple[np.ndarray, TextAnnot], None, List[Coords]]:
        p_coords = [
            (0,-1), (-1,-1), (-1,0), (-1, 1),
            (0,1), (1,1), (1,0), (1,-1)
        ]
        def add_coord(c1, c2):
            return c1[0] + c2[0], c1[1] + c2[1]
        def next_p(p):
            return (p + 1) % 8
        def new_p(p):
            return (p - (p%2) - 1) % 8

        bound_start = self.find_leftmost_free(start, num)
        bound = [bound_start]
        p = 1
        cur = bound_start
        while True:
            yield self.explore_to_mat(bound_start, num, cur)
            for _ in range(8):
                cand = add_coord(cur, p_coords[p])
                if valid_coords(self.game.size(), cand):
                    yield self.explore_to_mat(bound_start, num, cur, cand)
                    if not self.is_obstacle(dist(cand, start), cand, num):
                        cur = cand
                        p = new_p(p)
                        break
                    else:
                        p = next_p(p)
                        yield self.explore_to_mat(bound_start, num, cur)
                else:
                    print("invalid cand", cand)
                    p = next_p(p)
            if cur == bound_start:
                yield self.explore_to_mat(bound_start, num, bound_start)
                break
            else: bound.append(cur)
        return bound

    def explore_to_mat(self, bound_start:Coords, num:int, cur: Coords, cand: Optional[Coords]=None) -> Tuple[np.ndarray, List[TextAnnot]]:
        im_mat, text_list = self.enemy_mats.step_im_mat(num)

        im_mat[bound_start] = [0, 0, 100]
        im_mat[cur] = [0, 150, 0]
        if cand and valid_coords(self.game.size(), cand):
            im_mat[cand] = [0, 100, 0]
        return im_mat, text_list

    def is_obstacle(self, g: int, coords: Coords, num: int) -> bool:
        unit = self.game[coords]
        if unit and unit[0] == Game.Human and unit[1] > num:
            return True
        else:
            return 1.5 * self.enemy_mats[(g,) + coords] > num

    # def fill(self, start, bound) -> Set[Coords]: NOT WORKING
    #     print("filling")
    #     filled = set(bound)
    #     print(filled)
    #     def add_line(s, dir_coords):
    #         n = add_coords(s, dir_coords)
    #         while n not in filled:
    #             print(n)
    #             filled.add(n)
    #             n = add_coords(n, dir_coords)

    #     add_line(start, (1,0))
    #     add_line(start, (-1,0))
    #     add_line(start, (0,1))
    #     add_line(start, (0,-1))

    #     def add_diag_line(s, diag_dir):
    #         ndiag = add_coords(s, diag_dir)
    #         d1 = (diag_dir[0], 0)
    #         d2 = (0, diag_dir[1])
    #         while ndiag not in filled or add_coords(ndiag, d1) not in filled or add_coords(ndiag, d2) not in filled:
    #             add_line(ndiag, d1)
    #             add_line(ndiag, d2)
    #             filled.add(ndiag)
    #             ndiag = add_coords(ndiag, diag_dir)

    #     add_diag_line(start, (1,1))
    #     add_diag_line(start, (1,-1))
    #     add_diag_line(start, (-1,1))
    #     add_diag_line(start, (-1,-1))
    #     print('filled')
    #     return filled

class ExplorerStepRunner(BaseRunner):
    def __init__(self, game: Game, unit_coord: Coords):
        super().__init__()
        faction, num = game[unit_coord]
        self.pf = Explorer(game, faction)
        self.gen = self.pf.explore_it(start=unit_coord, num=num)

    def __call__(self) -> Optional[Tuple[np.ndarray, TextAnnot, Game]]:
        try:
            return next(self.gen) + (self.pf.game,)
        except StopIteration:
            return None


class ExplorerStepPlotter(ButtonGamePlotter):
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


class ExplorerRunner(BaseRunner):
    def __init__(self, game_gen):
        super().__init__()
        self.game_gen = game_gen

    def __call__(self, game=None):
        game = self.game_gen()
        return self.game_to_plot_els(game)

    @staticmethod
    def game_to_plot_els(game: Game) -> Tuple[Game, np.ndarray, List[TextAnnot]]:
        t0 = time()
        explorer = Explorer(game, Game.Vampire)
        i,j,num = next(game.vampires())
        bound = explorer.explore((i,j), num)
        # filled = explorer.fill((i,j), bound)
        t1 = time()
        print(f"Time spent: {t1-t0} s")  # : dist = {max(i,j)} / len path = {len(path) - 1}
        free8 = explorer.find_diag_non_diag_free((i,j), num)
        mat, text_annots = explorer.enemy_mats.step_im_mat(num)
        for c in bound:
            mat[c] = [0,100,0]
        for p in free8:
            mat[p] = [0,200,0]
        return game, mat, text_annots
    

class ExplorerPlotter(ButtonGamePlotter):
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
    gg = GameGenerator(30,30,vamp_pop=50, were_pop=150, human_pop=100, vamp_spread=1, were_spread=5, human_spread=5)
    
    ExplorerPlotter(ExplorerRunner(gg))

    # game = gg()
    # unit_coords = next(game.vampires())[:2]
    # ExplorerStepPlotter(ExplorerStepRunner(game, unit_coords))

    # game = Game(30, 30)
    # game[28,6] = Game.Vampire, 50
    # game[15,2] = Game.Human, 21
    # game[19,18] = Game.Human, 11
    # game[16,3] = Game.Werewolf, 31

    # from ..runner import CallableRuner
    # runner = CallableRuner(lambda _: ExplorerRunner.game_to_plot_els(game))
    # ExplorerPlotter(runner)