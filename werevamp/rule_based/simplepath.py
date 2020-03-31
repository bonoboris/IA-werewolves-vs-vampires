from typing import List
from ..utils import Coords, add_coords, sub_coords, scale_coords, srange, sign

def simple_path(start: Coords, dest:Coords, forbidden=dict(), step_offset=0) -> List[Coords]:
    path = _simplepath(start, dest)
    forbidden_in_path = [case for step, case in enumerate(path) if case in forbidden.get(step+step_offset, set())]
    if not forbidden_in_path:
        return path
    else:
        ff = forbidden_in_path[0]
        if _is_diag(start, ff):
            dif = sub_coords(dest, start)
            udif = (sign(dif[0]), sign(dif[1]))
            mdir = 0 if abs(dif[0]) >= abs(dif[1]) else 1
            print(mdir)
            rk = abs(dif[mdir]) - abs(dif[1-mdir])
            mdir_t = list(udif)
            mdir_t[1-mdir] = 0
            mdir_t = tuple(mdir_t)
            for k in range(1, rk):
                piv1 = add_coords(start, scale_coords(k, mdir_t))
                path = _hv_path(start, piv1)
                path.extend(_simplepath(piv1, dest))
                fip = [case for step, case in enumerate(path) if case in forbidden.get(step+step_offset, set())]
                if not fip:
                    return path




def _simplepath(start: Coords, dest:Coords) -> List[Coords]:
    dif = sub_coords(dest, start)
    udif = (sign(dif[0]), sign(dif[1]))
    k = min(abs(dif[0]), abs(dif[1]))
    pivot = add_coords(start, scale_coords(k, udif))
    path = _diag_path(start, pivot)
    path.extend(_hv_path(pivot, dest))
    return path


def _hv_path(start: Coords, dest:Coords) -> List[Coords]:
    assert start[0] == dest[0] or start[1]==dest[1]
    path = []
    if start[0] == dest[0]:
        for j in srange(start[1], dest[1], True, False):
            path.append((dest[0], j))
    elif start[1] == dest[1]:
        for i in srange(start[0], dest[0], True, False):
            path.append((i, dest[1]))
    return path

def _diag_path(start: Coords, dest:Coords) -> List[Coords]:
    dif = sub_coords(dest, start)
    assert abs(dif[0]) == abs(dif[1])
    udif = (sign(dif[0]), sign(dif[1]))
    k = min(abs(dif[0]), abs(dif[1]))
    path = []
    for i in range(k):
        path.append(add_coords(start, scale_coords(i+1, udif)))
    return path


def _is_diag(c1: Coords, c2: Coords) -> bool:
    dif = sub_coords(c1, c1)
    return abs(dif[0]) == abs(dif[1])


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from ..plotting import ButtonGamePlotter
    from ..runner import BaseRunner
    from ..game_gen import rd_coords
    import random as rd
    import numpy as np

    class Runner(BaseRunner):
        def __init__(self, m=30, n=30):
            super().__init__()
            self.m = m
            self.n = n

        def __call__(self, game=None):
            # coords = rd_coords((self.m, self.n), 12)
            # start = coords[0]
            # dest = coords[1]
            # obs = coords[2:]
            start = (0,0)
            dest = (9,5)
            obs = {2:{(2,2)}}
            path = simple_path(start, dest, obs)
            print("Len path:", len(path))
            print(path)
            mat = np.zeros((self.m, self.n, 3), dtype=int)
            mat[start] = [0,100,0]
            for c in path:
                mat[c] = [0,100,100]
            for step, lo in obs.items():
                for o in lo:
                    mat[o] = [100, 0, 0]
            mat[dest] = [0,0,100]
            return mat
    
    class Plotter(ButtonGamePlotter):
        def update(self, mat):
            self._update_data(mat)
            plt.draw()
    
    Plotter(Runner())