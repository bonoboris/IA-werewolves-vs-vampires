from typing import List
from ..utils import Coords, add_coords, sub_coords, scale_coords, srange, sign

def simple_path(start: Coords, dest:Coords) -> List[Coords]:
    path = list()
    dif = sub_coords(dest, start)
    udif = (sign(dif[0]), sign(dif[1]))
    k = min(abs(dif[0]), abs(dif[1]))
    for i in range(k):
        path.append(add_coords(start, scale_coords(i+1, udif)))
    m = path[-1] if path else start
    if m[0] == dest[0]:
        for j in srange(m[1], dest[1], True, False):
            path.append((dest[0], j))
    elif m[1] == dest[1]:
        for i in srange(m[0], dest[0], True, False):
            path.append((i, dest[1]))
    return path


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from ..plotting import ButtonGamePlotter
    from ..runner import BaseRunner
    import random as rd
    import numpy as np

    class Runner(BaseRunner):
        def __init__(self, m=30, n=30):
            super().__init__()
            self.m = m
            self.n = n

        def __call__(self, game=None):
            start = (rd.randint(0, self.m -1), rd.randint(0, self.n -1))
            dest = (rd.randint(0, self.m -1), rd.randint(0, self.n -1))
            path = simple_path(start, dest)
            print("Len path:", len(path))
            print(path)
            mat = np.zeros((self.m, self.n, 3), dtype=int)
            mat[start] = [0,100,0]
            for c in path:
                mat[c] = [0,100,100]
            mat[dest] = [0,0,100]
            return mat
    
    class Plotter(ButtonGamePlotter):
        def update(self, mat):
            self._update_data(mat)
            plt.draw()
    
    Plotter(Runner())