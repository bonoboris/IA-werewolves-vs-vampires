import random as rd
import math

import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

from .game import Game
from .game_gen import GameGenerator, SymGameGenerator
from .utils import dist


def gen_kernels(kernel_type, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    if kernel_type == "zero":
        return np.zeros((kernel_size, kernel_size), dtype=float)

    ker_radius = kernel_size // 2
    mid_point = (ker_radius, ker_radius)

    def linear_f(i,j): return 1 - dist(mid_point, (i,j)) / (1 + ker_radius)
    
    if kernel_type == "linear":
        def f(i,j): return linear_f(i,j)
    elif kernel_type == "square":
        def f(i,j): return linear_f(i,j) ** 2
    elif kernel_type == "inv":
        def f(i,j): return 1 / (dist(mid_point, (i,j)) + 1)
    elif kernel_type == "exp":
        def f(i,j): return math.exp(-dist(mid_point, (i,j)))

    ker = np.zeros((kernel_size, kernel_size), dtype=float)
    for i in range(kernel_size):
        for j in range(kernel_size):
            ker[i,j] = f(i,j)
    
    return ker

def get_influence_map(game: Game, ally_type:int=Game.Vampire, kernel_type="linear", kernel_size=101):
    oppo_type = Game.Werewolf if ally_type == Game.Vampire else Game.Vampire
    mat = game.to_matrix()
    ker = gen_kernels(kernel_type, kernel_size)
    
    ally_map = mat[:,:,ally_type]
    oppo_map = mat[:,:,oppo_type]
    return convolve2d(ally_map, ker, mode="same") - convolve2d(oppo_map, ker, mode="same")


def get_dominance_map(game: Game, kernel_type="linear", kernel_size=101):
    def all_dist(coords):
        return sorted((dist(coords, k), *v) for k, v in game.items())
    m,n = game.size()
    mat = np.zeros(game.size(), dtype=int)
    for i in range(m):
        for j in range(n):
            sd = all_dist((i,j))
            ty = sd[0][1]
            num = sd[0][2]
            for k in range(1, len(sd)):
                if sd[k][1] == ty:
                    num += sd[k][2]
                else:
                    break
            mat[i, j] = (1 - 2*ty)*num
    return mat


def plot_influence_map(game: Game, influence: np.ndarray):
    fig, ax = plt.subplots()
    ax: plt.Axes = ax
    im = ax.imshow(influence)
    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(0, game.m, 5))
    ax.set_yticks(np.arange(0, game.n, 5))
    ax.set_xticklabels(np.arange(0, game.m, 5))
    ax.set_yticklabels(np.arange(0, game.n, 5))

    ax.set_xticks(np.arange(-0.5, game.m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, game.n, 1), minor=True)
    ax.grid(True, which='minor', axis='both',  linestyle='-', linewidth=1, color="w")

    # Loop over data dimensions and create text annotations.
    for i, j, num in game.vampires():
        text = ax.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor="g", alpha=0.5))
    for i, j, num in game.werewolves():
        text = ax.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor="r", alpha=0.5))
            

    ax.set_title("Influence map")
    fig.tight_layout()
    plt.show()


def bin_plot_influence_map(game: Game, influence: np.ndarray):
    bin_influence = np.where(influence > 0, 1., -1.)
    bin_influence[influence==0] = 0.

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    im1 = ax1.imshow(bin_influence)
    im2 = ax2.imshow(influence)

    cbar = ax2.figure.colorbar(im2, ax=ax2)

    ax1.set_xticks(np.arange(0, game.m, 5))
    ax1.set_yticks(np.arange(0, game.n, 5))
    ax1.set_xticklabels(np.arange(0, game.m, 5))
    ax1.set_yticklabels(np.arange(0, game.n, 5))

    ax2.set_xticks(np.arange(0, game.m, 5))
    ax2.set_yticks(np.arange(0, game.n, 5))
    ax2.set_xticklabels(np.arange(0, game.m, 5))
    ax2.set_yticklabels(np.arange(0, game.n, 5))


    ax1.set_xticks(np.arange(-0.5, game.m, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, game.n, 1), minor=True)
    ax1.grid(True, which='minor', axis='both', linestyle='-', linewidth=1, color="w")

    ax2.set_xticks(np.arange(-0.5, game.m, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, game.n, 1), minor=True)
    ax2.grid(True, which='minor', axis='both',  linestyle='-', linewidth=1, color="w")

    # Loop over data dimensions and create text annotations.
    for i, j, num in game.vampires():
        text = ax1.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor="g", alpha=0.5))
        text = ax2.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor="g", alpha=0.5))
    for i, j, num in game.werewolves():
        text = ax1.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor="r", alpha=0.5))
        text = ax2.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor="r", alpha=0.5))
    for i, j, num in game.humans():
        text = ax1.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor="b", alpha=0.5))
        text = ax2.text(j, i, num, ha="center", va="center", color="w", bbox=dict(facecolor="b", alpha=0.5))

    ax1.set_title("Bin influence map")
    ax2.set_title("Influence map")
    fig.tight_layout()
    
    plt.show()



if __name__ == "__main__":
    for i in range(100):
        game = GameGenerator(m=10, n=10,vamp_spread=2, were_spread=2, human_pop=0, human_spread=0)()
        print(game)
        influence = get_dominance_map(game, "linear", 101)
        print(influence[next(game.vampires())[:2]])
        print(influence[next(game.werewolves())[:2]])
        plot_influence_map(game, influence)