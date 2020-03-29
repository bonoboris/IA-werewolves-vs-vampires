from typing import Optional, Iterable, List, Callable, Tuple, Dict, Any
import itertools
import functools

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import numpy as np

from .game import Game
from .utils import Coords
from .runner import INavigableGameStates
from .influence import get_dominance_map, get_influence_map


TextAnnot = Tuple[int, int, Any]


def set_text(
        ax: plt.Axes,
        coord_text_it: Iterable[TextAnnot],
        textcol: str ="w",
        bbox_col: Optional[str] = None,
        override: bool = True
    ) -> Dict[Coords, plt.Text]:

    existing = {(text._y, text._x): text for text in ax.texts}
    bbox = dict(facecolor=bbox_col, alpha=0.5) if bbox_col is not None else None
    for i, j, txt in coord_text_it:
        if not override and (i,j) in existing:
            raise ValueError("Trying to set two text at the same coordinates")
        existing[i, j] = ax.text(j, i, txt, ha="center", va="center", color=textcol, bbox=bbox)
    return existing


def set_grid(ax: plt.Axes, shape: Tuple[int, int]) -> None:
    m, n = shape
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


class ButtonGamePlotter():
    """Plot a game states and allow navigation with button and arrow keys."""
    def __init__(self, game_states: INavigableGameStates):
        self.game_states = game_states

        n_widgets = 3

        self.fig: plt.Figure = plt.figure(constrained_layout=True)
        self.gs = self.fig.add_gridspec(n_widgets, 2, width_ratios=[0.85,0.15])
        self.ax: plt.Axes = self.fig.add_subplot(self.gs[:, 0])
        self.wid_axs: List[plt.Axes] = [self.fig.add_subplot(self.gs[i,1]) for i in range(n_widgets)]
        
        self.next_bt: Button = Button(self.wid_axs[0], r"next $\rightarrow$")
        self.prev_bt: Button = Button(self.wid_axs[1], r"prev $\leftarrow$")
        self.influence_rbt: RadioButtons = RadioButtons(self.wid_axs[2], ["None", "Influence", "Dominance"])
        self.influence_mode = "None"

        self.next_bt.on_clicked(self._next_event_handler)
        self.prev_bt.on_clicked(self._prev_event_handler)
        self.fig.canvas.mpl_connect('key_press_event', self._key_pressed_event_handler)
        self.influence_rbt.on_clicked(self._infl_rbt_event_handler)

        self.im: Optional[mpl.image.AxesImage] = None
        if self.game_states.num_states == 0:
            self.game_states.next()
        self.update(self.game_states.cur())
        plt.show()

    def update(self, game: Game, influence: Optional = None, dont_draw=False) -> None:
        if game is None:
            return

        if self.influence_mode == "Influence":
            ker_size = max(game.size()) * 2 + 1
            if ker_size % 2 == 0: ker_size += 1
            influence = get_influence_map(game, kernel_size= ker_size)
        elif self.influence_mode == "Dominance":
            influence = get_dominance_map(game)

        if influence is None:
            self._update_data(self.get_game_rgb(game))
            self.ax.texts.clear()
            set_text(self.ax, itertools.chain(game.humans(), game.vampires(), game.werewolves()))
        
        else:
            self._update_data(influence)
            self.ax.texts.clear()
            set_text(self.ax, game.humans(), bbox_col='b')
            set_text(self.ax, game.vampires(), bbox_col='g')
            set_text(self.ax, game.werewolves(), bbox_col='r')

        if not dont_draw:
            plt.draw()
    
    @staticmethod
    def get_game_rgb(game: Game) -> np.ndarray:
        return np.where(game.to_matrix() > 0, 122, 0)
    
    def _update_data(self, data_mat, **kwargs):
        if self.im is None:
            self.im = self.ax.imshow(data_mat, **kwargs)
            set_grid(self.ax, data_mat.shape[:2])
        else:
            self.im.set_data(data_mat)
            self.im.set(**kwargs)

    def _key_pressed_event_handler(self, event):
        if event.key == "left":
            self._prev_event_handler()
        elif event.key == "right":
            self._next_event_handler()

    def _next_event_handler(self, event=None):
        self.update(self.game_states.next())
    
    def _prev_event_handler(self, event=None):
        self.update(self.game_states.prev())

    def _infl_rbt_event_handler(self, label):
        if label != self.influence_mode:
            self.influence_mode = label
            self.update(self.game_states.cur())


# def plot_mat(mat) -> Tuple[plt.Axes, Any]:
#     m, n = mat.shape[:2]
#     plt.figure()
#     im = plt.imshow(mat, aspect='equal')

#     ax = plt.gca()

#     # Major ticks
#     ax.set_xticks(np.arange(0, m, max(1, m//10)))
#     ax.set_yticks(np.arange(0, n, max(1, m//10)))

#     # Labels for major ticks
#     ax.set_xticklabels(np.arange(1, m + 1, max(1, m//10)))
#     ax.set_yticklabels(np.arange(1, n + 1, max(1, m//10)))

#     # Minor ticks
#     ax.set_xticks(np.arange(-.5, m, 1), minor=True)
#     ax.set_yticks(np.arange(-.5, n, 1), minor=True)

#     # Gridlines based on minor ticks
#     ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

#     return ax, im

if __name__ == "__main__":
    from .game_gen import GameGenerator
    from .runner import GameRunner2 as GameRunner
    from .player import TestPlayer 

    game = GameGenerator(m=10, n=10, human_pop=100, human_spread=4)()
    
    p1 = TestPlayer(Game.Vampire)
    p2 = TestPlayer(Game.Werewolf)
    gr = GameRunner(game, p1, p2)
    gp = ButtonGamePlotter(gr)
