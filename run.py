from time import sleep, time

import torch

from werevamp.game_gen import SymGameGenerator
from werevamp.game import GamePlotter, Game, GameRunner
from werevamp.dummy import Dummy
from werevamp.cnn import PlayerCNN
from werevamp.rule_based.player import Player as PlayerRB


def plot_game(game:Game, p1, p2, delay=1., p1_first:bool=True, num_max_turns=100):
    gr = GameRunner()
    gp = GamePlotter(game)
    for game in gr.runit(game, p1, p2, p1_first, num_max_turns):
        if delay == "input":
            input("Press enter to compute next move...")
        else:
            sleep(delay)
        gp.update(game)
    print("Winner: ", gr.winner)


def run_multiple_games(game_gen, p1, p2, ngames, num_max_turns):
    results = []
    gr = GameRunner()
    for i in range(ngames):
        print(f"Game {i+1}/{ngames}")
        results.append(gr.run(game_gen(), p1, p2, bool(i%2), num_max_turns).winner())
    return results

def run_multiple_games_plot(game_gen, p1, p2, ngames, num_max_turns):
    results = []
    gr = GameRunner()
    gp = GamePlotter(game_gen())
    for i in range(ngames):
        print(f"Game {i+1}/{ngames}")
        for g in gr.runit(game_gen(), p1, p2, bool(i%2), num_max_turns):
            gp.update(g)
        results.append(gr.winner)
    return results


def _debug_game(vamp_pos, wer_pos, human_pos):
    g = Game(50,50)
    g[vamp_pos] = Game.Vampire, 100
    g[wer_pos] = Game.Werewolf, 100
    g[human_pos] = Game.Human, 100
    return g


if __name__ == "__main__":
    gg = SymGameGenerator(15, 15, players_pop=50, human_pop=50, human_spread=4, sym="x")
    # g = gg()

    # ug = _debug_game((0,5), (25,30), (12, 8))

    pop = torch.load("werevamp/data/gen20.pt")
    print(pop[0].conv_block.radial1.get_params())

    for i in range(len(pop)):
        p1 = PlayerCNN(Game.Vampire, pop[18], True)
        p2 = PlayerCNN(Game.Werewolf, pop[18], True)
        plot_game(gg(), p1, p2, delay=0.001, num_max_turns=50)

    # main(ug, p1, p2, delay=0.1)
        
