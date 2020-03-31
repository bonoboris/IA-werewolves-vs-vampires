from werevamp.game import Game
from werevamp.game_gen import GameGenerator, SymGameGenerator
from werevamp.rule_based.player import Player
from werevamp.runner import GameRunner2
from werevamp.plotting import ButtonGamePlotter


if __name__ == "__main__":
    # gg = SymGameGenerator(20,20, human_spread=8, sym="r")
    gg = GameGenerator(20,20, human_spread=8)
    p1 = Player(Game.Vampire, True)
    p2 = Player(Game.Werewolf, True)

    for i in range(1000):
        print("Game", i+1)
        gr = GameRunner2(gg(), p1, p2)
        gr.run(max_turns=200)
        print("Done in", gr.num_states, "turns")


    # g = Game(10, 10)
    # g[2,2] = Game.Vampire, 10
    # g[5,5] = Game.Human, 5
    # g[8,8] = Game.Werewolf, 10

    # gr = GameRunner2(g, p1, p2)
    # ButtonGamePlotter(gr)