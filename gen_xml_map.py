from argparse import ArgumentParser
from werevamp.game_gen import GameGenerator, SymGameGenerator


if __name__ == "__main__":
    gg = SymGameGenerator(m=8, n=8, players_pop=20, human_pop=40, human_spread=8, sym="x")
    gg().to_xml()
