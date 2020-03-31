from argparse import ArgumentParser
from werevamp.game_gen import GameGenerator, SymGameGenerator


if __name__ == "__main__":
    gg = SymGameGenerator(m=20, n=20, players_pop=10, human_pop=20, human_spread=6, sym="x")
    gg().to_xml()
