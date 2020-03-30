# encoding: utf-8
"""
@author: Jiahao LU
@contact: lujiahao8146@gmail.com
@file: try.py
@time: 2020/3/30
@desc:
"""
from itertools import product, chain
def itertry():
    comb_possibility = [[{(0,0,1,1,1)}, {2}, {3}], [{4}, {5}, {6}]]
    for possible_move in product(*comb_possibility):
        # yield chain(*possible_move)
        yield list(chain(*possible_move))

# for i in itertry():
    # print(i)
army = {(0,0,1),(1,1,2),(2,2,3)}
army  = army | {(4,4,4)}
print(army)


# army = self.game.werewolves() if self.next_player == Game.Werewolf else self.game.vampires()