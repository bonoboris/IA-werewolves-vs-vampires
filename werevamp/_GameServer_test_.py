# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:15:34 2020

@author: simed
"""

from .GameServer import GameServer
from .game import Game

initial_pop = {2 : [[(5,5),2],[(6,6),5]],
               1 : [[(0,0),10]],
               0 : [[(9,9),10]]}
g = Game(10,10, initial_pop)

gs = GameServer(g,'','', True)


# test get_valid_mov
tests = {(0,0) :[(0, 1), (1, 0), (1, 1)],
         (5,5) : [(4, 4), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5), (6, 6)],
         (9,9) : [(8,8),(8,9),(9,8)]}

for inp,out in tests.items():
    if gs.get_valid_move(inp) != out:
        print(f'error on get_valid_mov with input {inp}, received = {gs.get_valid_mov(inp)} instead of {out}')
#Testing end detection
if gs.is_game_ended()[0] == True:
    print('Error, server returns game ended while not ended')
del g[(0,0)]
if gs.is_game_ended()[0] != True:
    print('error, server returns game not ended while ended')
if gs.is_game_ended()[1] != 0:
    print(' error in determining the winning team')
    
# Testing battles
results_wanted = [(1,25),(1,27),(0,10),(0,19)]
results=[]
results.append(gs.fight((1,15),(2,10)))
results.append(gs.fight((1,15),(2,12)))
results.append(gs.fight((0,10),(1,5)))
results.append(gs.fight((2,9),(0,10)))
if results != results_wanted:
    print(f'error in battles,received {results} instead of {results_wanted}')

#testing movements authorisation
initial_pop = {2 : [[(1,1),5]],
               1 : [[(0,0),10]],
               0 : [[(2,2),10]]}
g = Game(3,3, initial_pop)
gs = GameServer(g,'','', True)

tests = {'no movements' : [],
         'not our unit' : [[2,2,4,1,1]],
         'invalid end coord' : [[0,0,5,2,2]],
         'too many units moved' : [[0,0,12,1,1]],
         'too many units moved in separate mov' : [[0,0,6,1,1],[0,0,5,1,0]],
         'no movements with action' : [[0,0,0,1,1]]}
for test, inp in  tests.items():
    if gs.is_authorized(inp,1) != False:
        print(f'test {test} failed !')
actions = [[0,0,5,0,1],[0,0,5,1,0]]
if gs.is_authorized(actions, 1) != True:
    print('server said action was false while True')
    
# test game update
gs.save_img()
gs.update_game(actions, 1)
gs.save_img()
gs.update_game([[2,2,10,2,1]],0)
gs.save_img()

gs.update_game([[0,1,5,1,1],[1,0,5,1,1]],1)
gs.save_img()

gs.update_game([[2,1,5,2,2],[2,1,5,1,2]],0)
gs.save_img()

if gs.is_authorized([[2,2,5,1,2],[1,2,5,1,1]],0):
    print(' Authorized movements while it shouldnt be')
gs.update_game([[1,1,15,1,2]],1)
gs.save_img()

gs.update_game([[2,2,5,1,2]],0)
gs.save_img()
ended, winner = gs.is_game_ended()
if not ended or winner != 1:
    print(' End game not properly detected')
