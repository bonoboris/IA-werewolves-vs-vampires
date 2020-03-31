

# -*- coding: utf-8 -*- 
""" 
Created on Sun Mar 29 13:06:32 2020 
 
@author: simed 
""" 
 
import time 
import sys 
import numpy as np

from werevamp.game import Game 
from werevamp.client import Client 
from werevamp.alpha_beta import make_move1
from werevamp.rule_based.player import Player

  
def play_one_game(init_params): 
    game, index_player = init_board(init_params) 
    rep = client.get_message('UPD') 
    print(rep) 
    cont = True 
    if rep ==[]: 
        print('Playing first') 
        decision, command = make_move(game, game.Vampire, depth) 
        time.sleep(0.5) 
        #print('1action = ', command) 
        rep = client.send_message(command) 
        #print('1Command sended') 
        #rep = client.get_message('UPD') 
        #print('1Update received') 
        game = update_game(game, rep, index_player) 
    else: 
        # print('Playing second') 
        # time.sleep(2) 
        # a=time.time() 
        # rep = client.get_message('UPD') 
        # print('2Received first update after', time.time()-a) 
        # print(rep) 
        game = update_game(game, rep, index_player) 
        decision, command = make_move(game, game.Vampire, depth) 
        rep = client.send_message(command) 
        game = update_game(game, rep, index_player) 
    while cont: 
        deb=time.time() 
        print('My turn') 
        decision, command = make_move(game, game.Vampire, depth) 
        #print(command) 
        #print(f'Elapsed {time.time()-deb} second for this move') 
        #deb=time.time() 
        time.sleep(0.5) 
        rep = client.send_message(command) 
        print(f'Elapsed {time.time()-deb} second for this turn') 
        #deb=time.time() 
        #rep = client.get_message('UPD') 
        #print(f'Elapsed {time.time()-deb} second for Receiving message') 
        if rep =='END': 
            ans = client.get_message('SET') 
            if ans =='BYE': 
                cont=False 
            else: 
                play_one_game(ans) 
        elif rep=='BYE': 
            cont=False 
        else: 
            game = update_game(game, rep, index_player) 
            print('Game updated') 
            print(game) 
 
     
def update_game(game, upd, index_player): 
    for up in upd: 
        x,y,n_h,n1,n2 =up 
        if n_h==0 and n1==0 and n2==0: 
            del game[x,y] 
        else: 
            nb_player = up[index_player] 
            if nb_player ==0: 
                game[x,y] = (Game.Werewolf, max(n1,n2)) 
            else: 
                game[x,y] = (Game.Vampire, max(n1,n2)) 
    return game 
             
         
         
def init_board(ans): 
    m,n = ans[0] 
    n_h = ans[1] 
    depart = ans[n_h+2] 
    initial_pop = {Game.Human : []} 
    for i in range(n_h+3, len(ans)): 
        x,y,n_h, n1, n2 = ans[i] 
        if (x,y) == depart: 
            initial_pop[Game.Vampire] = [[(x,y),max(n1,n2)]] 
            index_player = 3 if n1>n2 else 4 
        elif n_h ==0: 
            initial_pop[Game.Werewolf] = [[(x,y),max(n1,n2)]] 
        else: 
            initial_pop[Game.Human].append([(x,y),n_h])         
    game=Game(n,m,initial_pop) 
    return game, index_player 
 
# def action_to_mov_command(actions: List[Action]): 
#     action_tuples = [(*action.start(), action.num, *action.dest()) for action in actions] 
#     return ['MOV', len(action), action_tuples] 
 
if __name__ =='__main__': 
    name = sys.argv[1]
    if name=='Boris':
        p2 = Player(Game.Werewolf, True)
        def make_move(game, faction, *args):
            return p2.make_move(game, faction)
    else:
        make_move=make_move1
    depth = int(sys.argv[2]) 
    client = Client() 
    ans = client.connect_to_server('127.0.0.1', 5555, name) 
    play_one_game(ans) 
 
    

 
     
     
     
 
 
 
