

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
# from werevamp.alpha_beta import make_move1
from werevamp.rule_based.player import Player

d = {3 : Game.Vampire,
     4 : Game.Werewolf}
             
def play_one_game(init_params, client, *args): 
    game, index_player, us = init_board(init_params)
    p2 = Player(us, False)
    rep = client.get_message('UPD') 
    print(rep) 
    cont = True 
    if rep ==[]: 
        command = p2.make_move(game) 
        time.sleep(0.5) 
        rep = client.send_message(command) 
        game = update_game(game, rep, index_player) 
    else: 
        game = update_game(game, rep, index_player) 
        command = p2.make_move(game) 
        rep = client.send_message(command) 
        game = update_game(game, rep, index_player) 
    while cont: 
        deb=time.time() 
        print('My turn') 
        command = p2.make_move(game) 
        time.sleep(0.5) 
        rep = client.send_message(command) 
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

     
def update_game(game, upd, index_player): 
    for up in upd: 
        x,y,n_h,n1,n2 =up 
        if n_h==0 and n1==0 and n2==0: 
            del game[x,y] 
        else: 
            nb_player = up[index_player]
            index_other = 4 if index_player ==3 else 3
            if nb_player ==0: 
                game[x,y] = (d[index_other], max(n1,n2)) 
            else: 
                game[x,y] = (d[index_player], max(n1,n2)) 
    return game


         
         
def init_board(ans): 
    m,n = ans[0] 
    nb_h = ans[1] 
    depart = ans[nb_h+2] 
    initial_pop = {Game.Human : []} 
    for i in range(nb_h+3, len(ans)):
        x,y,n_h, n1, n2 = ans[i] 
        if (x,y) == depart: 
           index_player = 3 if n1>n2 else 4
           if index_player ==3:
               us = Game.Vampire
               them = Game.Werewolf
           else:
               us = Game.Werewolf
               them = Game.Vampire
    for i in range(nb_h+3, len(ans)): 
        x,y,n_h, n1, n2 = ans[i] 
        if (x,y) == depart: 
            initial_pop[us] = [[(x,y),max(n1,n2)]] 
        elif n_h ==0: 
            initial_pop[them] = [[(x,y),max(n1,n2)]] 
        else: 
            initial_pop[Game.Human].append([(x,y),n_h])         
    game=Game(n,m,initial_pop) 
    return game, index_player, us 
 

if __name__ =='__main__': 
    ip = sys.argv[1]
    port = sys.argv[2]
    name = 'BORIS'
    client = Client() 
    ans = client.connect_to_server(ip, int(port), name) 
    play_one_game(ans, client) 
 
    

 
     
     
     
 
 
 
