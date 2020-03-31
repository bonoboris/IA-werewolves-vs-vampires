# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:04:42 2020

@author: simed
"""
import sys
from werevamp.client import Client
from play_game import play_one_game


if __name__ =='__main__':
    ip = sys.argv[1]
    port = sys.argv[2]
    depth = 1 
    client = Client()
    name = 'Groupe 4'
    ans = client.connect_to_server(ip,int(port), name)
    n_h = ans[1]
    play_one_game(ans, client)