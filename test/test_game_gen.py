from os import path
from itertools import product
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))

import pytest
from werevamp.game import Game
from werevamp.game_gen import GameGenerator, SymGameGenerator

def sym_val(val):
    ty, num = val
    if ty == Game.Human:
        return val
    elif ty == Game.Vampire:
        return Game.Werewolf, num
    elif ty == Game.Werewolf:
        return Game.Vampire, num

def test_game_gen():
    N_ITER = 1

    m_range = [15, 20, 25, 50]
    n_range = [15, 20, 25, 50] 
    vp_range = [30,50,100] 
    wp_range = [30,50,100] 
    hp_range = [30,50,100]
    vs_range = [1,5,10] 
    ws_range = [1,5,10] 
    hs_range = [1,5,10]

    for args in product(m_range, n_range, vp_range, wp_range, hp_range, vs_range, ws_range, hs_range):
        m, n, vp, wp, hp = args[:5]
        gg = GameGenerator(*args)
        for i in range(N_ITER):
            game = gg()
            assert game.m == m
            assert game.n == n
            assert game.vampire_pop() == vp
            assert game.werewolf_pop() == wp
            assert game.human_pop() == hp

def test_sym_game_gen_x():
    N_ITER = 10

    m_range = [15, 20, 25, 50]
    n_range = [15, 20, 25, 50] 
    pp_range = [30,50,100] 
    hp_range = [30,50,100]
    hs_range = [4,10]

    try:
        for args in product(m_range, n_range, pp_range, hp_range, hs_range):
            m, n, pp, hp = args[:4]
            gg = SymGameGenerator(*args)
            for i in range(N_ITER):
                game = gg()
                assert game.m == m
                assert game.n == n
                assert game.vampire_pop() == pp
                assert game.werewolf_pop() == pp
                assert game.human_pop() == hp

                for (i, j), val in game.items():
                    assert game[m-1-i, j] == sym_val(val)
    except Exception as e:
        print("args:", args)
        raise e

def test_sym_game_gen_y():
    N_ITER = 10

    m_range = [15, 20, 25, 50]
    n_range = [15, 20, 25, 50] 
    pp_range = [30,50,100] 
    hp_range = [30,50,100]
    hs_range = [4,10]
    
    try: 
        for args in product(m_range, n_range, pp_range, hp_range, hs_range):
            m, n, pp, hp = args[:4]
            gg = SymGameGenerator(*args, sym='y')
            for i in range(N_ITER):
                game = gg()
                assert game.m == m
                assert game.n == n
                assert game.vampire_pop() == pp
                assert game.werewolf_pop() == pp
                assert game.human_pop() == hp

                for (i, j), val in game.items():
                    assert game[i, n-1-j] == sym_val(val)
    except Exception as e:
        print("args:", args)
        raise e

def test_sym_game_gen_r():
    N_ITER = 10

    m_range = [15, 20, 25, 50]
    n_range = [15, 20, 25, 50] 
    pp_range = [30,50,100] 
    hp_range = [30,50,100]
    hs_range = [4,10]

    try:
        for args in product(m_range, n_range, pp_range, hp_range, hs_range):
            m, n, pp, hp = args[:4]
            gg = SymGameGenerator(*args, sym="r")
            for i in range(N_ITER):
                game = gg()
                assert game.m == m
                assert game.n == n
                assert game.vampire_pop() == pp
                assert game.werewolf_pop() == pp
                assert game.human_pop() == hp

                for (i, j), val in game.items():
                    assert game[m-1-i, n-1-j] == sym_val(val)
    except Exception as e:
        print("args:", args)
        raise e
