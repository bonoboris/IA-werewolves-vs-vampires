from os import path
import sys
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))

import pytest
from werevamp.game import Game


def test_set():
    g = Game(10, 5)
    k, v = (1,2), (Game.Vampire, 10)
    
    g[k] = v
    assert g.items() == {k:v}.items()

def test_set_bad_key():
    g = Game(10, 5)
    v = (Game.Vampire, 10)
    ks = [(0, -1), (-1, 0), (10, 0), (0, 5)]

    for k in ks:
        with pytest.raises(IndexError):
            g[k] = v

def test_set_bad_val():
    g = Game(10, 5)
    k = (1,2)
    vs = [(-1, 10), (4, 10), (Game.Vampire, -1)]

    for v in vs:
        with pytest.raises(ValueError):
            g[k] = v

def test_size():
    g = Game(10, 5)
    assert g.size() == (10, 5)

def test_populations():
    # test grow with set
    g = Game(10, 5)
    g[0,1] = (Game.Human, 5)
    g[0,2] = (Game.Vampire, 10)
    g[1,1] = (Game.Vampire, 5)
    assert g.populations() == {Game.Human: 5, Game.Werewolf: 0, Game.Vampire:15}
    assert g.human_pop() == 5
    assert g.werewolf_pop() == 0
    assert g.vampire_pop() == 15

    # test copy
    g.populations()[Game.Human] += 5
    assert g.human_pop() == 5

    # test adjust correctly
    g[0,1] = (Game.Werewolf, 5)
    assert g.populations() == {Game.Human: 0, Game.Werewolf: 5, Game.Vampire:15}

    del g[1,1]
    assert g.populations() == {Game.Human: 0, Game.Werewolf: 5, Game.Vampire:10}

def test_lists():
    g = Game(10, 5)
    g[0,1] = (Game.Human, 5)
    g[0,2] = (Game.Vampire, 10)
    g[1,1] = (Game.Vampire, 5)

    assert g.humans() == {(0,1,5)}
    assert g.werewolves() == set()
    assert g.vampires() == {(0,2,10), (1,1,5)}

    g[1,1] = (Game.Werewolf, 5)

    assert g.humans() == {(0,1,5)}
    assert g.werewolves() == {(1,1,5)}
    assert g.vampires() == {(0,2,10)}