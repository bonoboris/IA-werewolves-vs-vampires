from typing import NamedTuple
import torch
from collections import namedtuple

class Foo():
    def __init__(self):
        self.foo = 42
    
    @staticmethod
    def get_player_class():
        return Bar

class Bar():
    def __init__(self):
        self.bar = "hello"
    
    @staticmethod
    def get_model_class():
        return Foo

f = Foo()
b = Foo.get_player_class()
f2 = b.get_model_class()()
b2 = f2.get_player_class()()

print(f2.foo)
print(b2.bar)