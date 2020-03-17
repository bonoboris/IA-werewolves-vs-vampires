from typing import NamedTuple
import torch
from collections import namedtuple

class SimpleRepr(object):
    "Simple __repr__ mixin class."
    def __repr__(self):
        klass = self.__class__.__name__
        attrs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{klass}({attrs})"

class ModuleSummary(object):
    """Add a summary method to torch.nn.module."""
    
    def summary(self) -> str:
        return "summary"

class Bar():
    class Baz(NamedTuple, SimpleRepr):
        baz: str
    def __init__(self):
        pass
    
    @property
    def params(self):
        return self.Baz("hello")

class Foo(torch.nn.Module, ModuleSummary):
    class Params(NamedTuple, SimpleRepr):
        foo: Bar

    def __init__(self, bar):
        super().__init__()
        self.bar = bar
        self.l = torch.nn.Linear(5,10,bias=False)
    
    @property
    def params(self):
        print("in params")
        return self.Params(self.bar.params)

class Test(SimpleRepr):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

f = Test(params="Hello", foo = 42)
r = repr(f)
print(r)