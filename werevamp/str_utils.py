from tabulate import tabulate
import torch
import numpy as np


class SimpleRepr(object):
    "Simple __repr__ mixin class."
    __repr_exclude = {"_SimpleRepr__repr_exclude"}
    def __init__(self, exclude=set()):
        self.__repr_exclude = {"_SimpleRepr__repr_exclude"}.union(exclude)

    def __repr__(self):
        klass = self.__class__.__name__
        attrs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if k not in self.__repr_exclude)
        return f"{klass}({attrs})"


class TensorShapeRepr(object):
    def __repr__(self):
        klass = self.__class__.__name__
        attrs = []
        for k, v in vars(self).items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                attrs.append(f"{k}={v.shape!r}")
            else:
                attrs.append(f"{k}={v!r}")
        attrs = ", ".join(attrs)
        return f"{klass}({attrs})"


class ModuleSummary(object):
    """Add a summary method to torch.nn.module."""
    
    def summary(self) -> str:
        headers = "Name Dtype Shape Num Mem".split(" ")
        data = list()
        totel = 0
        totmem = 0
        for name, param in self.named_parameters():
            nel = param.nelement()
            totel += nel
            mem = nel * param.element_size()
            totmem += mem
            units = "B kB MB".split()
            u = 0
            for _ in range(len(units)):
                if mem > 1024:
                    mem /= 1024
                    u += 1
                else:
                    break
            mem = f"{mem:.3f} {units[u]}"
            data.append((".".join(name.split(".")[:-1]), param.dtype, tuple(param.size()), nel, mem))

        units = "B kB MB GB".split()
        u = 0
        for _ in range(len(units)):
            if totmem > 1024:
                totmem /= 1024
                u += 1
            else:
                break
        totmem = f"{totmem:.3f} {units[u]}"
        return tabulate(data, headers) + f"\n\nTotal: {totel} parameters ({totmem})"