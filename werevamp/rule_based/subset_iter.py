from typing import Iterable, Set, TypeVar, List, Generator
from itertools import combinations

def to_bin(idx: int) -> int:
    return 1 << idx

def to_idx(bin: int) -> int:
    return bin.bit_length() - 1


T = TypeVar("T")

class SubsetIter():
    def __init__(self, seq: Iterable[T]):
        self.seq: List[T] = list(seq)
        self._excluded = list()
        self._bin = [to_bin(i) for i in range(len(self.seq))]
        self._cur_sum = None

    def __iter__(self) -> Generator[Set[T], None, None]:
        return self.gen()
    
    def gen(self) -> Generator[Set[T], None, None]:
        for nel in range(1, len(self.seq) + 1):
            for bin_comb in combinations(self._bin, nel):
                cont = False
                sum_bin = sum(bin_comb)
                for excl in self._excluded:
                    if sum_bin & excl:
                        cont = True
                        break
                if cont: continue
                self._cur_sum = sum_bin
                yield {self.seq[i] for i in (to_idx(b) for b in bin_comb)}
    
    def exclude(self) -> None:
        self._excluded.append(self._cur_sum)



if __name__ == "__main__":
    seq = list(range(2))
    it = SubsetIter(seq)
    for part in it:
        print(part)
        if part == {4} or part == {1,3}:
            it.exclude()