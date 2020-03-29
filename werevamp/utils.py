from typing import Union, Tuple, Optional, Sequence, TypeVar, List, Callable, Iterable
from numbers import Real

import numpy as np

Coords = Tuple[int, int]
RCoords = Tuple[Real, Real]

T = TypeVar('T')


def find(seq: Sequence[T], predicate: Callable[[T], bool]) -> int:
    """Find the first element of 'seq' for which the `predicate` callable returns True, and return its index, or return -1 if None found."""
    for i, el in enumerate(seq):
        if predicate(el):
            return i
    return -1 


def min_argmin(seq: Sequence[T]) -> Tuple[T, List[int]]:
    """Returns the min value of `seq` and the list of indices with min value."""
    if len(seq) == 0:
        raise ValueError("Empty sequence doen't have a minimum")
    mval, midx = seq[0], [0]
    for idx, val in enumerate(seq[1:]):
        if val < mval:
            mval = val
            midx = [idx+1]
        elif val == mval:
            midx.append(idx+1)
    return mval, midx


def valid_coords(shape: Coords, coords: Coords) -> bool:
    return 0 <= coords[0] and coords[0] < shape[0] and 0 <= coords[1] and coords[1] < shape[1]


def add_coords(c1: Coords, c2:Coords) -> Coords:
    """Return c1 + c2.""" 
    return c1[0] + c2[0], c1[1] + c2[1]


def sub_coords(c1: Coords, c2:Coords) -> Coords:
    """Return c1 - c2.""" 
    return c1[0] - c2[0], c1[1] - c2[1]


def scale_coords(k:int, c: Coords) -> Coords:
    return k * c[0], k * c[1]

def sigmoid(val):
    """Scalar sigmoid function."""
    return 1 / (1 + np.exp(-val))


def sign(val, zero_val=0):
    """Return 1 if `val` > 0, -1 if `val` < 0 and `zero_val` if `val` == 0 (`zero_val` defaults to 0)."""
    if val > 0: return 1
    elif val == 0: return zero_val
    else: return -1


def clamp(val: Union[Real, RCoords], lims: Union[Real, RCoords]):
    """Clamps `val` between `lims`.

    if `val`= int and `lims`= int then clamps `val` between 0 included and `lims` excluded
    if `val`= int and `lims`= (a:int, b:int) then clamps `val` between a included and b excluded
    if `val` is a sequence then `lims` must also be a sequence of the same lenght and the function is recursively called for each pair of element in zip(val, lims)  
    """
    if isinstance(val, Sequence):
        return tuple((clamp(v, l) for v, l in zip(val, lims)))
    
    if isinstance(lims, Sequence):
        lmin, lmax = lims
    else:
        lmin, lmax = 0, lims
    val = max(lmin, val)
    val = min(lmax - 1, val)
    return val

def dist(i1: Union[int, Coords], j1: Union[int, Coords], i2: Optional[int] = None, j2: Optional[int] = None) -> int:
    """Distance in terms of moves between two points on the board.
    
    Args
    ----
        either `(i1,j1), (i2,j2)` or `i1,j1,i2,j2`
    
    Returns
    -------
        max(abs(i1-i2), abs(j1-j2))
    """
    if i2 is None:
        i1, j1, i2, j2 = (*i1, *j1)
    return max(abs(i1-i2), abs(j1-j2))


def get_adj_case(shape: Coords, case: Coords) -> Sequence[Coords]:
    """Get adjacent squares, taking board `shape` into account.
    
    Args
    ----
        case: Sequence[int]
            The case for which the adjacent squares are returned, the 2 first elements must be the coordinates
        shape: (int, int)
            The board dimensions
    Returns
    -------
        The list of adjacent squares 
    """
    i, j = case
    m, n = shape
    ret = []
    # list adj_case going through corners (-1,-1) -> (-1, +1) -> (+1, +1) -> (+1, -1) -> (-1,-1)
    #   0----1----2     --j-> 
    #   |         |
    #   7    x    3
    #   |         |
    #   6----5----4
    #
    #   |
    #   i
    #   |
    #   v

    mask = np.repeat(True, 8)
    if i-1 < 0: mask[0:3] = False
    if j+1 >= n: mask[2:5] = False
    if i+1 >= m: mask[4:7] = False
    if j-1 < 0:
        mask[6:] = False
        mask[0] = False
    
    all_adj = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]
    ret = []
    for m, c in zip(mask, all_adj):
        if m: ret.append(c)
    return ret

def srange(b1:int, b2:int, exclude_b1=False, exclude_b2=False) -> range:
    """Returns a range from from `b1` to `b2` with step either +1 or -1, and by default including both `b1` and `b2`."""
    if b1 <= b2:
        r1 = b1 + 1 if exclude_b1 else b1 
        r2 = b2 if exclude_b2 else b2 + 1 
        return range(r1, r2)
    else:
        r1 = b1 - 1 if exclude_b1 else b1
        r2 = b2 if exclude_b2 else b2 - 1
        return range(r1, r2, -1)


def get_rectangle(corner1: Coords, corner2:Coords, inside:bool=False) -> List[Coords]:
    """Returns the coords of all squares in the rectangle defined by the corners `corner1` and `corner2`, if inside is true returns also the interior squares."""
    ret = list()
    i1, j1 = corner1
    i2, j2 = corner2
    if i1==i2:
        return [(i1, j) for j in srange(j1, j2)]
    elif j1==j2:
        return [(i, j1) for i in srange(i1, i2)]

    if inside:
        for j in srange(j1, j2):
            for i in srange(i1, i2):
                ret.append((i,j))
    else:
        ret.extend(((i1, j) for j in srange(j1, j2)))
        ret.extend(((i, j2) for i in srange(i1, i2, exclude_b1=True)))
        ret.extend(((i2, j) for j in srange(j2, j1, exclude_b1=True)))
        ret.extend(((i, j1) for i in srange(i2, i1, exclude_b1=True, exclude_b2=True)))
    
    return ret

def get_diag_rectangle(corner1: Coords, corner2, inside:bool=False):
    pass

# def get_ring(center: Sequence[int], distance:int, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
#     """Get squares at distance `distance` from `center`, taking board `shape` into account.
    
#     Argsdefine
#     ----
#         center: Sequence[int]
#             The case from which the squares at the distance specified distance are returned, the 2 first elements must be the coordinates
#         distance: int
#             The distance 
#         shape: (int, int)
#             The board dimensions
#     Returns
#     -------
#         The list of adjacent squares 
#     """
#     i, j = case[:2]
#     m, n = shape
#     ret = []

if __name__ == "__main__":
    pass