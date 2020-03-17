from typing import Union, Tuple, Optional, Sequence

import numpy as np

def sigmoid(val):
    return 1 / (1 + np.exp(val))


def dist(i1: Union[int, Tuple[int, int]], j1: Union[int, Tuple[int, int]], i2: Optional[int] = None, j2: Optional[int] = None) -> int:
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


def get_adj_case(case: Sequence[int], shape: Tuple[int, int]) -> Sequence[Tuple[int, int]]:
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
    i, j = case[:2]
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


# def get_ring(center: Sequence[int], distance:int, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
#     """Get squares at distance `distance` from `center`, taking board `shape` into account.
    
#     Args
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
