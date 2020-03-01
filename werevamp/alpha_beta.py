from game import Game
import math
from typing import Tuple, Iterator, List, Iterable
from anytree import Node, RenderTree
from copy import deepcopy


def print_tree(root):
    if root is not None:
        for pre, _, node in RenderTree(root):
            print("%s%s next player: %s" % (pre, node.name.move, node.name.next_player))  # node.name is Game instance
    else:
        raise ValueError("Tree root is empty.")


class State:
    def __init__(self, this_board: Game, next_player: int, this_move: List[Tuple[int, int, int, int, int]] = None):
        """

        :param this_move: the move which leads to this_board
        :param this_board: current chessboard
        :param next_player: next player
        """
        self.game = this_board
        self.move = this_move
        self.next_player = next_player

    def expand_tree(self):
        next_player = Game.Werewolf + Game.Vampire - self.next_player

        for possible_move in self.explore_possibilities():
            next_turn = self.move_on_board(possible_move)
            yield State(next_turn, next_player, possible_move)

    # to complete
    def explore_possibilities(self) -> Iterator[List[Tuple[int, int, int, int, int]]]:
        army = self.game.werewolves() if self.next_player == Game.Werewolf else self.game.vampires()
        for x, y, population in army:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    # for now just move the entire units
                    if x + dx >= self.game.m or x + dx < 0 or y + dy >= self.game.n or y + dy < 0:
                        continue
                    # only consider moves here, do not consider result after random battle
                    print([(x, y, population, x + dx, y + dy)])
                    yield [(x, y, population, x + dx, y + dy)]
                    # possibilities.append(new_state)

    # to complete
    def move_on_board(self, mov: List[Tuple[int, int, int, int, int]]) -> Game:
        # get board after random battle
        new_state = deepcopy(self.game)
        if len(mov) == 0:
            return self.game
        player = None
        for x, y, popu, xp, yp in mov:
            if player is None:
                player = new_state[x, y][0]
            else:
                if player != new_state[x, y][0]:
                    raise ValueError("Cannot move enemy's chess.")
            if new_state[x, y][1] - popu <= 0:
                del new_state[x, y]
            else:
                new_state[x, y] = (player, new_state[x, y][1] - popu)
            new_state[xp, yp] = (player, popu)
        return new_state

    def heuristic(self) -> float:
        # random battle remained to be considered
        return 0.0

    def win_draw_lose(self) -> int:
        state, player = self.game, self.next_player
        if player == Game.Werewolf:
            if state.vampire_pop() == 0 and state.werewolf_pop() > 0:
                return 1
            elif state.vampire_pop() > 0 and state.werewolf_pop() == 0:
                return -1
            return 0
        else:
            if state.vampire_pop() > 0 and state.werewolf_pop() == 0:
                return 1
            elif state.vampire_pop() == 0 and state.werewolf_pop() > 0:
                return -1
            return 0


def alpha_beta(node: Node, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[Node, float]:
    state = node.name
    if depth == 0:  # or node.is_leaf:
        return node, state.heuristic()
    if maximizing_player:
        if state.win_draw_lose() != 0:
            return node, 1e9 * state.win_draw_lose()
        v = -math.inf
        chosen_child = node
        for child in state.expand_tree():
            c = Node(child, parent=node)
            grandson, grand_v = alpha_beta(c, depth - 1, alpha, beta, False)
            if grand_v > v:
                chosen_child, v = grandson, grand_v
            if beta <= alpha:
                return chosen_child, v
            alpha = max(alpha, v)
        return chosen_child, v
    else:
        if state.win_draw_lose() != 0:
            return node, -1e9 * state.win_draw_lose()
        v = math.inf
        chosen_child = node
        for child in state.expand_tree():
            c = Node(child, parent=node)
            grandson, grand_v = alpha_beta(c, depth - 1, alpha, beta, True)
            if grand_v < v:
                chosen_child, v = grandson, grand_v
            if beta <= alpha:
                return chosen_child, v
            beta = min(beta, v)
        return chosen_child, v


def make_move(init_game: Game, who_plays: int, depth: int = 2):
    root = Node(State(init_game, who_plays))
    best_choice, alpha_beta_value = alpha_beta(root, depth=depth,
                                               alpha=-math.inf, beta=math.inf, maximizing_player=True)
    print_tree(root)
    return best_choice.name.game, ["MOV", len(best_choice.name.move), best_choice.name.move]


if __name__ == '__main__':
    game = Game(5, 5)
    game[0, 0] = (Game.Vampire, 10)
    game[4, 4] = (Game.Werewolf, 10)
    game[2, 2] = (Game.Human, 10)

    decision, command = make_move(game, Game.Werewolf)
