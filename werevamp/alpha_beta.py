# -*- coding: utf-8 -*-
"""
Created on Sun Mar 1 12:21:31 2020

@author: Jiahao
"""
from game import Game
import math
from typing import Tuple, Iterator, List, Iterable
from anytree import Node, RenderTree
from copy import deepcopy
from itertools import combinations, product, chain
import numpy as np


def print_tree(root):
    if root is not None:
        for pre, _, node in RenderTree(root):
            print("%s%s next player: %s" % (pre, node.name.move, node.name.next_player))  # node.name is Game instance
    else:
        raise ValueError("Tree root is empty.")


class State:
    move_vector = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]

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
            if next_turn is not None:
                yield State(next_turn, next_player, possible_move)

    def explore_possibilities(self, max_divide: int = 3, min_troop: int = 3) \
            -> Iterator[List[Tuple[int, int, int, int, int]]]:
        """
        Iterate on combinations of all possibilities of moves of each troop under param constraints
        :param max_divide: maximum number one troop can be devided into troops
        :param min_troop: minimum population one troop can have
        :return: list of moves [depart x, depart y, population, arrive x, arrive y]
        """
        max_divide = min(max_divide, 9)
        army = self.game.werewolves() if self.next_player == Game.Werewolf else self.game.vampires()
        potential_of_whole_army = [[] for _ in range(len(army))]
        for i, (x, y, population) in enumerate(army):
            for num_divide in range(1, max_divide + 1):
                if population // min_troop < num_divide:  # cut into too large troops
                    continue
                for division in self.__division(num_divide, min_troop, population):
                    for vector_comb in combinations(State.move_vector, num_divide):
                        possiblility = []
                        for dx, dy in vector_comb:
                            if x + dx >= self.game.m or x + dx < 0 or y + dy >= self.game.n or y + dy < 0:
                                break
                            if dx == 0 and dy == 0:
                                possiblility.append("stay")
                                # print("{0}'s potential move: {1} soldier stay still".format(
                                #     self.next_player, division[len(possiblility)-1]))
                            else:
                                # only consider moves here, do not consider result after random battle
                                # no chain prohibition neither
                                possiblility.append((x, y, division[len(possiblility)], x + dx, y + dy))
                                # print("{0}'s potential move: {1}".format(
                                #     self.next_player, [(x, y, division[len(possiblility)-1], x + dx, y + dy)]))
                        if len(possiblility) == num_divide:
                            if "stay" in possiblility:
                                possiblility.remove("stay")
                            if possiblility:
                                potential_of_whole_army[i].append(possiblility)
        for num_moving in range(1, len(army) + 1):
            for comb_possibility in combinations(potential_of_whole_army, num_moving):
                for possible_move in product(*comb_possibility):
                    yield list(chain(*possible_move))

    def __division(self, num_divide, min_troop, population):
        if num_divide == 1:
            yield [population]
        else:
            for troop_pop in range(min_troop, population - min_troop * (num_divide - 1) + 1):
                residual_troops = self.__division(num_divide - 1, min_troop, population - troop_pop)
                for residual_troop in residual_troops:
                    yield [troop_pop] + residual_troop

    # to complete
    def move_on_board(self, mov: List[Tuple[int, int, int, int, int]]) -> Game:
        # get board after random battle and chain prohibition
        new_state = deepcopy(self.game)
        if len(mov) == 0:
            return self.game
        player = None
        troop_source = []
        for x, y, popu, xp, yp in mov:
            if player is None:
                player = self.game[x, y][0]
            else:
                if player == game.Human:
                    print("Human is not player")
                    return None
                if player != self.game[x, y][0]:
                    print("Cannot move enemy's chess.")
                    return None
            if (xp, yp) in troop_source:
                print("A troop can only be source or target at the same time")
                return None

            if new_state[x, y][1] - popu <= 0:
                del new_state[x, y]
            else:
                new_state[x, y] = (player, new_state[x, y][1] - popu)
            if self.game[xp, yp] is not None:
                if player != self.game[xp, yp][0]:
                    new_state[xp, yp] = self.__fake_random_battle((player, popu), self.game[xp, yp])
                else:
                    new_state[xp, yp] = (player, popu + new_state[xp, yp][1])
            else:
                new_state[xp, yp] = (player, popu)
            troop_source.append((x, y))
        return new_state

    def __fake_random_battle(self, attacker: Tuple[int, int], defender: Tuple[int, int], threshold: float = 0.6) \
            -> Tuple[int, int]:
        if defender[0] == Game.Human and attacker[1] >= defender[1]:
            return attacker[0], attacker[1] + defender[1]
        elif defender[0] != Game.Human and attacker[1] >= 1.5 * defender[1]:
            return attacker[0], attacker[1]
        else:
            p = attacker[1] / (2 * defender[1]) if attacker[1] <= defender[1] else attacker[1] / defender[1] - 0.5
            if p >= threshold:  # attacker win
                n_surv = math.floor(p * attacker[1])
                n_conv = math.floor(p * defender[1]) if defender[0] == Game.Human else 0
                return attacker[0], n_surv + n_conv
            else:  # attacker lose
                n_surv = math.floor((1 - p) * defender[1])
                return defender[0], n_surv

    def __distance_to_human(self, army1: List[Tuple[int, int, int]], harmy: List[Tuple[int, int, int]], dmax: int)\
            -> int:
        try:
            d_all = 0
            for a in army1:
                dmin = dmax
                argmin = 0
                for k, b in enumerate(harmy):
                    if a[2] >= b[2]:
                        d = self.__distance(a, b)
                        if d <= dmin and b[2] >= harmy[argmin][2]:
                            dmin = d
                            argmin = k
                d_all += (dmax - dmin) * harmy[argmin][2]
        except:
            d_all = 0
        return d_all

    def __distance(self, a: Tuple[int,int,int], b: Tuple[int,int,int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def heuristic(self) -> int:
        # random battle remained to be considered
        us = Game.Werewolf + Game.Vampire - self.next_player
        our_army = self.game.vampires() if self.next_player == Game.Werewolf else self.game.werewolves()
        their_army = self.game.vampires() if self.next_player == Game.Vampire else self.game.werewolves()
        our_army = list(our_army)
        their_army = list(their_army)
        human_army = list(self.game.humans())

        dmax = max(self.game.size())
        heuristic = dmax * 30 * (self.game.vampire_pop() - self.game.werewolf_pop())  # us - them
        if self.next_player == Game.Vampire:
            heuristic = -heuristic
        d_hum_us = self.__distance_to_human(our_army, human_army, dmax)
        d_hum_them = self.__distance_to_human(their_army, human_army, dmax)
        heuristic += 20 * (3 * d_hum_us - d_hum_them)
        for u in our_army:
            d_min = dmax
            enemy_min = (0, 0, 0)
            for enemy in their_army:
                d = self.__distance(enemy, u)
                if d < d_min:
                    d_min = d
                    enemy_min = enemy
            winner1, n_survive1 = self.__fake_random_battle((us, u[2]), (self.next_player, enemy_min[2]))
            winner2, n_survive2 = self.__fake_random_battle((self.next_player, enemy_min[2]), (us, u[2]))
            f0 = 0
            if (winner1, n_survive1) == (us, u[2]):
                f0 = 1
            elif (winner2, n_survive2) == (self.next_player, enemy_min[2]):
                f0 = -1
            f1 = 1 if winner1 == us else -1
            f2 = 1 if winner2 == us else -1
            heuristic += 5 * (f0 * 10 * u[2] + f1 * n_survive1 + f2 * n_survive2) * (dmax - d_min)

        return heuristic

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
        score = -math.inf
        chosen_child = node
        for child in state.expand_tree():
            c = Node(child, parent=node)
            _, potential_score = alpha_beta(c, depth - 1, alpha, beta, False)
            if potential_score > score:
                chosen_child, score = c, potential_score
            if beta <= alpha:
                return chosen_child, score
            alpha = max(alpha, score)
        return chosen_child, score
    else:
        if state.win_draw_lose() != 0:
            return node, -1e9 * state.win_draw_lose()
        score = math.inf
        chosen_child = node
        for child in state.expand_tree():
            c = Node(child, parent=node)
            _, potential_score = alpha_beta(c, depth - 1, alpha, beta, True)
            if potential_score < score:
                chosen_child, score = c, potential_score
            if beta <= alpha:
                return chosen_child, score
            beta = min(beta, score)
        return chosen_child, score


def make_move(init_game: Game, who_plays: int, depth: int = 2):
    root = Node(State(init_game, who_plays))
    best_choice, alpha_beta_value = alpha_beta(root, depth=depth,
                                               alpha=-math.inf, beta=math.inf, maximizing_player=True)
    print_tree(root)
    return best_choice.name.game, ["MOV", len(best_choice.name.move), best_choice.name.move]


if __name__ == '__main__':
    init_pop = {
        Game.Human: [[(2, 2), 10]],
        Game.Vampire: [[(1, 2), 10]],
        Game.Werewolf: [[(3, 4), 3], [(3, 3), 3], [(4, 3), 4]]
    }
    game = Game(5, 5, init_pop)

    # s = State(game, Game.Werewolf)

    decision, command = make_move(game, Game.Vampire)
    print(decision)
    print(command)
