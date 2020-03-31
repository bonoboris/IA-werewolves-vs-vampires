# -*- coding: utf-8 -*-
"""
Created on Sun Mar 1 12:21:31 2020

@author: Jiahao
"""
from .game import Game
from .GameServer import GameServer
import math
from typing import Tuple, Iterator, List, Iterable, Set
from anytree import Node, RenderTree
from copy import deepcopy
from itertools import combinations, product, chain
import random
import time


def print_tree(root):
    if root is not None:
        for pre, _, node in RenderTree(root):
            print("%s%s next player: %s" % (pre, node.name.move, node.name.next_player))  # node.name is Game instance
    else:
        raise ValueError("Tree root is empty.")


class State:
    move_vector = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]

    def __init__(self, this_board: Game, next_player: int, this_move: List[Tuple[int, int, int, int, int]] = []):
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
        # army = self.game.werewolves() if self.next_player == Game.Werewolf else self.game.vampires()
        # for possible_move in self.explore_possibilities2(list(army)):
        for possible_move in self.explore_possibilities():
            eat_human, next_turn = self.move_on_board(possible_move)
            if next_turn is not None:
                yield eat_human, State(next_turn, next_player, possible_move)

    def explore_possibilities(self, max_divide: int = 2, min_troop: int = 3) \
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
                for division in self.__division(num_divide, population):
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
                    possible_moves= list(chain(*possible_move))
                    start = set((i[0],i[1]) for i in possible_moves)
                    end = set((i[3],i[4]) for i in possible_moves)
                    if not start.intersection(end):
                        yield possible_moves

    # def explore_possibilities2(self, army: List[Tuple[int, int, int]], src: Set[Tuple[int, int]] = set(),
    #                            max_divide: int = 2, min_troop: int = 3) \
    #         -> Iterator[List[Tuple[int, int, int, int, int]]]:
    #     """
    #     Iterate on combinations of all possibilities of moves of each troop under param constraints
    #     :param src:
    #     :param army:
    #     :param max_divide: maximum number one troop can be devided into troops
    #     :param min_troop: minimum population one troop can have
    #     :return: list of moves [depart x, depart y, population, arrive x, arrive y]
    #     """
    #     max_divide = min(max_divide, 9)
    #     if len(army) == 1:
    #         x, y, population = army[0]
    #         real_num_divide = min(max(population // min_troop, 1), max_divide)
    #         for num_divide in range(1, real_num_divide + 1):
    #             for division in self.__division(num_divide, min_troop, population):
    #                 for vector_comb in combinations(State.move_vector, num_divide):
    #                     possiblility = []
    #                     for dx, dy in vector_comb:
    #                         if x + dx >= self.game.m or x + dx < 0 or\
    #                                 y + dy >= self.game.n or y + dy < 0 or (x + dx, y + dy) in src:
    #                             break
    #                         if dx == 0 and dy == 0:
    #                             possiblility.append("stay")
    #                         else:
    #                             # only consider moves here, do not consider result after random battle
    #                             possiblility.append((x, y, division[len(possiblility)], x + dx, y + dy))
    #                     if len(possiblility) == num_divide:
    #                         if "stay" in possiblility:
    #                             possiblility.remove("stay")
    #                         if possiblility:
    #                             yield possiblility
    #     else:
    #         for soldier in army:
    #             for p in self.explore_possibilities2([soldier], src, max_divide, min_troop):
    #                 cibles = [(move[3], move[4]) for move in p]
    #                 remain_army = [a for a in army if ((a[0], a[1]) not in cibles and a != soldier)]
    #                 new_src = src | {(soldier[0], soldier[1])}
    #                 for p_rest in self.explore_possibilities2(remain_army, new_src, max_divide, min_troop):
    #                     yield p + p_rest

    def __division(self, num_divide, population):
        n_div = 3-len(self.game.vampires())
        if num_divide == 1:
            yield [population]
        else:
            step = max(1,population//max(1,(n_div+1)))
            for i in range(1, n_div+1):
                v = i*step
                if v<population:
                    yield [population-v, v]

    # to complete
    def move_on_board(self, mov: List[Tuple[int, int, int, int, int]]) -> Tuple[int, Game]:
        # get board after random battle and chain prohibition
        new_state = deepcopy(self.game)
        eat_human = 0
        if len(mov) == 0:
            return eat_human, self.game
        player = None
        troop_source = set()
        for x, y, popu, xp, yp in mov:
            if player is None:
                player = self.game[x, y][0]
            else:
                if player == Game.Human:
                    print("Human is not player")
                    return 0, None
                if player != self.game[x, y][0]:
                    print("Cannot move enemy's chess.")
                    return 0, None
            if (xp, yp) in troop_source:
                print("A troop can only be source or target at the same time")
                return 0, None

            troop_source.add((x, y))
            if new_state[x, y][1] - popu <= 0:
                del new_state[x, y]
            else:
                new_state[x, y] = (player, new_state[x, y][1] - popu)
            if self.game[xp, yp] is not None:
                if player != self.game[xp, yp][0]:
                    if self.game[xp, yp][0] == Game.Human:
                        eat_human += self.game[xp, yp][1]
                    new_state[xp, yp] = self.__fake_random_battle((player, popu), self.game[xp, yp])
                else:
                    new_state[xp, yp] = (player, popu + new_state[xp, yp][1])
            else:
                new_state[xp, yp] = (player, popu)
        return eat_human, new_state

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

    def __distance_to_human(self, army1: List[Tuple[int, int, int]], harmy: List[Tuple[int, int, int]])\
            -> float:
        # try:
        #     d_all = 0
        #     for a in army1:
        #         dmin = dmax
        #         argmin = 0
        #         for k, b in enumerate(harmy):
        #             if a[2] >= b[2]:
        #                 d = self.__distance(a, b)
        #                 if d <= dmin and b[2] >= harmy[argmin][2]:
        #                     dmin = d
        #                     argmin = k
        #         d_all += (dmax - dmin) * harmy[argmin][2]
        # except IndexError:
        #     d_all = 100
        # return d_all
        d_all = 0
        # d_sum = 0
        if not harmy:
            return 100
        for h in harmy:
            d_nearest_us = math.inf
            nearest_us = army1[0]
            for u in army1:
                d = self.__distance(h, u)
                if d < d_nearest_us:
                    d_nearest_us = d
                    nearest_us = u
                elif d == d_nearest_us:
                    if u[2] > nearest_us[2]:
                        nearest_us = u
            p = nearest_us[2] / (2 * h[2]) if nearest_us[2] < h[2] else 1
            d_all += (p * p * (nearest_us[2] + h[2]) - nearest_us[2]) / d_nearest_us
            # d_sum += d_nearest_us**2
        # return d_all / math.sqrt(d_sum) if d_sum != 0 else d_all
        return d_all

    def __distance(self, a: Tuple[int,int,int], b: Tuple[int,int,int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def heuristic(self) -> int:
        # random battle remained to be considered
        # us = Game.Werewolf + Game.Vampire - self.next_player
        # our_army = self.game.vampires() if self.next_player == Game.Werewolf else self.game.werewolves()
        # their_army = self.game.vampires() if self.next_player == Game.Vampire else self.game.werewolves()
        us = Game.Vampire
        our_army = self.game.vampires()
        their_army = self.game.werewolves()
        our_army = list(our_army)
        their_army = list(their_army)
        human_army = list(self.game.humans())

        # dmax = max(self.game.size())
        heuristic = (self.game.vampire_pop() / len(our_army) - self.game.werewolf_pop() / len(their_army))  # us - them
        # heuristic = 0
        # if self.next_player == Game.Vampire:
        #     heuristic = -heuristic
        d_hum_us = self.__distance_to_human(our_army, human_army)
        d_hum_them = self.__distance_to_human(their_army, human_army)
        if d_hum_us >= d_hum_them:
            heuristic += d_hum_us
        else:
            heuristic -= d_hum_them

        # d_sum = 0
        s = 0
        for u in our_army:
            for enemy in their_army:
                p = u[2] / (2 * enemy[2]) if u[2] <= enemy[2] else min(1.0, u[2] / enemy[2] - 0.5)
                if p >= 0.7:
                    s += (enemy[2] - (1 - p) * u[2]) / self.__distance(u, enemy)
                else:
                    s -= (enemy[2] - (1 - p) * u[2]) / self.__distance(u, enemy)
                # d_sum += self.__distance(u, enemy) ** 2
        # heuristic += s / math.sqrt(d_sum) if d_sum != 0 else s
        heuristic += s
        if len(our_army) == 1:
            heuristic += our_army[0][2] * 1.5
        else:
            num_comb = len(list(combinations(our_army, 2)))
            for i in range(len(our_army) - 1):
                for j in range(i + 1, len(our_army)):
                    heuristic += (our_army[i][2] + our_army[j][2]) / self.__distance(our_army[i], our_army[j]) / num_comb
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
        for _, child in state.expand_tree():
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
        for _, child in state.expand_tree():
            c = Node(child, parent=node)
            _, potential_score = alpha_beta(c, depth - 1, alpha, beta, True)
            if potential_score < score:
                chosen_child, score = c, potential_score
            if beta <= alpha:
                return chosen_child, score
            beta = min(beta, score)
        return chosen_child, score


def random_move(state: State):
    moves_to_eat = [(num_human, c) for num_human, c in state.expand_tree() if (c.move and num_human > 0)]
    if moves_to_eat:
        move_to_eat = max(moves_to_eat, key=(lambda x: x[0]))[1]
        return move_to_eat.game, ['MOV', len(move_to_eat.move), move_to_eat.move]
    else:
        random_choice = random.choice([c for _, c in state.expand_tree() if c.move])
        return random_choice.game, ['MOV', len(random_choice.move), random_choice.move]


def make_move1(init_game: Game, who_plays: int, depth: int = 5):
    print(' in 1')
    root_state = State(init_game, who_plays)
    root = Node(root_state)
    best_choice, alpha_beta_value = alpha_beta(root, depth=depth,
                                               alpha=-math.inf, beta=math.inf, maximizing_player=True)
    #print_tree(root)
    if not best_choice.name.move:
        return random_move(root_state)
    return best_choice.name.game, ["MOV", len(best_choice.name.move), best_choice.name.move]


if __name__ == '__main__':
    # init_pop = {
    #     Game.Human: [[(2, 4), 2],[(0, 2), 2]],
    #     Game.Vampire: [[(4, 4), 10]],
    #     Game.Werewolf: [[(0, 0), 10]]
    #
    # }
    # game = Game(5, 5, init_pop)
    #
    # # s = State(game, Game.Werewolf)
    # sumtime = 0
    # for i in range(10):
    #     start = time.time()
    #     decision, command = make_move(game, Game.Vampire, depth=4)
    #     end = time.time()
    #     print(decision)
    #     print(command)
    #     sumtime += end - start
    #
    # print('time cost = %f s' % (sumtime / 10))
    #
    init_pop = {
        Game.Human: [[(2,3), 3]],
        Game.Vampire: [[(2, 4), 4], [(2,2), 4]],
        Game.Werewolf: [[(0, 0), 7]]
    }
    game = Game(5, 5, init_pop)

    decision, command = make_move1(game, Game.Vampire, depth=1)
    print(decision)
    print(command)