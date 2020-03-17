from typing import Tuple, Union, Sequence
from copy import copy, deepcopy
import random as rd
from os import path
from time import time

import torch
import numpy as np
import tqdm
from tabulate import tabulate

from .cnn import Model, Params, PlayerCNN
from .game import Game, GameRunner
from .game_gen import GameGenerator, SymGameGenerator


def empty_params(model) -> Params:
    p = Params()
    for k, v in vars(model).items():
        if isinstance(v, Params):
            p[k] = empty_params(v)
        else:
            p[k] = None
    return p


def deep_get(dct, keys):
    if len(keys) == 1:
        return dct[keys[0]]
    else:
        return deep_get(dct[keys[0]], keys[1:])


def deep_set(dct, keys, val):
    if len(keys) == 1:
        dct[keys[0]] = val
        return dct
    else:
        return deep_set(dct[keys[0]], keys[1:], val)


def crossover(models, weights=None, granularity="coarse") -> Model:
    params = [m.get_params() for m in models]
    child_params = empty_params(params[0])
    
    if weights is not None:
        weights = np.asarray(weights)
        weights = weights / weights.sum()

    if granularity == "coarse":
        groups = [
            ["conv_block"],
            ["human_block", "enemy_block", "ally_block"],
            ["hidden"],
            ["pred"]
        ]
        for g in groups:
            p = rd.choices(params, weights=weights, k=1)[0]
            for k in g:
                child_params[k] = deepcopy(p[k]) 
    
    elif granularity == "fine":
        groups = [
            ["conv_block.radial1"],
            ["conv_block.radial2"],
            ["conv_block.octo"],
            ["human_block"],
            ["enemy_block"],
            ["ally_block"],
            ["hidden"],
            ["pred"]
        ]
        for g in groups:
            p = rd.choices(params, weights=weights, k=1)
            for k in g:
                ks = k.split(".")
                deep_set(child_params, ks, deep_get(p, ks))
    return Model.from_params(child_params)


def mutate(model: Model, mutation_rate:float, group_mutation_rate:float) -> Model:
    params = deepcopy(model.get_params())
    groups = [
        ["conv_block.radial1"],
        ["conv_block.radial2"],
        ["conv_block.octo"],
        ["human_block"],
        ["enemy_block"],
        ["ally_block"],
        ["hidden"],
        ["pred"]
    ]

    for g in groups:
        alt = rd.random()
        if alt > group_mutation_rate:
            continue
        else:
            for k in g:
                ks = k.split(".")
                sub_param = deep_get(params, ks)
                sub_param = _mutatate(sub_param, mutation_rate)
                deep_set(params, ks, sub_param)
    
    return Model.from_params(params) 


def _mutatate(params: Union[Params, torch.Tensor], mutation_rate: float):
    if isinstance(params, Params):
        for k, v in vars(params).items():
            params[k] = _mutatate(v, mutation_rate)
    elif isinstance(params, torch.Tensor):
        flat = params.view(-1)
        nb_el = len(flat)
        nb_mut = round(float(nb_el * mutation_rate))
        mut_idxs = rd.sample(range(nb_el), nb_mut)
        mut_val = torch.empty(nb_mut).uniform_(-1, 1)
        for i, v in zip(mut_idxs, mut_val):
            flat[i] = v

    return params


# -----------------------------------------------


def score_game(last_state: Union[Game, Sequence[Game]]):
    if isinstance(last_state, Game):
        return {
            Game.Vampire: last_state.vampire_pop() - last_state.werewolf_pop(),
            Game.Werewolf: last_state.werewolf_pop() - last_state.vampire_pop(),
        }
    else:
        dct = {
            Game.Vampire: 0,
            Game.Werewolf: 0,
        }
        for el in last_state:
            res = score_game(el)
            dct[Game.Vampire] += res[Game.Vampire]
            dct[Game.Werewolf] += res[Game.Werewolf]
        return dct


def run_multiple_games(game_gen, p1, p2, ngames, num_max_turns):
    results = []
    gr = GameRunner()
    for i in range(ngames):
        results.append(gr.run(game_gen(), p1, p2, bool(i%2), num_max_turns))
    return results


def rd_population(Npop):
    return [Model() for _ in range(Npop)]


def save_generation(num, population):
    fpath = path.join(path.dirname(path.relpath(__file__)), f"data/gen{num}.pt")
    torch.save(population, fpath)


def evolve(population, fitness, num_parent=5, mutation_rate=0.1, group_mutation_rate=0.5):
    n_pop = len(population)
    new_pop = []
    for i in range(n_pop):
        sample = rd.choices(range(n_pop), weights=fitness, k=num_parent)
        parents = [population[i] for i in sample]
        parents_fitness = [fitness[i] for i in sample]
        child = crossover(parents, parents_fitness)
        child = mutate(child, mutation_rate, group_mutation_rate)
        new_pop.append(child)
    return new_pop


def get_fitness(scores):
    scores = np.asarray(scores, dtype=float)
    scores = scores - scores.min()
    scores = scores / scores.sum()
    return scores


def evolution(game_gen, ngames, max_turns, num_generation, population,
              num_parent=5, mutation_rate=0.1, group_mutation_rate=0.5, cuda_device=True):
    n_pop = len(population)
    for gen in range(1, num_generation + 1):
        print(f"Starting match for generation {gen}")
        score_list = np.zeros(n_pop, dtype=int)
        tbar = tqdm.tqdm(total=ngames*n_pop*(n_pop-1)//2, unit="game")
        players = [PlayerCNN(Game.Vampire, population[k], cuda_device) for k in range(n_pop)]
        for i in range(n_pop):
            for j in range(i + 1, n_pop): 
                players[i].player = Game.Vampire
                players[j].player = Game.Werewolf
                if rd.randint(0, 1):
                    res = run_multiple_games(game_gen, players[i], players[j], ngames, max_turns)
                else:
                    res = run_multiple_games(game_gen, players[j], players[i], ngames, max_turns)
                scores = score_game(res)
                score_list[i] += scores[Game.Vampire]
                score_list[j] += scores[Game.Werewolf]
                tbar.update(ngames)
        print()
        print("Scores")
        print(score_list)
        print()
        fitness = get_fitness(score_list)
        population = evolve(population, fitness, num_parent, mutation_rate, group_mutation_rate)
        save_generation(gen, population)


if __name__ == "__main__":
    gg = SymGameGenerator(15, 15, 50, 50, 4, sym='r')
    pop = rd_population(20)
    save_generation(0, pop)

    evolution(
        game_gen=gg,
        ngames=6,
        max_turns=50,
        num_generation=20,
        population=pop
    )
