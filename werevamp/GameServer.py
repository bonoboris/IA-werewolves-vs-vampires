
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:21:31 2020

@author: simed
"""
from collections import defaultdict
import numpy as np
import pygame
import os

class GameServer:
    """
    player_1 = vampire
    player_2 = werewolf
    """
    def __init__(self, game, player_1, player_2, video=False):
        self.game = game
        self.players = [player_1, player_2]
        self.maxturns=25
        self.size = (500,500)
        self.cell_size = (self.size[0]/self.game.m, self.size[1]/self.game.n)
        self.screen = pygame.display.set_mode(self.size)
        self.videopath = 'D:/Ecole/3A/IA/super-secret/werevamp/videos'
        self.n_img = 0
        self.video = video
        if video:
            self.init_video()
        
        
    def init_video(self):
        pygame.init()
        videos = os.listdir(self.videopath)
        max_vid = max([int(v) for v in videos if not v.endswith('toc2')])
        self.videopath = f'{self.videopath}/{max_vid+1}'
        os.mkdir(self.videopath)
        

        
        
    def launch(self):
        index_player = 0
        self.save_img()
        for t in range(self.maxturns):
            self.play_turn(index_player)
            index_player = 1 if index_player ==0 else 0
            self.save_img()
            ended, winner = self.is_game_ended()
            if ended:
                print(f'{winner} won')
                return winner
        print('tie')
        return 'tie'
            
    def play_turn(self, index_player):
        player = self.players[index_player]
        actions = player.get_actions(self.game)
        if self.is_authorized(actions, index_player):
            self.update_game(actions, index_player)
        
    def is_authorized(self, actions, index_player):
        if len(actions) < 1:
            return False
        dict_actions = defaultdict(list)
        for action in actions:
            if len(action) != 5:
                return False
            dict_actions[(action[0],action[1])].append([(action[3], action[4]), action[2]])
        all_starts = list(dict_actions.keys())
        all_ends = [act[0] for d in dict_actions.values() for act in d]
        for start in all_starts:
            if start in all_ends:
                return False
        total_moved = 0
        for start, actions in dict_actions.items():
            if self.game[start][0] != index_player:
                return False
            all_mov = sum([act[1] for act in actions])
            total_moved += all_mov
            if all_mov > self.game[start][1]:
                return False
            for act in actions:
                end = act[0]
                if end not in self.get_valid_move(start):
                    return False
        if total_moved < 1:
            return False
        return True
    
    def update_game(self, actions, index_player):
        dict_actions_starts = defaultdict(list)
        dict_actions_ends = defaultdict(list)
        for action in actions:
            dict_actions_starts[(action[0],action[1])].append([(action[3], action[4]), action[2]])
            dict_actions_ends[(action[3],action[4])].append([(action[0], action[1]), action[2]])
        for coord, actions in dict_actions_starts.items():
            all_mov_start = sum([act[1] for act in actions])
            if all_mov_start == self.game[coord][1]:
                del self.game[coord]
            else:
                self.game[coord] = (self.game[coord][0], self.game[coord][1]-all_mov_start)
        for coord,actions in dict_actions_ends.items():
            all_mov_end = sum([act[1] for act in actions])
            if self.game[coord] != None and self.game[coord][0] != index_player:
                index_ennemy = self.game[coord][0]
                n_ennemy = self.game[coord][1]
                winner, n_winner = self.fight((index_player,all_mov_end),(index_ennemy,n_ennemy))
                self.game[coord] = (winner, n_winner)
            else:
                self.game[coord] = (index_player, all_mov_end)
            
    def fight(self, p1, p2):
        index_1, number_1 = p1
        index_2, number_2 = p2
        if index_1 ==2 and number_2 >= number_1:
            return index_2, number_2+number_1
        elif index_2 == 2 and number_1 >= number_2:
            return index_1, number_1+number_2
        elif number_2>= 1.5*number_1 and index_2 !=2:
            return index_2, number_2
        elif number_1 >= 1.5*number_2 and index_1 !=2:
            return index_1, number_1
        else:
            return self.random_battle(p1, p2)
    
    def is_game_ended(self):
        if self.game.vampire_pop() ==0 or self.game.werewolf_pop() == 0:
            winner = 1 if self.game.vampire_pop() ==0 else 0
            return True,winner
        return False,''
    
    @staticmethod
    def random_battle( p1, p2):
        index_1, number_1 = p1
        index_2, number_2 = p2
        if number_1 == number_2:
            p=0.5
        elif number_1 < number_2:
            p = number_1/(2*number_2)
        else:
            p = number_1/number_2 - 0.5
        if np.random.random() <= p:# attacker win
            n_surv = round(p*number_1)
            if index_2 == 2:
                n_conv = round(p*number_2)
            else:
                n_conv = 0
            return index_1, n_surv+n_conv
        else:#attacker lose
            n_surv = round((1-p)*number_2)
            return index_2, n_surv

                 
    def get_valid_move(self,coord):
        x,y = coord
        temp_valid = [(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)]
        final_valid = [ c for c in temp_valid if (c[1] >=0 and c[0] >=0 and c[0]<self.game.m and c[1] < self.game.n)]
        return final_valid
                
    def save_img(self):
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        font = pygame.font.SysFont('Calibri', 25, True, False)
        self.screen.fill(WHITE)
        for i in range(self.game.m+1):
            pygame.draw.line(self.screen, BLACK, [i*self.cell_size[0], 0], [i*self.cell_size[0], self.size[1]], 5)
        for i in range(self.game.n+1):
            pygame.draw.line(self.screen, BLACK, [0,i*self.cell_size[1]], [self.size[0], i*self.cell_size[1]], 5)
        for x,y,n in self.game.humans():
            rect = [x*self.cell_size[1],y*self.cell_size[0], self.cell_size[1], self.cell_size[0]]
            self.screen.fill(GREEN, rect=rect)
            self.screen.blit(font.render(str(n),False,BLACK), [x*self.cell_size[1],y*self.cell_size[0]])
        for x,y,n in self.game.vampires():
            rect = [x*self.cell_size[1],y*self.cell_size[0], self.cell_size[1], self.cell_size[0]]
            self.screen.fill(RED, rect=rect)
            self.screen.blit(font.render(str(n),False,BLACK), [x*self.cell_size[1],y*self.cell_size[0]])
        for x,y,n in self.game.werewolves():
            rect = [x*self.cell_size[1],y*self.cell_size[0], self.cell_size[1], self.cell_size[0]]
            self.screen.fill(BLUE, rect=rect)
            self.screen.blit(font.render(str(n),False,BLACK), [x*self.cell_size[1],y*self.cell_size[0]])
        pygame.image.save(self.screen, f'{self.videopath}/{self.n_img}.jpg')
        self.n_img+=1