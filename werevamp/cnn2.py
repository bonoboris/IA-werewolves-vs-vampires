    # from __future__ import annotations

from typing import Sequence, List, Tuple, Dict, Union, NamedTuple, Optional
from collections import namedtuple
import random as rd
from time import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax

from .str_utils import ModuleSummary, TensorShapeRepr
from .game import Action, Game
from .utils import sigmoid
from .cnn import BOARD_MAX_M, BOARD_MAX_N, Params, LinearBlock, LinearLayer, ChannelNorm, RadialBlock, ConstraintConv2d
from .game_gen import SymGameGenerator


class OctoBlock2(nn.Module):
    def __init__(self, in_channels:int, out_channels_per_dir:int, kernel_size:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 8*out_channels_per_dir
        self.kernel_size = kernel_size

        self.convs = nn.ModuleList()
        for direction in "ul u ur r dr d dl l".split():
            self.convs.append(ConstraintConv2d(in_channels, out_channels_per_dir, kernel_size, constraint=direction))

        self.norm = ChannelNorm(inplace=True)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        kernel = torch.cat([c.weight for c in self.convs])
        outputs = F.conv2d(inputs, kernel, padding=padding)
        outputs = self.activation(outputs)
        outputs = self.pool(outputs)
        return outputs

    
    def init_params(self, lims=(-1, 1)):
        params = torch.empty(self.get_params().shape).uniform_(*lims)
        self.set_params(params)
    
    def get_params(self) -> torch.nn.Parameter:
        return self.convs[0].get_params()
    
    def set_params(self, params: torch.Tensor):
        for c in self.convs:
            c.set_params(params)


class ConvBlock2(nn.Module, ModuleSummary):

    def __init__(self):
        super().__init__()
        self.radial1 = RadialBlock(3, 10, 13)
        self.octo = OctoBlock2(10, 3, 13)
        self.radial2 = RadialBlock(3*8, 3, 13)

    def forward(self, inputs:torch.Tensor, batch=False):
        if not batch:
            inputs.unsqueeze_(0)
        outputs = self.radial1(inputs)
        outputs = self.octo(outputs)
        outputs = self.radial2(outputs)

        if not batch:
            outputs.squeeze_(0)
        return outputs

    def init_params(self, lims=(-1, 1)):
        t0 = time()
        self.radial1.init_params(lims)
        # print(f"Radial1 initialisation: {time() - t0:.2f} s")
        t0 = time()
        self.radial2.init_params(lims)
        # print(f"Radial2 initialisation: {time() - t0:.2f} s")
        t0 = time()
        self.octo.init_params(lims)
        # print(f"Octo initialisation: {time() - t0:.2f} s")
    
    def get_params(self) -> Params:
        return Params(
            radial1=self.radial1.get_params(),
            radial2=self.radial2.get_params(),
            octo=self.octo.get_params()
        )
    
    def set_params(self, value: Params):
        self.radial1.set_params(value.radial1)
        self.radial2.set_params(value.radial2)
        self.octo.set_params(value.octo)


class Model2(nn.Module, ModuleSummary):
    def __init__(self, im_size = (50, 50), num_ally_block=10, num_human_block=20, num_enemy_block=10, out_unit_block_features=5, out_hidden_features=20):
        super().__init__()
        self.num_ally_block = num_ally_block
        self.num_enemy_block = num_enemy_block
        self.num_human_block = num_human_block
        
        self.out_unit_block_features = out_unit_block_features
        self.out_hidden_features = out_hidden_features

        self.conv_block = ConvBlock2()
        conv_output_size = (self.conv_block.radial2.out_channels * 6 * 6)
        self.ally_conv_block = LinearBlock(3 + conv_output_size, out_unit_block_features)
        self.enemy_conv_block = LinearBlock(3 + conv_output_size, out_unit_block_features)
        self.human_conv_block = LinearBlock(3 + conv_output_size, out_unit_block_features)
        self.hidden = LinearBlock(out_unit_block_features * (num_ally_block + num_enemy_block + num_human_block), out_hidden_features)
        self.pred_layer = LinearLayer(2 + 3 + out_hidden_features, out_features=1+8, bias=True)
    
    def get_params(self) -> Params:
        return Params(
            conv_block=self.conv_block.get_params(),
            ally_block=self.ally_conv_block.get_params(),
            enemy_block=self.enemy_conv_block.get_params(),
            human_block=self.human_conv_block.get_params(),
            hidden=self.hidden.get_params(),
            pred=self.pred_layer.get_params()
        )

    def set_params(self, value: Params):
        self.conv_block.set_params(value.conv_block)
        self.ally_conv_block.set_params(value.ally_block)
        self.enemy_conv_block.set_params(value.enemy_block)
        self.human_conv_block.set_params(value.human_block)
        self.hidden.set_params(value.hidden) 
        self.pred_layer.set_params(value.pred)

    def forward(self, ally_units:Sequence[torch.Tensor], enemy_units:Sequence[torch.Tensor], human_units: Sequence[torch.Tensor],
                game_image: torch.Tensor, ally_borders: torch.Tensor) -> List[torch.Tensor]:
        # print("Game im shape", game_image.shape)
        conv_output = self.conv_block(game_image).view(-1)
        # print("Conv out shape", conv_output.shape)

        nblock = self.num_ally_block + self.num_enemy_block + self.num_human_block
        block_nfeat = self.out_unit_block_features 
        hidden_input = torch.zeros(nblock * block_nfeat, dtype=torch.float32, device=self.device)

        for i, unit in enumerate(ally_units):
            block_output = self.ally_conv_block(torch.cat((unit, conv_output)))
            hidden_input[i*block_nfeat: (i+1)*block_nfeat] = block_output

        off = self.num_ally_block * block_nfeat
        for i, unit in enumerate(enemy_units):
            block_output = self.enemy_conv_block(torch.cat((unit, conv_output)))
            hidden_input[off + i*block_nfeat: off + (i+1)*block_nfeat] = block_output

        off += self.num_enemy_block * block_nfeat
        for i, unit in enumerate(human_units):
            block_output = self.human_conv_block(torch.cat((unit, conv_output)))
            hidden_input[off + i*block_nfeat: off + (i+1)*block_nfeat] = block_output

        hidden_output = self.hidden(hidden_input)
        outputs = list()
        for border, unit in zip(ally_borders, ally_units):
            pred_input = torch.cat((border, unit, hidden_output))
            outputs.append((unit, border, self.pred_layer(pred_input)))
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
    
    @classmethod
    def from_params(cls, params) -> "Model2":
        inst = cls()
        inst.set_params(params)
        return inst


class PlayerCNN2():
    def __init__(self, player:int, model:Optional[Model2] = None, cuda_device=None):
        self.player = player
        self.opponent = Game.Vampire if player == Game.Werewolf else Game.Werewolf
        self.model = model or Model2()
        self.model.eval()
        self.model.conv_block.init_params()
        self.cuda_device = cuda_device
        self.model = self._cuda_wrap(self.model)

        self.timers = {
            "prepare": 0,
            "predict": 0,
            "decode": 0
        }
        self.ntimed = 0
    
    def play(self, game: Game) -> Sequence[Action]:
        t0 = time()
        inputs = self.prepare_inputs(game)
        t1 = time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        t2 = time()
        ret = self.decode_outputs(outputs)
        t3 = time()
        self.timers["prepare"] += t1 - t0
        self.timers["predict"] += t2 - t1
        self.timers["decode"] += t3 - t2
        self.ntimed += 1
        return ret

    def prepare_inputs(self, game: Game) -> Dict:
        ally_units = []
        ally_borders = []
        for unit in game.units(self.player):
            ally_units.append(self._cuda_wrap(torch.FloatTensor(unit)))
            ally_borders.append(self._cuda_wrap(torch.FloatTensor(self._border(unit[:2], game.size()))))
        enemy_units = [self._cuda_wrap(torch.FloatTensor(unit)) for unit in game.units(self.opponent)]
        human_units = [self._cuda_wrap(torch.FloatTensor(unit)) for unit in game.units(Game.Human)]
        game_im = torch.zeros(3, BOARD_MAX_M, BOARD_MAX_N)
        game_mat = torch.FloatTensor(game.to_matrix_CHW()) 
        game_im[:game_mat.shape[0],:game_mat.shape[1], :game_mat.shape[2]] = game_mat
        game_im = self._cuda_wrap(game_im)
        
        return {
            "ally_units": ally_units,
            "human_units": human_units,
            "enemy_units": enemy_units,
            "game_image": game_im,
            "ally_borders": ally_borders        
        }

    def decode_outputs(self, outputs:Sequence[Tuple[torch.Tensor,torch.Tensor]]) -> Sequence[Action]:
        actions = list()
        max_num_val, max_num_idx, num0_val = -np.inf, -1, 0
        for idx, (unit, border, out) in enumerate(outputs):
            i0, j0, num0 = np.round(unit.cpu().numpy()).astype(int)
            i0, j0 = np.round([i0, j0]).astype(int)
            bi, bj = border.cpu().numpy()
            out = out.cpu().tolist()

            num = out[0]
            if num > max_num_val:
                max_num_val = num
                max_num_idx = idx
                num0_val = num0

            num = 1.25 * sigmoid(num) * num0
            num = min(num0, num)
            num = round(float(num))
            di, dj = self._process_dir_out(out[1:], border.cpu().numpy())
            a = Action(i0, j0, i0 + di, j0 + dj, num)
            actions.append(a)

        # Correcting actions
        # Removing action moving zero units
        actions_ = [a for a in actions if a.num > 0]
        # If ally merging remove the other case action if existing
        starts = {a.start() for a in actions}
        dests = {a.dest() for a in actions}
        both = starts.intersection(dests)
        # print("Both:", both)
        actions_ = [a for a in actions_ if a.start() not in both]

        # Selecting a square to move if no remaining actions 
        if not actions_:
            a = actions[max_num_idx]
            actions_.append(Action(a.start_i, a.start_j, a.dest_i, a.dest_j, num0_val))
        # print("Before ", actions)
        # print("After ", actions_)
        return actions_
    
    def _cuda_wrap(self, obj):
        device = None
        if isinstance(self.cuda_device, str):
            device = self.cuda_device
        if self.cuda_device:
            obj = obj.cuda(device)
        return obj

    @staticmethod
    def _border(coords, shape):
        i,j = coords
        m,n = shape
        ret = [0,0]
        
        if i == 0: ret[0] = -1
        elif i == m-1: ret[0] = 1
        
        if j == 0: ret[1] = -1
        elif j == n-1: ret[1] = 1
        return ret

    @staticmethod
    def _process_dir_out(dir_vals, borders):
        bi, bj = np.round(borders).astype(int)

        dir_clf = softmax(dir_vals)
        if bi == -1:
            dir_clf[[0,1,2]] = -np.inf
        elif bi == 1:
            dir_clf[[4,5,6]] = -np.inf
        if bj == 1:
            dir_clf[[2,3,4]] = -np.inf
        elif bj == -1:
            dir_clf[[6,7,0]] = -np.inf
        dir_val = np.argmax(dir_clf)
        
        di, dj = 0, 0
        if dir_val in (0,1,2):
            di = -1
        elif dir_val in (4,5,6):
            di = 1
        
        if dir_val in (2,3,4):
            dj = 1
        elif dir_val in (6,7,0):
            dj = -1
        return di, dj


if __name__ == "__main__":
    gg = SymGameGenerator(10, 10, players_pop=100, human_pop=100, human_spread=4)
    p = PlayerCNN2(Game.Vampire)
    print(p.play(gg()))
    print(p.model.get_params().nelement())
    print(p.model.get_params())
    
