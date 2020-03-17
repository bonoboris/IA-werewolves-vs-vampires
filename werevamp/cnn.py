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

BOARD_MAX_M = 50
BOARD_MAX_N = 50

class Params():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
    
    def __getitem__(self, k):
        return self.__getattribute__(k)
    
    def __setitem__(self, k, v):
        return self.__setattr__(k, v)

    def __repr__(self) -> str:
        rattrs = []
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                rv = repr(v.shape) + f" [{v.nelement()} els.]"
            else:
                rv = repr(v)
            if '\n' in rv:
                rvl = rv.split('\n')
                rvl = ['\t' + l for l in rvl]
                rv = '\n'.join(rvl)
                rattrs.append(f"{k}:\n{rv}")
            else:
                rattrs.append(f"{k}: {rv}")
        return "\n".join(rattrs)
    
    def nelement(self) -> int:
        cnt = 0
        for k,v in vars(self).items():
            if isinstance(v, torch.Tensor):
                cnt += v.nelement()
            elif isinstance(v, Params):
                cnt += v.nelement()
        return cnt

class Norm1d(nn.Module):
    def __init__(self, inplace:bool =False):
        super().__init__()
        self.inplace = inplace

    def forward(self, inputs: torch.Tensor):
        mean = inputs.mean(dim=[-1], keepdim=True)
        var = inputs.var(dim=[-1], keepdim=True)
        if self.inplace:
            inputs -= mean
            inputs /= var
            ret = inputs
        else:
            ret = (inputs - mean) / var 
        return ret


class ChannelNorm(nn.Module):
    def __init__(self, inplace:bool =False):
        super().__init__()
        self.inplace = inplace

    def forward(self, inputs: torch.Tensor):
        mean = inputs.mean(dim=[-1, -2], keepdim=True)
        var = inputs.var(dim=[-1,-2], keepdim=True)
        if self.inplace:
            inputs -= mean
            inputs /= var
            ret = inputs
        else:
            ret = (inputs - mean) / var 
        return ret


class ConstraintConv2d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, constraint: str):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size must be even")

        if constraint == "rad":
            num_params_per_channel = kernel_size // 2+1
        elif constraint in "u d l r ul ur dl dr".split(" "):
            num_params_per_channel = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.constraint = constraint
        self._params: torch.Tensor = torch.empty(out_channels, in_channels, num_params_per_channel, dtype=torch.float32)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32), requires_grad=False)
        self.init_params()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        return F.conv2d(inputs, self.weight, padding=padding)

    def init_params(self, lims = (-1, 1)) -> None:
        self._params.uniform_(*lims)
        self._compute_weights()
    
    def get_params(self) -> torch.Tensor:
        return self._params

    def set_params(self, value: np.ndarray):
        if value.shape != self._params.shape:
            raise ValueError(f"Expected a tensor with shape {self._params.shape} got a tensor with shape {value.shape}")
        self._params = torch.Tensor(value)
        self._compute_weights()

    def _compute_weights(self) -> None:
        if self.constraint == "rad":
            weight = self.values_to_sker(self._params.numpy().astype(np.float64))
        elif self.constraint in "u d l r ul ur dl dr".split(" "):
            weight = self.values_to_dker(self._params.numpy().astype(np.float64), self.constraint)
        self.weight = nn.Parameter(torch.FloatTensor(weight), requires_grad=False)

    # def cuda(self, device=None):
    #     print("custom cuda called")
    #     self.weight = self.weight.cuda(device)

    @staticmethod
    def sker_to_values(weights: torch.Tensor) -> np.ndarray:
        if weights.shape[-1] % 2 == 0:
            raise ValueError("Expected the last dimension of the weight tensor to be even.")
        
        n = weights.shape[-1] // 2 + 1
        values_shape = weights.shape[:-2] + (n,)
        values = np.empty(values_shape, dtype=np.float32)
        if len(values_shape) > 1:
            for i in range(weights.shape[0]):
                values[i] = ConstraintConv2d.sker_w_to_values(weights[i])
            return values
        
        for i in range(n):
            values[i] = weights[i, n-1].numpy()
        return values

    @staticmethod
    def dker_to_values(weights: torch.Tensor, direction) -> np.ndarray:
        if weights.shape[-1] % 2 == 0:
            raise ValueError("Expected the last dimension of the weight tensor to be even.")
        n = weights.shape[-1]
        values_shape = weights.shape[:-2] + (n,)
        values = np.empty(values_shape, dtype=np.float32)
        if len(values_shape) > 1:
            for i in range(weights.shape[0]):
                values[i] = ConstraintConv2d.dker_w_to_values(weights[i], direction)
            return values
        m = n // 2
        values[0] = weights[m, m]
        if direction in "u ul l".split(" "):
            for k in range(m):
                values[2*k+1] = weights[m-k-1, m-k-1].numpy()
                values[2*(k+1)] = weights[m+k+1, m+k+1].numpy()
        elif direction in "dl d dr".split(" "):
            for k in range(m):
                values[2*k+1] = weights[m+k+1, m].numpy()
                values[2*(k+1)] = weights[m-k-1, m].numpy()
        elif direction in "r ur".split(" "):
            for k in range(m):
                values[2*k+1] = weights[m, m+k +1].numpy()
                values[2*(k+1)] = weights[m, m-k-1].numpy()
        return values

    @staticmethod
    def values_to_sker(values: np.ndarray) -> np.ndarray:
        n = values.shape[-1]
        size = 2 * n -1
        ker_shape = values.shape[:-1] + (size, size)
        ker_w = np.empty(ker_shape, dtype=np.float32)
        if len(ker_shape) > 2:
            for i in range(values.shape[0]):
                ker_w[i] = ConstraintConv2d.values_to_sker(values[i])
            return ker_w

        for i in range(n):
            v = values[-(i+1)]
            for j in range(i, n):
                ker_w[i,j] = v
                ker_w[j,i] = v
        
        ker_w[:n,n:] = np.flip(ker_w[:n,:n-1], 1)
        ker_w[n:] = np.flip(ker_w[:n-1], 0)

                # ker_w[i,n-1:] = v
                # ker_w[i, mj] = v
                # ker_w[j, mi] = v
                # ker_w[mj, i] = v
                # ker_w[mi, j] = v
                # ker_w[mj, mi] = v
                # ker_w[mi, mj] = v
        return ker_w

    @staticmethod
    def values_to_dker(values: np.ndarray, direction:str) -> np.ndarray:
        """Create a directional kernel weights from values
        Args
        ----
            values: torch.Tensor
                A tensor which last dimension length is odd
            direction: str
                One of 'ul', 'u', 'ur', 'r', 'dr', 'd', 'dl', 'l'

        Notes
        -----
        if values = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        Returns
        
        dir='u'     dir='ur'    dir='r'
        33333       44333       44443
        41114       42113       42213
        42024       42013       42013
        42224       42224       42213
        44444       44444       44443

        """
        n = values.shape[-1]
        if n % 2 != 1:
            raise ValueError("Expected the last dimension of the weight tensor to be odd.")
        # size = (n + 1) // 2
        ker_shape = values.shape[:-1] + (n, n)
        ker_w = np.empty(ker_shape, dtype=np.float32)
        if len(ker_shape) > 2:
            for i in range(values.shape[0]):
                sub_ker = ConstraintConv2d.values_to_dker(values[i], direction)
                ker_w[i] = sub_ker
            return ker_w
        ker_w = ConstraintConv2d.values_to_sker(values[0::2])
        
        if direction == "u":
            for k, v in enumerate(values[1::2]):
                i = n // 2 - k - 1
                for j in range(i, n - i):
                    ker_w[i,j] = v
        elif direction == "d":
            for k, v in enumerate(values[1::2]):
                i = n // 2 + k + 1
                for j in range(n - i - 1, i + 1):
                    ker_w[i,j] = v
        elif direction == "l":
            for k, v in enumerate(values[1::2]):
                j = n // 2 - k - 1
                for i in range(j, n - j):
                    ker_w[i,j] = v
        elif direction == "r":
            for k, v in enumerate(values[1::2]):
                j = n // 2 + k + 1
                for i in range(n - j - 1, j + 1):
                    ker_w[i,j] = v
        elif direction == "ul":
            ul_w = ConstraintConv2d.values_to_sker(np.concatenate((values[:1],values[1::2])))
            m = n // 2 + 1
            ker_w[:m,:m] = ul_w[:m,:m]
        elif direction == "ur":
            ul_w = ConstraintConv2d.values_to_sker(np.concatenate((values[:1],values[1::2])))
            m = n // 2 + 1
            ker_w[:m,m-1:] = ul_w[:m,m-1:]
        elif direction == "dl":
            ul_w = ConstraintConv2d.values_to_sker(np.concatenate((values[:1],values[1::2])))
            m = n // 2 + 1
            ker_w[m-1:,:m] = ul_w[m-1:,:m]
        elif direction == "dt":
            ul_w = ConstraintConv2d.values_to_sker(np.concatenate((values[:1],values[1::2])))
            m = n // 2 + 1
            ker_w[m-1:,m-1:] = ul_w[m-1:,m-1:]
        return ker_w


class RadialBlock(nn.Module):
    def __init__(self, in_channels:int , out_channels:int, kernel_size:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = ConstraintConv2d(in_channels, out_channels, kernel_size, constraint='rad')
        self.norm = ChannelNorm(inplace=True)
        self.activation = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(2,2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        outputs = self.pooling(outputs)
        return outputs
    
    def init_params(self, lims=(-1, 1)):
        self.conv.init_params(lims)
    
    def get_params(self) -> nn.Parameter:
        return self.conv.get_params()
    
    def set_params(self, value: torch.Tensor) -> None:
        self.conv.set_params(value)


class OctoBlock(nn.Module):
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
        # self.pool = nn.MaxPool2d(2,2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        kernel = torch.cat([c.weight for c in self.convs])
        outputs = F.conv2d(inputs, kernel, padding=padding)
        # outputs = self.activation(outputs)
        # outputs = self.pool(outputs)
        return outputs

    
    def init_params(self, lims=(-1, 1)):
        params = torch.empty(self.get_params().shape).uniform_(*lims)
        self.set_params(params)
    
    def get_params(self) -> torch.nn.Parameter:
        return self.convs[0].get_params()
    
    def set_params(self, params: torch.Tensor):
        for c in self.convs:
            c.set_params(params)


class ConvBlock(nn.Module, ModuleSummary):

    def __init__(self):
        super().__init__()
        self.radial1 = RadialBlock(3, 10, 13)
        self.radial2 = RadialBlock(10, 3, 33)
        self.octo = OctoBlock(10, 3, 13)

    def forward(self, inputs:torch.Tensor, batch=False):
        if not batch:
            inputs.unsqueeze_(0)
        outputs = self.radial1(inputs)
        outputs = self.radial2(outputs)
        outputs = self.octo(outputs)

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

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)       

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)

    def get_params(self) -> Params:
        return Params(
            weight=self.linear.weight,
            bias=self.linear.bias
        )
    
    def set_params(self, value: Params) -> None:
        self.linear.weight = nn.Parameter(value.weight, requires_grad = False)
        if value.bias is not None:
            self.linear.bias = nn.Parameter(value.bias, requires_grad=False)


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.norm = Norm1d(inplace=True)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        return outputs
    
    def get_params(self) -> Params:
        return Params(
            weight=self.linear.weight,
            bias=self.linear.bias
        )
    
    def set_params(self, value: Params) -> None:
        self.linear.weight = nn.Parameter(value.weight, requires_grad = False)
        if value.bias is not None:
            self.linear.bias = nn.Parameter(value.bias, requires_grad=False)


class Model(nn.Module, ModuleSummary):
    def __init__(self, im_size = (50, 50), num_ally_block=10, num_human_block=20, num_enemy_block=10, out_unit_block_features=10, out_hidden_features=100):
        super().__init__()
        self.num_ally_block = num_ally_block
        self.num_enemy_block = num_enemy_block
        self.num_human_block = num_human_block
        
        self.out_unit_block_features = out_unit_block_features
        self.out_hidden_features = out_hidden_features

        self.conv_block = ConvBlock()
        conv_output_size = (self.conv_block.octo.out_channels * 12 * 12)
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
    def from_params(cls, params) -> "Model":
        inst = cls()
        inst.set_params(params)
        return inst


class PlayerCNN():
    def __init__(self, player:int, model:Optional[Model] = None, cuda_device=None):
        self.player = player
        self.opponent = Game.Vampire if player == Game.Werewolf else Game.Werewolf
        self.model = model or Model()
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
    model = Model()
    params = model.get_params()
    print(params)
    print("Num free parameters:", params.nelement())

