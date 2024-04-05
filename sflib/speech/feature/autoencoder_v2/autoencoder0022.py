# coding: utf-8
# --- メモリ食い過ぎであまりうまく行かなかった．もう少し検討が必要 ---
# マルチ解像度で32chのCNNをかけてからFull Connectで次元圧縮する
from .base import SpectrogramImageAutoEncoder
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class Floor(nn.Module):
    def __init__(self, shape):
        super(Floor, self).__init__()
        self.shape = shape
        self.weight = Parameter(torch.zeros(shape))
        self.bias = Parameter(torch.zeros(shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return self.weight.exp() * x + self.bias


class Unfloor(nn.Module):
    def __init__(self, floor_layer):
        super(Unfloor, self).__init__()
        self.weight = floor_layer.weight
        self.bias = floor_layer.bias

    def forward(self, x):
        return (x - self.bias) / self.weight.exp()


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        # バッチ軸だけ残して，フラットにする
        y = x.reshape(x.shape[0], -1)
        # ２乗する
        y = torch.pow(y, 2.0)
        # 和を取る．shape が (batch, 1) になるようにする
        y = torch.sum(y, (1, ), keepdim=True)
        # 平方根を取る
        y = torch.sqrt(y)
        # テンソルのサイズでわる
        y = y / np.prod(x.shape[1:])
        # 0になるとまずいので小さい値を足しておく
        y = y + 1e-5
        return y


class Normalize(nn.Module):
    """
    与えられた係数で正規化する
    """

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x, coef):
        coef = coef.reshape(-1, 1, 1, 1)
        return x / coef


class Denormalize(nn.Module):
    """
    与えられた係数で正規化を解除する
    """

    def __init__(self):
        super(Denormalize, self).__init__()

    def forward(self, x, coef):
        coef = coef.reshape(-1, 1, 1, 1)
        return x * coef


class SpectrogramImageAutoEncoder0022(SpectrogramImageAutoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # L2正規化
        self.l2norm = L2Norm()
        self.normalize = Normalize()
        # フロアリング
        self.floor = Floor((1, 512, 10,))
        # CNN-1 高さを1/4にするもの．
        #       512 -> 128 -> 32 -> 8 -> 2
        self.c1 = nn.Conv2d(1, 1, (4, 1), padding=0, stride=(4, 1))
        # CNN-2 高さを1/2にするもの．
        #       2 -> 1
        self.c2 = nn.Conv2d(1, 1, (2, 1), padding=0, stride=(2, 1))
        # CNN-3 高さ1の基底をかけて分解する
        self.c3 = nn.Conv2d(1, 32, (1, 10), padding=0, stride=(1, 10))
        # ここまでで，43712次元になっている予定なので，
        # Linearで圧縮していく
        self.l1 = nn.Linear(21856, 4096)
        self.l2 = nn.Linear(4096, 1024)
        self.l3 = nn.Linear(1024, 256)
        #
        self.dl3 = nn.Linear(256, 1024)
        self.dl2 = nn.Linear(1024, 4096)
        self.dl1 = nn.Linear(4096, 21856)
        # TCNN-3 高さ1の基底で逆畳み込みをする
        self.dc3 = nn.ConvTranspose2d(32, 1, (1, 10),
                                      padding=0, stride=(1, 10))
        # TCNN-2 高さを2倍にするもの
        self.dc2 = nn.ConvTranspose2d(1, 1, (2, 1), padding=0, stride=(2, 1))
        # TCNN-1 高さを4倍にするもの
        self.dc1 = nn.ConvTranspose2d(1, 1, (4, 1), padding=0, stride=(4, 1))
        # Unfloor
        self.unfloor = Unfloor(self.floor)
        # Denormalize
        self.denormalize = Denormalize()

    @property
    def bottleneck_dim(self):
        return 256

    def encode(self, x):
        l2 = self.l2norm(x)
        x = self.normalize(x, l2)
        x1 = x
        x2 = self.c1(x1)  # 512 -> 128
        x3 = self.c1(x2)  # 128 -> 32
        x4 = self.c1(x3)  # 32 -> 8
        x5 = self.c1(x4)  # 8 -> 2
        x6 = self.c2(x5)  # 2 -> 1
        x_all = torch.cat([x1, x2, x3, x4, x5, x6], dim=2)
        # x_all.shape is (batch, 1, 683, 10)
        x_all = self.c3(x_all)
        # x_all.shape is (batch, 64, 683, 1)
        x_all = torch.flatten(x_all, 1)
        # x_all.shape is (batch, 43712)
        h = torch.relu(self.l1(x_all))
        h = torch.relu(self.l2(h))
        h = torch.relu(self.l3(h))
        return h, l2

    def decode(self, x, l2):
        h = torch.relu(self.dl3(x))
        h = torch.relu(self.dl2(h))
        h = self.dl1(h)
        x_all = h.view(-1, 32, 683, 1)
        x_all = self.dc3(x_all)

        s = 0
        e = 512
        x1 = x_all[:, :, s:e, :]
        s = e
        e = e + 128
        x2 = x_all[:, :, s:e, :]
        s = e
        e = e + 32
        x3 = x_all[:, :, s:e, :]
        s = e
        e = e + 8
        x4 = x_all[:, :, s:e, :]
        s = e
        e = e + 2
        x5 = x_all[:, :, s:e, :]
        s = e
        e = e + 1
        x6 = x_all[:, :, s:e, :]
        
        x5 = x5 + self.dc2(x6)
        x4 = x4 + self.dc1(x5)
        x3 = x3 + self.dc1(x4)
        x2 = x2 + self.dc1(x3)
        x1 = x1 + self.dc1(x2)
        x = x1
        
        x = self.unfloor(x)
        x = self.denormalize(x, l2)
        return x
