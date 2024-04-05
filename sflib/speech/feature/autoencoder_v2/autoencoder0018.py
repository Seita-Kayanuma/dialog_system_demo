# coding: utf-8
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


class SpectrogramImageAutoEncoder0018(SpectrogramImageAutoEncoder):
    def __init__(self, *args, **kwargs):
        super(SpectrogramImageAutoEncoder0018, self).__init__(*args, **kwargs)

        # L2正規化
        self.l2norm = L2Norm()
        self.normalize = Normalize()
        # フロアリング
        self.floor = Floor((
            1,
            512,
            10,
        ))
        # CNN-1 (512, 10) -> (256, 5)
        self.c1 = nn.Conv2d(1, 32, (5, 5), padding=2, stride=(2, 2))
        self.bnc1 = nn.BatchNorm2d(32)
        # CNN-2 (256, 5) -> (86, 3)
        self.c2 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bnc2 = nn.BatchNorm2d(32)
        # CNN-3 (86, 3) -> (29, 2)
        self.c3 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bnc3 = nn.BatchNorm2d(32)
        # CNN-4 (29, 2) -> (10, 1)
        self.c4 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bnc4 = nn.BatchNorm2d(32)
        # CNN-5 (10, 1) -> (4, 1)
        self.c5 = nn.Conv2d(32, 64, (7, 1), padding=(3, 0), stride=(3, 1))
        self.bnc5 = nn.BatchNorm2d(64)
        # CNN-6 (4, 1) -> (1, 1)
        self.c6 = nn.Conv2d(64, 128, (4, 1), padding=(0, 0), stride=(1, 1))
        self.bnc6 = nn.BatchNorm2d(128)
        # Inverse CNN-6 (1, 1) -> (4, 1)
        self.dc6 = nn.ConvTranspose2d(128,
                                      64, (4, 1),
                                      padding=(0, 0),
                                      stride=(1, 1))
        # Inverse CNN-5 (4, 1) -> (10, 1)
        self.bndc5 = nn.BatchNorm2d(64)
        self.dc5 = nn.ConvTranspose2d(64,
                                      32, (7, 1),
                                      padding=(0, 0),
                                      stride=(1, 1))
        # Inverse CNN-4 (10, 1) -> (29, 2)
        self.bndc4 = nn.BatchNorm2d(32)
        self.dc4 = nn.ConvTranspose2d(32,
                                      32, (8, 6),
                                      padding=(3, 2),
                                      stride=(3, 2))
        # Inverse CNN-3 (29, 2) -> (86, 3)
        self.bndc3 = nn.BatchNorm2d(32)
        self.dc3 = nn.ConvTranspose2d(32,
                                      32, (8, 5),
                                      padding=(3, 2),
                                      stride=(3, 2))
        # Inverse CNN-2 (86, 3) -> (256, 5)
        self.bndc2 = nn.BatchNorm2d(32)
        self.dc2 = nn.ConvTranspose2d(32,
                                      32, (7, 5),
                                      padding=(3, 2),
                                      stride=(3, 2))
        # Inverse CNN-1 (256, 5) -> (256, 5)
        self.bndc1 = nn.BatchNorm2d(32)
        self.dc1 = nn.ConvTranspose2d(32,
                                      1, (6, 6),
                                      padding=(2, 2),
                                      stride=(2, 2))
        # Unfloor
        self.unfloor = Unfloor(self.floor)
        # Denormalize
        self.denormalize = Denormalize()

    @property
    def bottleneck_dim(self):
        return 128

    def encode(self, x):
        l2 = self.l2norm(x)
        x = self.normalize(x, l2)
        x = F.relu(self.floor(x))
        x = F.relu(self.c1(x))
        x = self.bnc1(x)
        x = F.relu(self.c2(x))
        x = self.bnc2(x)
        x = F.relu(self.c3(x))
        x = self.bnc3(x)
        x = F.relu(self.c4(x))
        x = self.bnc4(x)
        x = F.relu(self.c5(x))
        x = self.bnc5(x)
        x = self.c6(x)
        x = self.bnc6(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.tanh(x)
        return x, l2

    def decode(self, x, l2):
        x = x.reshape(-1, 128, 1, 1)
        x = F.relu(self.dc6(x))
        x = self.bndc5(x)
        x = F.relu(self.dc5(x))
        x = self.bndc4(x)
        x = F.relu(self.dc4(x))
        x = self.bndc3(x)
        x = F.relu(self.dc3(x))
        x = self.bndc2(x)
        x = F.relu(self.dc2(x))
        x = self.bndc1(x)
        x = self.dc1(x)
        x = self.unfloor(x)
        x = self.denormalize(x, l2)
        return x
