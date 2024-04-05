# coding: utf-8
# ResNetのBasicBlock+MaxPoolingでEncode
# Upsample+ResNetのBasicBlockでDecodeするバージョン
# デコーダの最後は，BasicBlockではなくて普通のConvolutionにする（ReLUによってスパースにしないため）
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


# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#  (BottleNeckとの共通性を維持するためのものは落とします）
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class SpectrogramImageAutoEncoder0021(SpectrogramImageAutoEncoder):
    def __init__(self, *args, **kwargs):
        super(SpectrogramImageAutoEncoder0021, self).__init__(*args, **kwargs)

        # L2正規化
        self.l2norm = L2Norm()
        self.normalize = Normalize()
        # フロアリング
        self.floor = Floor((1, 512, 10,))
        # CNN-1 サイズをいじらずChannelを増加させるだけ
        self.conv1 = nn.Conv2d(1, 32, (1, 1), padding=(0, 0), stride=(1, 1))
        # BasicBlock-1 (512, 10) -> (256, 5)
        self.bb1 = BasicBlock(32, 32)
        self.mp1 = nn.MaxPool2d((5, 5), padding=2, stride=(2, 2))
        # BasicBlock-2 (256, 5) -> (86, 3)
        self.bb2 = BasicBlock(32, 32)
        self.mp2 = nn.MaxPool2d((7, 5), padding=(3, 2), stride=(3, 2))
        # CNN-3 (86, 3) -> (29, 2)
        self.bb3 = BasicBlock(32, 32)
        self.mp3 = nn.MaxPool2d((7, 5), padding=(3, 2), stride=(3, 2))
        # CNN-4 (29, 2) -> (10, 1)
        self.bb4 = BasicBlock(32, 32)
        self.mp4 = nn.MaxPool2d((7, 5), padding=(3, 2), stride=(3, 2))
        # FC-1
        self.fc1 = nn.Linear(320, 256)
        # Inverse FC-1
        self.dfc1 = nn.Linear(256, 320)
        # Inverse CNN-4 (10, 1) -> (29, 2)
        self.up4 = nn.Upsample((29, 2))
        self.ibb4 = BasicBlock(32, 32)
        # Inverse CNN-3 (29, 2) -> (86, 3)
        self.up3 = nn.Upsample((86, 3))
        self.ibb3 = BasicBlock(32, 32)
        # Inverse CNN-2 (86, 3) -> (256, 5)
        self.up2 = nn.Upsample((256, 5))
        self.ibb2 = BasicBlock(32, 32)
        # Inverse CNN-1 (256, 5) -> (512, 10)
        self.up1 = nn.Upsample((512, 10))
        # self.ibb1 = BasicBlock(32, 32)
        self.iconv1 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        # Final Convolution
        self.iconv0 = nn.Conv2d(32, 1, (1, 1), padding=(0, 0), stride=(1, 1))
        # Unfloor
        self.unfloor = Unfloor(self.floor)
        # Denormalize
        self.denormalize = Denormalize()

        # 重みの初期化を行う
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.bb1.conv1.weight)
        nn.init.kaiming_normal_(self.bb1.conv2.weight)
        nn.init.kaiming_normal_(self.bb2.conv1.weight)
        nn.init.kaiming_normal_(self.bb2.conv2.weight)
        nn.init.kaiming_normal_(self.bb3.conv1.weight)
        nn.init.kaiming_normal_(self.bb3.conv2.weight)
        nn.init.kaiming_normal_(self.bb4.conv1.weight)
        nn.init.kaiming_normal_(self.bb4.conv2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.dfc1.weight)
        nn.init.kaiming_normal_(self.ibb4.conv1.weight)
        nn.init.kaiming_normal_(self.ibb4.conv2.weight)
        nn.init.kaiming_normal_(self.ibb3.conv1.weight)
        nn.init.kaiming_normal_(self.ibb3.conv2.weight)
        nn.init.kaiming_normal_(self.ibb2.conv1.weight)
        nn.init.kaiming_normal_(self.ibb2.conv2.weight)
        # nn.init.kaiming_normal_(self.ibb1.conv1.weight)
        # nn.init.kaiming_normal_(self.ibb1.conv2.weight)
        nn.init.kaiming_normal_(self.iconv1.weight)
        nn.init.kaiming_normal_(self.iconv0.weight)

    @property
    def bottleneck_dim(self):
        return 256

    def encode(self, x):
        l2 = self.l2norm(x)
        x = self.normalize(x, l2)
        x = F.relu(self.floor(x))
        x = self.conv1(x)
        x = self.bb1(x)
        x = self.mp1(x)
        x = self.bb2(x)
        x = self.mp2(x)
        x = self.bb3(x)
        x = self.mp3(x)
        x = self.bb4(x)
        x = self.mp4(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.tanh(self.fc1(x))
        return x, l2

    def decode(self, x, l2):
        x = F.relu(self.dfc1(x))
        x = x.reshape(-1, 32, 10, 1)
        x = self.up4(x)
        x = self.ibb4(x)
        x = self.up3(x)
        x = self.ibb3(x)
        x = self.up2(x)
        x = self.ibb2(x)
        x = self.up1(x)
        # x = self.ibb1(x)
        x = self.iconv1(x)
        x = self.bn1(x)
        x = self.iconv0(x)
        x = self.unfloor(x)
        x = self.denormalize(x, l2)
        return x
