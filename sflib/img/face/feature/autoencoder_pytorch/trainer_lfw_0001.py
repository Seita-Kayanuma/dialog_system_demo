# coding: utf-8
import numpy as np
from sflib.img.face.alignment import FaceAligner
from sflib.corpus.lfw.process import AlignedFaces
from .base import FaceAutoEncoderTrainer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# 顔画像のマスクを初期化
mask = FaceAligner().mask
mask = torch.tensor(np.float32(mask > 0))


class MaskedMSELoss(nn.MSELoss):
    def __init__(self,
                 mask=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean'):
        super(MaskedMSELoss, self).__init__(size_average, reduce, reduction)
        self.mask = mask

    def forward(self, input, target):
        # import ipdb; ipdb.set_trace()
        return super(MaskedMSELoss, self).forward(input * self.mask,
                                                  target * self.mask)

    def to(self, device):
        that = super(MaskedMSELoss, self).to(device)
        that.mask = self.mask.clone().detach().to(device)
        return that


class FaceAutoEncoderTrainerLFW0001(FaceAutoEncoderTrainer):
    def __init__(self):
        super().__init__()

        self.x_train = None
        self.x_test = None

        self.crit = MaskedMSELoss(mask)

    def build_data(self):
        # 学習データ，テストデータの生成
        af = AlignedFaces()
        ap = np.float32(af.dataframe.iloc[:, (2 + 4 + 68 * 2):])
        ap = ap.reshape(-1, 96, 96)
        ap_flipped = np.flip(ap, 2)

        ap_train = np.stack([ap[:12000], ap_flipped[:12000]], axis=0)
        ap_test = np.stack([ap[12000:], ap_flipped[12000:]], axis=0)

        x_train = ap_train.reshape(-1, 96, 96)
        x_train /= 255
        self.x_train = x_train
        x_test = ap_test.reshape(-1, 96, 96)
        x_test /= 255
        self.x_test = x_test

    def get_train_dataset(self):
        if self.x_train is None:
            self.build_data()
        return TensorDataset(torch.tensor(self.x_train))

    def get_validation_dataset(self):
        if self.x_test is None:
            self.build_data()
        return TensorDataset(torch.tensor(self.x_test))
