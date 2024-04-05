from .base import FaceAutoEncoder
from torch.nn import Conv2d, BatchNorm2d, Linear, BatchNorm1d, ConvTranspose2d
import torch


class FaceAutoEncoder0002PyTorch(FaceAutoEncoder):
    def __init__(self):
        super(FaceAutoEncoder0002PyTorch, self).__init__()
        # ---- encode side ----
        # convolution 1: (96, 96) -> (48, 48)
        self.conv1 = Conv2d(1, 32, (3, 3), stride=(2, 2), padding=1)
        self.bnc1 = BatchNorm2d(32)
        # convolution 2: (48, 48) -> (24, 24)
        self.conv2 = Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1)
        self.bnc2 = BatchNorm2d(64)
        # convolution 3: (24, 24) -> (12, 12)
        self.conv3 = Conv2d(64, 128, (3, 3), stride=(2, 2), padding=1)
        self.bnc3 = BatchNorm2d(128)
        # convolution 4: (12, 12) -> (6, 6)
        self.conv4 = Conv2d(128, 256, (3, 3), stride=(2, 2), padding=1)
        self.bnc4 = BatchNorm2d(256)
        # Flatten size -> 6 * 6 * 256 = 9216
        self.l1 = Linear(9216, 1028)
        self.bnl1 = BatchNorm1d(1)
        self.l2 = Linear(1028, 256)
        self.bnl2 = BatchNorm1d(1)
        self.l3 = Linear(256, 32)
        # --- decode side ---
        self.bndl3 = BatchNorm1d(1)
        self.dl3 = Linear(32, 256)
        self.bndl2 = BatchNorm1d(1)
        self.dl2 = Linear(256, 1024)
        self.bndl1 = BatchNorm1d(1)
        self.dl1 = Linear(1024, 9216)
        # deconvolution 4: (6, 6) -> (12, 12)
        self.bndc4 = BatchNorm2d(256)
        self.convd4 = ConvTranspose2d(256,
                                      128, (3, 3),
                                      stride=(2, 2),
                                      padding=1,
                                      output_padding=1)
        # deconvolution 3: (12, 12) -> (24, 24)
        self.bndc3 = BatchNorm2d(128)
        self.convd3 = ConvTranspose2d(128,
                                      64, (3, 3),
                                      stride=(2, 2),
                                      padding=1,
                                      output_padding=1)
        # deconvolution 2: (24, 24) -> (48, 48)
        self.bndc2 = BatchNorm2d(64)
        self.convd2 = ConvTranspose2d(64,
                                      32, (3, 3),
                                      stride=(2, 2),
                                      padding=1,
                                      output_padding=1)
        # deconvolution 1: (48, 48) -> (96, 96)
        self.bndc1 = BatchNorm2d(32)
        self.convd1 = ConvTranspose2d(32,
                                      1, (3, 3),
                                      stride=(2, 2),
                                      padding=1,
                                      output_padding=1)


    @property
    def bottleneck_dim(self):
        return 32

    def _encode(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bndc1(x)
        x = torch.relu(self.conv2(x))
        x = self.bndc2(x)
        x = torch.relu(self.conv3(x))
        x = self.bndc3(x)
        x = torch.relu(self.conv4(x))
        x = self.bndc4(x)
        x = x.view((-1, 1, 9216))
        x = torch.relu(self.l1(x))
        x = self.bnl1(x)
        x = torch.relu(self.l2(x))
        x = self.bnl2(x)
        x = torch.tanh(self.l3(x))
        return x

    def _decode(self, x):
        x = self.bndl3(x)
        x = torch.relu(self.dl3(x))
        x = self.bndl2(x)
        x = torch.relu(self.dl2(x))
        x = self.bndl1(x)
        x = torch.relu(self.dl1(x))
        x = x.view((-1, 256, 6, 6))
        x = self.bndc4(x)
        x = torch.relu(self.convd4(x))
        x = self.bndc3(x)
        x = torch.relu(self.convd3(x))
        x = self.bndc2(x)
        x = torch.relu(self.convd2(x))
        x = self.bndc1(x)
        x = torch.sigmoid(self.convd1(x))
        return x
