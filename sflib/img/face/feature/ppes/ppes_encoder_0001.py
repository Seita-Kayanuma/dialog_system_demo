from .base import PPESEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPESEncoder0001(PPESEncoder):
    def __init__(self, n_person: int):
        super().__init__(n_person)
        # 1x64x64の画像 -> 2048 (4x4x128) -> 256
        # ('ic' stands for input convolution, 'il' for input linear)
        self.ic1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.ic1b = nn.BatchNorm2d(16)
        self.ic2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.ic2b = nn.BatchNorm2d(32)
        self.ic3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.ic3b = nn.BatchNorm2d(64)
        self.ic4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.ic4b = nn.BatchNorm2d(128)
        self.ic5 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.ic5b = nn.BatchNorm2d(256)
        self.il1 = nn.Linear(4 * 4 * 256, 4096)
        self.il1b = nn.BatchNorm1d(4096)
        self.il2 = nn.Linear(4096, 2048)
        self.il2b = nn.BatchNorm1d(2048)
        self.il3 = nn.Linear(2048, 1024)
        self.il3b = nn.BatchNorm1d(1024)
        # 個人情報 256 -> 64 -> n_person
        self.pe_l1 = nn.Linear(1024, 128)
        self.pe_l1b = nn.BatchNorm1d(128)
        self.pe_l2 = nn.Linear(128, 64)
        self.pe_l2b = nn.BatchNorm1d(64)
        self.pe_o = nn.Linear(64, n_person)
        # self.pe_l1 = nn.Linear(1024, 1024)
        # self.pe_l2 = nn.Linear(1024, 1024)
        # self.pe_o = nn.Linear(1024, n_person)
        # ポーズ情報 256 -> 16 -> 6 + 52
        self.po_l1 = nn.Linear(1024, 64)
        self.po_l1b = nn.BatchNorm1d(64)
        self.po_l2 = nn.Linear(64, 16)
        self.po_l2b = nn.BatchNorm1d(16)
        self.po_o1 = nn.Linear(16, 6)
        self.po_o2 = nn.Linear(16, 52)
        # 表情情報 256 -> 16 -> 84
        self.ex_l1 = nn.Linear(1024, 64)
        self.ex_l1b = nn.BatchNorm1d(64)
        self.ex_l2 = nn.Linear(64, 16)
        self.ex_l2b = nn.BatchNorm1d(16)
        self.ex_o = nn.Linear(16, 84)
        # 画像再合成
        self.fa_l1 = nn.Linear(64 + 16 + 16, 1024)
        # self.fa_l1 = nn.Linear(1024 + 16 + 16, 1024)
        self.fa_l1b = nn.BatchNorm1d(1024)
        self.fa_l2 = nn.Linear(1024, 2048)
        self.fa_l2b = nn.BatchNorm1d(2048)
        self.fa_l3 = nn.Linear(2048, 4096)
        self.fa_l3b = nn.BatchNorm1d(4096)
        self.fa_l4 = nn.Linear(4096, 4 * 4 * 256)
        self.fa_l4b = nn.BatchNorm1d(4 * 4 * 256)
        self.id4 = nn.ConvTranspose2d(256, 128, \
                                      kernel_size=6, stride=2, padding=2)
        self.id4b = nn.BatchNorm2d(128)
        self.id3 = nn.ConvTranspose2d(128, 64, \
                                      kernel_size=6, stride=2, padding=2)
        self.id3b = nn.BatchNorm2d(64)
        self.id2 = nn.ConvTranspose2d(64, 32, \
                                      kernel_size=6, stride=2, padding=2)
        self.id2b = nn.BatchNorm2d(32)
        self.id1 = nn.ConvTranspose2d(32, 1, \
                                      kernel_size=6, stride=2, padding=2)

    @property
    def feature_dims(self) -> tuple:
        return 64, 16, 16

    def encode(self, x: torch.tensor) -> tuple:
        batch, _, w, h = x.shape
        x = F.relu(self.ic1(x))
        x = self.ic1b(x)
        x = F.relu(self.ic2(x))
        x = self.ic2b(x)
        x = F.relu(self.ic3(x))
        x = self.ic3b(x)
        x = F.relu(self.ic4(x))
        x = self.ic4b(x)
        x = F.relu(self.ic5(x))
        x = self.ic5b(x)
        x = x.reshape(batch, -1)
        x = F.relu(self.il1(x))
        x = self.il1b(x)
        x = F.relu(self.il2(x))
        x = self.il2b(x)
        x = F.relu(self.il3(x))
        x = self.il3b(x)
        x_pe = F.relu(self.pe_l1(x))
        x_pe = self.pe_l1b(x_pe)
        x_pe = self.pe_l2(x_pe)
        x_pe = self.pe_l2b(x_pe)
        x_pe = torch.tanh(x_pe)
        x_po = F.relu(self.po_l1(x))
        x_po = self.po_l1b(x_po)
        x_po = self.po_l2(x_po)
        x_po = self.po_l2b(x_po)
        x_po = torch.tanh(x_po)
        x_ex = F.relu(self.ex_l1(x))
        x_ex = self.ex_l1b(x_ex)
        x_ex = self.ex_l2(x_ex)
        x_ex = self.ex_l2b(x_ex)
        x_ex = torch.tanh(x_ex)
        return x_pe, x_po, x_ex

    def decode(self, x_per, x_pose, x_expr):
        y_per = self.pe_o(x_per)
        y_pose1 = self.po_o1(x_pose)
        y_pose2 = self.po_o2(x_pose)
        y_expr = self.ex_o(x_expr)
        x_all = torch.cat([x_per, x_pose, x_expr], dim=1)
        x = F.relu(self.fa_l1(x_all))
        x = self.fa_l1b(x)
        x = F.relu(self.fa_l2(x))
        x = self.fa_l2b(x)
        x = F.relu(self.fa_l3(x))
        x = self.fa_l3b(x)
        x = F.relu(self.fa_l4(x))
        x = self.fa_l4b(x)
        x = x.reshape(-1, 256, 4, 4)
        x = F.relu(self.id4(x))
        x = self.id4b(x)
        x = F.relu(self.id3(x))
        x = self.id3b(x)
        x = F.relu(self.id2(x))
        x = self.id2b(x)
        x = torch.sigmoid(self.id1(x))
        y_image = x
        return y_per, y_pose1, y_pose2, y_expr, y_image
