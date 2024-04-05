from .base import PPESv2Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPESv2Encoder0001(PPESv2Encoder):
    def __init__(self, n_person: int):
        super().__init__(n_person)
        # 1 x 64 x 64の画像 -> 512 (4 x 4 x 32) -> 256
        # (i) ... input
        self.i_c1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.i_c1b = nn.BatchNorm2d(32)
        self.i_c2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.i_c2b = nn.BatchNorm2d(32)
        self.i_c3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.i_c3b = nn.BatchNorm2d(32)
        self.i_c4 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.i_c4b = nn.BatchNorm2d(32)
        self.i_l1 = nn.Linear(4 * 4 * 32, 512)
        self.i_l1b = nn.BatchNorm1d(512)
        self.i_l2 = nn.Linear(512, 256)
        self.i_l2b = nn.BatchNorm1d(256)
        # 個人情報(S) 256 -> 32
        # (pe_s) ... person (S)
        self.pe_s_l1 = nn.Linear(256, 128)
        self.pe_s_l1b = nn.BatchNorm1d(128)
        self.pe_s_l2 = nn.Linear(128, 32)
        self.pe_s_l2b = nn.BatchNorm1d(32)
        # ポーズ情報(S) 256 -> 16
        # (po_s) ... pose (S)
        self.po_s_l1 = nn.Linear(256, 64)
        self.po_s_l1b = nn.BatchNorm1d(64)
        self.po_s_l2 = nn.Linear(64, 16)
        self.po_s_l2b = nn.BatchNorm1d(16)
        # 表情情報(S) 256 -> 16
        # (ex_s) ... expression (S)
        self.ex_s_l1 = nn.Linear(256, 64)
        self.ex_s_l1b = nn.BatchNorm1d(64)
        self.ex_s_l2 = nn.Linear(64, 16)
        self.ex_s_l2b = nn.BatchNorm1d(16)
        # アライメント情報合成 (32 + 16 = 48) -> (2 x 64 x 64)
        # (a) ... alignment
        self.a_l1 = nn.Linear(16 + 16, 512)
        self.a_l1b = nn.BatchNorm1d(512)
        self.a_l2 = nn.Linear(512, 4 * 4 * 32)
        self.a_l2b = nn.BatchNorm1d(4 * 4 * 32)
        self.a_d4 = nn.ConvTranspose2d(32, 32, \
                                       kernel_size=6, stride=2, padding=2)
        self.a_d4b = nn.BatchNorm2d(32)
        self.a_d3 = nn.ConvTranspose2d(32, 32, \
                                       kernel_size=6, stride=2, padding=2)
        self.a_d3b = nn.BatchNorm2d(32)
        self.a_d2 = nn.ConvTranspose2d(32, 32, \
                                       kernel_size=6, stride=2, padding=2)
        self.a_d2b = nn.BatchNorm2d(32)
        self.a_d1 = nn.ConvTranspose2d(32, 2, \
                                       kernel_size=6, stride=2, padding=2)
        # アライメント後の圧縮
        # 1 x 64 x 64 -> 512 (4 x 4 x 32) -> 256
        # (ai) ... aligned input
        self.ai_c1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.ai_c1b = nn.BatchNorm2d(32)
        self.ai_c2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.ai_c2b = nn.BatchNorm2d(32)
        self.ai_c3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.ai_c3b = nn.BatchNorm2d(32)
        self.ai_c4 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.ai_c4b = nn.BatchNorm2d(32)
        self.ai_l1 = nn.Linear(4 * 4 * 32, 512)
        self.ai_l1b = nn.BatchNorm1d(512)
        self.ai_l2 = nn.Linear(512, 256)
        self.ai_l2b = nn.BatchNorm1d(256)
        # 個人情報(A) 256 -> 32
        # (pe_a) ... person (A)
        self.pe_a_l1 = nn.Linear(256, 128)
        self.pe_a_l1b = nn.BatchNorm1d(128)
        self.pe_a_l2 = nn.Linear(128, 32)
        self.pe_a_l2b = nn.BatchNorm1d(32)
        # 表情情報(A) 256 -> 16
        # (ex_a) ... expression (A)
        self.ex_a_l1 = nn.Linear(256, 64)
        self.ex_a_l1b = nn.BatchNorm1d(64)
        self.ex_a_l2 = nn.Linear(64, 16)
        self.ex_a_l2b = nn.BatchNorm1d(16)
        # 個人情報出力
        # (pe_o) ... person out
        self.pe_o = nn.Linear(32 + 32, n_person)
        # ポーズ1出力（アフィン変換）
        # (po_o1) ... pose output 1
        self.po_o1 = nn.Linear(16, 6)
        # ポーズ2出力（鼻，顎ライン）
        # (po_o2) ... pose output 2
        self.po_o2 = nn.Linear(16, 52)
        # 表情出力
        # (ex_o) ... expression output
        self.ex_o = nn.Linear(16 + 16, 84)
        # 画像再合成 32 + 16 -> 1 x 64 x 64
        # (ao) ... aligned output
        self.ao_l1 = nn.Linear(32 + 16, 512)
        self.ao_l1b = nn.BatchNorm1d(512)
        self.ao_l2 = nn.Linear(512, 4 * 4 * 32)
        self.ao_l2b = nn.BatchNorm1d(4 * 4 * 32)
        self.ao_d4 = nn.ConvTranspose2d(32, 32, \
                                        kernel_size=6, stride=2, padding=2)
        self.ao_d4b = nn.BatchNorm2d(32)
        self.ao_d3 = nn.ConvTranspose2d(32, 32, \
                                        kernel_size=6, stride=2, padding=2)
        self.ao_d3b = nn.BatchNorm2d(32)
        self.ao_d2 = nn.ConvTranspose2d(32, 32, \
                                        kernel_size=6, stride=2, padding=2)
        self.ao_d2b = nn.BatchNorm2d(32)
        self.ao_d1 = nn.ConvTranspose2d(32, 1, \
                                        kernel_size=6, stride=2, padding=2)

    @property
    def feature_dims(self) -> tuple:
        return 32, 32, 16, 16, 16

    def encode(self, x_in: torch.tensor) -> tuple:
        batch, _, w, h = x_in.shape
        # 入力画像のベクトル化
        x = F.relu(self.i_c1(x_in))
        x = self.i_c1b(x)
        x = F.relu(self.i_c2(x))
        x = self.i_c2b(x)
        x = F.relu(self.i_c3(x))
        x = self.i_c3b(x)
        x = F.relu(self.i_c4(x))
        x = self.i_c4b(x)
        x = x.reshape(batch, -1)
        x = F.relu(self.i_l1(x))
        x = self.i_l1b(x)
        x = F.relu(self.i_l2(x))
        x = self.i_l2b(x)
        # 個人情報(S)
        x_pe_s = F.relu(self.pe_s_l1(x))
        x_pe_s = self.pe_s_l1b(x_pe_s)
        x_pe_s = self.pe_s_l2(x_pe_s)
        x_pe_s = self.pe_s_l2b(x_pe_s)
        x_pe_s = torch.tanh(x_pe_s)
        # ポーズ情報(S)
        x_po_s = F.relu(self.po_s_l1(x))
        x_po_s = self.po_s_l1b(x_po_s)
        x_po_s = self.po_s_l2(x_po_s)
        x_po_s = self.po_s_l2b(x_po_s)
        x_po_s = torch.tanh(x_po_s)
        # 表情情報(S)
        x_ex_s = F.relu(self.ex_s_l1(x))
        x_ex_s = self.ex_s_l1b(x_ex_s)
        x_ex_s = self.ex_s_l2(x_ex_s)
        x_ex_s = self.ex_s_l2b(x_ex_s)
        x_ex_s = torch.tanh(x_ex_s)
        # アライメント情報合成
        x_po_ex_s = torch.cat([x_po_s, x_ex_s], dim=1)
        x_a = F.relu(self.a_l1(x_po_ex_s))
        x_a = self.a_l1b(x_a)
        x_a = F.relu(self.a_l2(x_a))
        x_a = self.a_l2b(x_a)
        x_a = x_a.reshape(-1, 32, 4, 4)
        x_a = F.relu(self.a_d4(x_a))
        x_a = self.a_d4b(x_a)
        x_a = F.relu(self.a_d3(x_a))
        x_a = self.a_d3b(x_a)
        x_a = F.relu(self.a_d2(x_a))
        x_a = self.a_d2b(x_a)
        x_a = self.a_d1(x_a)
        # アライメントをする
        x_a_np = x_a.detach().cpu().numpy().reshape(-1, 2, 64, 64)
        orig_x = np.arange(0.0, 1.0, 1 / 64)
        orig = np.stack(np.meshgrid(orig_x, orig_x), -1)
        x_a_np = np.int32((x_a_np.transpose((0, 2, 3, 1)) + orig) * 64)
        x_a_np[x_a_np < 0] = 0
        x_a_np[x_a_np > 63] = 63
        x_in = x_in.cpu().numpy().reshape(-1, 64, 64)
        z = np.arange(x_in.shape[0]).reshape(-1, 1, 1) * np.ones((1, 64, 64), np.int32)
        x_t = x_in[z, x_a_np[:, :, :, 1], x_a_np[:, :, :, 0]]
        x_ = x_t.reshape(-1, 1, 64, 64)
        x_ = torch.tensor(x_)
        if self.get_device().type != 'cpu':
            x_ = x_.to(self.get_device())
        # アライメント後の圧縮
        x_ai = x_
        x_ai = F.relu(self.ai_c1(x_ai))
        x_ai = self.ai_c1b(x_ai)
        x_ai = F.relu(self.ai_c2(x_ai))
        x_ai = self.ai_c2b(x_ai)
        x_ai = F.relu(self.ai_c3(x_ai))
        x_ai = self.ai_c3b(x_ai)
        x_ai = F.relu(self.ai_c4(x_ai))
        x_ai = self.ai_c4b(x_ai)
        x_ai = x_ai.reshape(batch, -1)
        x_ai = F.relu(self.ai_l1(x_ai))
        x_ai = self.ai_l1b(x_ai)
        x_ai = F.relu(self.ai_l2(x_ai))
        x_ai = self.ai_l2b(x_ai)
        # 個人情報(A)
        x_pe_a = F.relu(self.pe_a_l1(x_ai))
        x_pe_a = self.pe_a_l1b(x_pe_a)
        x_pe_a = F.relu(self.pe_a_l2(x_pe_a))
        x_pe_a = self.pe_a_l2b(x_pe_a)
        x_pe_a = torch.tanh(x_pe_a)
        # 表情情報(A)
        x_ex_a = F.relu(self.ex_a_l1(x_ai))
        x_ex_a = self.ex_a_l1b(x_ex_a)
        x_ex_a = F.relu(self.ex_a_l2(x_ex_a))
        x_ex_a = self.ex_a_l2b(x_ex_a)
        x_ex_a = torch.tanh(x_ex_a)
        return x_pe_s, x_pe_a, x_po_s, x_ex_s, x_ex_a

    def decode(self, x_pe_s, x_pe_a, x_po_s, x_ex_s, x_ex_a):
        # 個人情報
        x_per = torch.cat([x_pe_s, x_pe_a], dim=1)
        y_per = self.pe_o(x_per)
        # ポーズ1
        y_pose1 = self.po_o1(x_po_s)
        # ポーズ2
        y_pose2 = self.po_o2(x_po_s)
        # 表情
        x_expr = torch.cat([x_ex_s, x_ex_a], dim=1)
        y_expr = self.ex_o(x_expr)
        # 画像出力
        x_ao = torch.cat([x_pe_a, x_ex_a], dim=1)
        x_ao = F.relu(self.ao_l1(x_ao))
        x_ao = self.ao_l1b(x_ao)
        x_ao = F.relu(self.ao_l2(x_ao))
        x_ao = self.ao_l2b(x_ao)
        x_ao = x_ao.reshape(-1, 32, 4, 4)
        x_ao = F.relu(self.ao_d4(x_ao))
        x_ao = self.ao_d4b(x_ao)
        x_ao = F.relu(self.ao_d3(x_ao))
        x_ao = self.ao_d3b(x_ao)
        x_ao = F.relu(self.ao_d2(x_ao))
        x_ao = self.ao_d2b(x_ao)
        x_ao = torch.sigmoid(self.ao_d1(x_ao))
        y_image = x_ao
        # アフィン変換は計算し直す（効率悪いが...）
        x_po_ex_s = torch.cat([x_po_s, x_ex_s], dim=1)
        x_a = F.relu(self.a_l1(x_po_ex_s))
        x_a = self.a_l1b(x_a)
        x_a = F.relu(self.a_l2(x_a))
        x_a = self.a_l2b(x_a)
        x_a = x_a.reshape(-1, 32, 4, 4)
        x_a = F.relu(self.a_d4(x_a))
        x_a = self.a_d4b(x_a)
        x_a = F.relu(self.a_d3(x_a))
        x_a = self.a_d3b(x_a)
        x_a = F.relu(self.a_d2(x_a))
        x_a = self.a_d2b(x_a)
        x_a = self.a_d1(x_a)
        y_alignment = torch.tanh(x_a)
        
        return y_per, y_pose1, y_pose2, y_expr, y_alignment, y_image 
