import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm

from .base import PPESEncoder, PPESEncoderTrainer, TorchTrainerForPPESEncoder
from .....corpus.lfw import LFW
from .....corpus.lfw.process import PPESData as LFW_PPESData
from .extraction import PPESExtractor
from .....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from .....ext.torch.callbacks.early_stopper import EarlyStopper
from .....ext.torch.callbacks.snapshot import Snapshot
from .....ext.torch.callbacks.train import ClippingGrad


class PPESDataset(Dataset):
    def __init__(self, ppes_data: LFW_PPESData, indices=None):
        if indices is None:
            indices = list(range(len(ppes_data)))
        self._ppes_data = ppes_data
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        ppes = self._ppes_data[self._indices[i]]
        ppes.refresh_noise()
        # import ipdb; ipdb.set_trace()
        return ppes


class CollatePPES:
    pose2_pts = \
        (0, 1, 2, 3, 4 , 5, 6,  7, 8, 9, 10,  11, 12, 13,  14, 15, 16) + \
        (27, 28, 29, 30, 31, 32, 33, 34, 35)
    expr_pts = \
        (17, 18, 19, 20, 21) + (22, 23, 24, 25, 26) + \
        (36, 37, 38, 39, 40, 41) + (42, 43, 44, 45, 46,  47) + \
        (48, 49, 50, 51, 52, 53, 54, 55, 56, 57,  58, 59, 60, 61, 62,  63, 64,  65, 66, 67)

    def __init__(self, id2index: dict):
        self._id2index = id2index

    def get_index(self, id: str):
        if id not in self._id2index:
            return 0
        else:
            return self._id2index[id]

    def __call__(self, ppes_list):
        image_list = []
        mask_list = []
        person_list = []
        pose1_list = []
        pose2_list = []
        expr_list = []
        for ppes in ppes_list:
            # 画像（チャネル次元を加える）
            image = np.float32(ppes.image) / 255.0
            image = image.reshape((1, ) + image.shape)
            image_list.append(image)
            # マスク（こちらもチャネルの次元を加える．型はもともとfloat32なのでいじらない）
            mask = ppes.mask.reshape((1, ) + ppes.mask.shape)
            mask_list.append(mask)
            # 人物．番号に直す
            person = self.get_index(ppes.id)
            # import ipdb; ipdb.set_trace()
            person_list.append(person)
            # ポーズ1情報．特にいじらなくてよい
            pose1 = ppes.pose
            pose1_list.append(pose1)
            # ポーズ2情報
            pose2 = ppes.expression[CollatePPES.pose2_pts, :]
            pose2 = pose2.reshape(-1)
            pose2_list.append(pose2)
            # 表情情報
            expr = ppes.expression[CollatePPES.expr_pts, :]
            expr = expr.reshape(-1)
            expr_list.append(expr)
        return \
            torch.tensor(np.stack(image_list)), \
            torch.tensor(np.stack(mask_list)), \
            torch.tensor(np.stack(person_list)), \
            torch.tensor(np.stack(pose1_list)), \
            torch.tensor(np.stack(pose2_list)), \
            torch.tensor(np.stack(expr_list))


class MSELossWithMask(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target, mask):
        loss = F.mse_loss(
            input * mask,
            target * mask,
            # size_average=self.size_average,
            reduction='none')
        # if self.reduction == 'mean':
        #     return torch.sum(loss) / torch.sum(mask)
        # if self.reduction == 'sum':
        #     return torch.sum(loss)

        # 1枚あたりの合計値
        loss = loss.sum(dim=(2, 3))
        # # 各画像のマスク内の面積
        # mask_s = mask.sum(dim=(2, 3))
        # # 画像毎にピクセル辺りのロスが求まる
        # loss = loss / mask_s
        # これの平均値？
        loss = loss.sum() / loss.shape[0]

        return loss


class PPESLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_image = MSELossWithMask()
        self._loss_person = nn.CrossEntropyLoss()
        self._loss_pose1 = nn.MSELoss()
        self._loss_pose2 = nn.MSELoss()
        self._loss_expr = nn.MSELoss()

    def forward(self, y_image, x_image, x_mask, y_per, x_per, y_po1, x_po1,
                y_po2, x_po2, y_expr, x_expr):
        loss_image = self._loss_image(y_image, x_image, x_mask)
        loss_person = self._loss_person(y_per, x_per)
        loss_pose1 = self._loss_pose1(y_po1, x_po1)
        loss_pose2 = self._loss_pose2(y_po2, x_po2)
        loss_expr = self._loss_expr(y_expr, x_expr)
        # import ipdb; ipdb.set_trace()
        loss = \
               0.01  * loss_image  + \
               0.1   * loss_person + \
               10.0  * loss_pose1  + \
               100.0 * loss_pose2  + \
               100.0 * loss_expr
        return loss


class PPESEncoderTrainerLFW0001(PPESEncoderTrainer):
    def __init__(self, ppes_encoder: PPESEncoder):
        super().__init__(ppes_encoder)

    def build_torch_trainer(self, ppes_encoder: PPESEncoder):
        criterion = PPESLoss()
        if ppes_encoder.get_device().type != 'cpu':
            criterion = criterion.to(ppes_encoder.get_device())
        optimizer = optim.Adam(ppes_encoder.parameters())

        lfw_ppes_data = LFW_PPESData()
        n_total = len(lfw_ppes_data)
        n_train = n_total // 10 * 9
        n_valid = n_total - n_train
        # n_train = 2000
        # n_valid = 1000
        idx_train = list(range(n_train))
        idx_valid = list(range(n_train, n_train + n_valid))

        id2index = {'unknown': 0}
        count = 1
        for i in tqdm.tqdm(idx_train):
            ppes = lfw_ppes_data[i]
            if ppes.id in id2index:
                continue
            elif count < ppes_encoder.n_person:
                id2index.update({ppes.id: count})
            count += 1
        print("{} persons are found. {} person are accepted.".format(
            count, ppes_encoder.n_person - 1))

        dataset_train = PPESDataset(lfw_ppes_data, idx_train)
        dataset_valid = PPESDataset(lfw_ppes_data, idx_valid)

        collate_fn = CollatePPES(id2index)

        train_loader = DataLoader(dataset_train,
                                  batch_size=100,
                                  collate_fn=collate_fn,
                                  num_workers=0,
                                  shuffle=True)
        valid_loader = DataLoader(dataset_valid,
                                  batch_size=100,
                                  collate_fn=collate_fn,
                                  num_workers=0,
                                  shuffle=False)
        callbacks = [
            # ClippingGrad(1.),
            StandardReporter(train_report_interval=1,
                             validation_report_interval=1),
            CsvWriterReporter(self.get_csv_log_filename()),
            Snapshot(final_filename=self.get_model_filename()),
            # EarlyStopper(patience=3, verbose=True),
        ]

        trainer = TorchTrainerForPPESEncoder(
            ppes_encoder,
            criterion,
            optimizer,
            train_loader,
            valid_loader,
            callbacks=callbacks,
            device=ppes_encoder.get_device().type,
            epoch=20)

        return trainer


def construct0001(device=None):
    from .ppes_encoder_0001 import PPESEncoder0001 as PPESEncoder
    ppes_encoder = PPESEncoder(n_person=4500)
    if device is not None:
        ppes_encoder.to(device)
    trainer = PPESEncoderTrainerLFW0001(ppes_encoder)
    return trainer


def construct0002(device=None):
    from .ppes_encoder_0002 import PPESEncoder0002 as PPESEncoder
    ppes_encoder = PPESEncoder(n_person=4500)
    if device is not None:
        ppes_encoder.to(device)
    trainer = PPESEncoderTrainerLFW0001(ppes_encoder)
    return trainer


def train(device=None, construct=construct0001):
    trainer = construct(device)
    trainer.train()

    from . import base as b
    b.save_ppes_encoder(trainer, upload=True)
    trainer.upload_csv_log()
