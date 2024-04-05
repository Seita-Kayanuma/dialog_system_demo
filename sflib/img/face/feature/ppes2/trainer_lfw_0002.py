# EPOCH数100でベストを取得．
# 未知人物をなしでやる（6000人までOKにする）．
# バリデーション時は人物のロスを無視する
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm

from .extraction import PPESv2
from .base import PPESv2Encoder, PPESv2EncoderTrainer, TorchTrainerForPPESv2Encoder
from .....corpus.lfw import LFW
from .....corpus.lfw.process import PPESv2Data as LFW_PPESv2Data
from .extraction import PPESv2Extractor
from .....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from .....ext.torch.callbacks.early_stopper import EarlyStopper
from .....ext.torch.callbacks.snapshot import Snapshot
from .....ext.torch.callbacks.train import ClippingGrad


class PPESv2Dataset(Dataset):
    def __init__(self, ppes_data: LFW_PPESv2Data, indices=None):
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


class CollatePPESv2:
    def __init__(self, id2index: dict):
        self._id2index = id2index

    def get_index(self, id: str):
        if id not in self._id2index:
            return 0
        else:
            return self._id2index[id]

    def __call__(self, ppes_list):
        image_list = []
        person_list = []
        pose1_list = []
        pose2_list = []
        expr_list = []
        alignment_list = []
        aligned_image_list = []
        for ppes in ppes_list:
            # 画像（チャネル次元を加える）
            image = np.float32(ppes.image) / 255.0
            image = image.reshape((1, ) + image.shape)
            image_list.append(image)
            # 人物．番号に直す
            person = self.get_index(ppes.id)
            # import ipdb; ipdb.set_trace()
            person_list.append(person)
            # ポーズ1情報．特にいじらなくてよい
            pose1 = np.float32(ppes.pose1)
            pose1_list.append(pose1)
            # ポーズ2情報
            pose2 = np.float32(ppes.pose2)
            pose2_list.append(pose2)
            # 表情情報
            expr = np.float32(ppes.expression)
            expr_list.append(expr)
            # アライメント情報
            alignment = ppes.alignment.transpose((2, 0, 1))
            alignment_list.append(alignment)
            # アライメント後の画像
            aligned_image = np.float32(ppes.aligned_image) / 255.0
            aligned_image = aligned_image.reshape((1, ) + aligned_image.shape)
            aligned_image_list.append(aligned_image)

        return \
            torch.tensor(np.stack(image_list)), \
            torch.tensor(np.stack(person_list)), \
            torch.tensor(np.stack(pose1_list)), \
            torch.tensor(np.stack(pose2_list)), \
            torch.tensor(np.stack(expr_list)), \
            torch.tensor(np.stack(alignment_list)), \
            torch.tensor(np.stack(aligned_image_list))


class MSELossWithMask(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target, mask):
        # import ipdb; ipdb.set_trace()
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


class PPESv2Loss(nn.Module):
    def __init__(self, *args, mask, person_loss_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_person = nn.CrossEntropyLoss()
        self._loss_pose1 = nn.MSELoss()
        self._loss_pose2 = nn.MSELoss()
        self._loss_expr = nn.MSELoss()
        self._loss_alignment = MSELossWithMask()
        self._loss_aligned_image = MSELossWithMask()
        self._mask = mask
        self._person_loss_enabled = person_loss_enabled

    def forward(self, y_per, x_per, y_po1, x_po1, y_po2, x_po2, y_expr, x_expr,
                y_alignment, x_alignment, y_aligned_image, x_aligned_image):
        loss_person = self._loss_person(y_per, x_per)
        loss_pose1 = self._loss_pose1(y_po1, x_po1)
        loss_pose2 = self._loss_pose2(y_po2, x_po2)
        loss_expr = self._loss_expr(y_expr, x_expr)
        loss_alignment = self._loss_alignment(y_alignment, x_alignment,
                                              self._mask)
        loss_aligned_image = self._loss_aligned_image(y_aligned_image,
                                                      x_aligned_image,
                                                      self._mask)

        # import ipdb; ipdb.set_trace()
        
        loss = \
               10.0  * loss_pose1  + \
               100.0 * loss_pose2  + \
               100.0 * loss_expr + \
               0.1   * loss_alignment + \
               0.01  * loss_aligned_image
        if self._person_loss_enabled:
            loss += 0.1 * loss_person

        return loss


class PPESv2EncoderTrainerLFW0002(PPESv2EncoderTrainer):
    def __init__(self, ppes_encoder: PPESv2Encoder):
        super().__init__(ppes_encoder)

    def build_torch_trainer(self, ppes_encoder: PPESv2Encoder):
        mask = torch.tensor(PPESv2.aligned_mask)
        if ppes_encoder.get_device().type != 'cpu':
            mask = mask.to(ppes_encoder.get_device())
        criterion = PPESv2Loss(mask=mask)
        validation_criterion = PPESv2Loss(mask=mask, person_loss_enabled=False)
        if ppes_encoder.get_device().type != 'cpu':
            criterion = criterion.to(ppes_encoder.get_device())
        optimizer = optim.Adam(ppes_encoder.parameters())

        lfw_ppes_data = LFW_PPESv2Data()
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

        dataset_train = PPESv2Dataset(lfw_ppes_data, idx_train)
        dataset_valid = PPESv2Dataset(lfw_ppes_data, idx_valid)

        collate_fn = CollatePPESv2(id2index)

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
            EarlyStopper(patience=100, verbose=True),
        ]

        trainer = TorchTrainerForPPESv2Encoder(
            ppes_encoder,
            criterion,
            optimizer,
            train_loader,
            valid_loader,
            callbacks=callbacks,
            device=ppes_encoder.get_device().type,
            epoch=100,
            validation_criterion=validation_criterion
        )

        return trainer


def construct0001(device=None):
    from .ppes_v2_encoder_0001 import PPESv2Encoder0001 as PPESv2Encoder
    ppes_encoder = PPESv2Encoder(n_person=6000)
    if device is not None:
        ppes_encoder.to(device)
    trainer = PPESv2EncoderTrainerLFW0002(ppes_encoder)
    return trainer


def train(device=None, construct=construct0001):
    trainer = construct(device)
    trainer.train()

    from . import base as b
    b.save_ppes_v2_encoder(trainer, upload=True)
    trainer.upload_csv_log()
