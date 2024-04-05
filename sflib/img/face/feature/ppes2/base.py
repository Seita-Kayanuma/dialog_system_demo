from abc import ABCMeta, abstractmethod
from os import path
import torch.nn as nn
import torch
from ..... import config
from .....cloud.google import GoogleDriveInterface
from .....ext.torch.trainer import TorchTrainer
from .extraction import PPESv2, _apply_affine_transform
import numpy as np


class PPESv2Encoder(nn.Module):
    def __init__(self, n_person: int):
        """
        Args:
          n_person (int): 個人識別のバリエーション
        """
        super().__init__()
        self._n_person = n_person

    def get_filename_base(self):
        """パラメタ保存のためのファイル名の元となる文字列
        """
        return self.__class__.__name__ + '+{}'.format(self.n_person)

    @property
    def feature_dims(self) -> tuple:
        """各特徴量の次元数．
        長さ3のリストで，それぞれ
        個人性(S)，個人性(A)，ポーズ，表情(S)，表情(A)の次元数を表す．
        """
        raise NotImplementedError()

    @property
    def n_person(self):
        """個人識別対象のバリエーション数
        """
        return self._n_person

    def forward(self, x: torch.tensor) -> tuple:
        """
        Args:
          x: (batch, 1, 64, 64)
        Returns:
          [0] 個人情報の対数尤度（batch, 人数のバリエーション）
          [1] ポーズ情報（アフィン変換情報） (batch, 6)
          [2] ポーズ情報（鼻・顎ライン情報） (batch, 52)
          [3] 表情情報 (batch, 84)
          [4] アライメント情報 (batch, 2, 64, 64)
          [5] アライメント後の合成画像 (batch, 1, 64, 64)
        """
        return self.decode(*(self.encode(x)))

    def get_device(self):
        """
        デバイスを取得する
        """
        return next(self.parameters()).device

    def encode(self, image: torch.tensor) -> tuple:
        """
        画像をエンコードしてベクトルにする．
        
        Args:
           image (torch.tensor): 画像をエンコードしてベクトルにする．
             0.0〜1.0の値を持つ画像．
             shape は (batch, 1, 64, 64)．

        Returns:
           tuple:
             [0] 個人情報に対応するベクトル(S)．(batch, 次元)
             [1] 個人情報に対応するベクトル(A)．(batch, 次元)
             [2] ポーズ情報に対応するベクトル．(batch, 次元)
             [3] 表情情報に対応するベクトル(S)．(batch, 次元)
             [4] 表情情報に対応するベクトル(A)．(batch, 次元)
        """
        raise NotImplementedError()

    def decode(self, x_per_s: torch.tensor, x_per_a: torch.tensor,
               x_pose: torch.tensor, x_expr_s: torch.tensor,
               x_expr_a: torch.tensor) -> tuple:
        """
        エンコードされた各種情報から，出力に変換する．
        
        Args:
          x_per_s: 個人情報(S)に対応する入力ベクトル (batch, 次元)
          x_per_a: 個人情報(A)に対応する入力ベクトル (batch, 次元)
          x_pose: ポーズ情報に対応する入力ベクトル (batch, 次元)
          x_expr_s: 表情(S)に対応する入力ベクトル (batch, 次元)
          x_expr_a: 表情(A)対応する入力ベクトル (batch, 次元)
        Returns:
          [0] 個人情報の対数尤度（batch, 人数のバリエーション）
          [1] ポーズ情報（アフィン変換情報） (batch, 6)
          [2] ポーズ情報（鼻・顎ライン情報） (batch, 52)
          [3] 表情情報 (batch, 84)
          [4] アライメント情報 (batch, 2, 64, 64)
          [5] アライメント後の合成画像 (batch, 1, 64, 64)
        """
        raise NotImplementedError()

    def save_model(self, filename):
        self.eval()
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, map_location=None):
        self.eval()
        self.load_state_dict(torch.load(filename, map_location=map_location))


class TorchTrainerForPPESv2Encoder(TorchTrainer):
    def __init__(self, ppes_encoder: PPESv2Encoder, *args, **kwargs):
        self._ppes_encoder = ppes_encoder
        # kwargs.update({'automatic_input_transfer': False})
        super().__init__(self._ppes_encoder, *args, **kwargs)

    def _forward(self, batch, update=True, criterion=None):
        """
        batch[0]: 画像 (b, 1, 64, 64) 
        batch[1]: 個人インデクス（b,) 
        batch[3]: ポーズ情報（アフィン変換情報） (b, 6)
        batch[4]: ポーズ情報（鼻・顎ライン情報） (b, 52)
        batch[5]: 表情情報 (b, 84)
        batch[6]: アライメント情報 (b, 2, 64, 64)
        batch[7]: アライメント後の合成画像 (b, 1, 64, 64)
        """
        x_image, x_per, x_po1, x_po2, x_expr, x_alignment, x_aligned_image = batch
        y_per, y_po1, y_po2, y_expr, y_alignment, y_image = \
            self._ppes_encoder.forward(x_image)
        if criterion is None:
            criterion = self._criterion
        loss = self._criterion(y_per, x_per, y_po1, x_po1, y_po2, x_po2,
                               y_expr, x_expr, y_alignment, x_alignment,
                               y_image, x_aligned_image)
        if update:
            self._optimzier.zero_grad()
            loss.backward()
            self._callback_train_before_optimizer_step()
            self._optimzier.step()

        return loss


class PPESv2EncoderTrainer:
    def __init__(self, ppes_encoder: PPESv2Encoder):
        self.ppes_encoder = ppes_encoder

    def get_filename_base(self):
        return self.__class__.__name__

    def get_model_filename(self):
        filename = self.ppes_encoder.get_filename_base()
        filename += '+' + self.get_filename_base()
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    def get_csv_log_filename(self):
        filename = self.ppes_encoder.get_filename_base()
        filename += '+' + self.get_filename_base() + '.csv'
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    @abstractmethod
    def build_torch_trainer(self, ppes_encoder: PPESv2Encoder
                            ) -> TorchTrainerForPPESv2Encoder:
        pass

    def train(self):
        self.torch_trainer = self.build_torch_trainer(self.ppes_encoder)
        self.torch_trainer.train()

    def upload_csv_log(self):
        filename = self.get_csv_log_filename()
        if path.exists(filename):
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename), mediaType='text/csv')


def save_ppes_v2_encoder(trainer, upload=False):
    filename = trainer.get_model_filename()
    trainer.ppes_encoder.save_model(filename)
    if upload is True:
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename))


def load_ppes_v2_encoder(trainer, download=False, map_location=None):
    filename = trainer.get_model_filename()
    # print(filename)
    if download is True or not path.exists(filename):
        g = GoogleDriveInterface()
        g.download(path.basename(filename), filename)
    trainer.ppes_encoder.load_model(filename, map_location=map_location)


def get_info_for_draw(y_pose1, y_pose2, y_expr, y_image):
    pts = PPESv2.pose2_pts
    pose2 = y_pose2.detach().cpu().numpy()
    pose1 = y_pose1.detach().cpu().numpy()
    pose2_pts = pose2.reshape(-1, 2) + PPESv2.template[pts, :]
    pose2_pts = _apply_affine_transform(pose2_pts,
                                        pose1.reshape(2, 3),
                                        inv=True)
    pts = PPESv2.expr_pts
    expression = y_expr.detach().cpu().numpy()
    expr_pts = expression.reshape(-1, 2) + PPESv2.template[pts, :]

    image = np.uint8(y_image.detach().cpu().numpy() * 255.0).reshape((64, 64))

    return PPESv2.template, pose2_pts, expr_pts, image


def construct_encoder(encoder_version: int, n_person: int) -> PPESv2Encoder:
    """エンコーダをバージョン指定で構築する"""
    module_name = 'sflib.img.face.feature.ppes2.ppes_v2_encoder_{:04d}'.format(encoder_version)
    class_name = "PPESv2Encoder{:04d}".format(encoder_version)
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    encoder = cls(n_person)
    return encoder


def construct_trainer(module_postfix: str, class_postfix: str, encoder: PPESv2Encoder):
    """学習器をモジュール名の最後の部分ととクラス名の最後の部分で構築する"""
    module_name = "sflib.img.face.feature.ppes2.trainer_{}".format(module_postfix)
    class_name = "PPESv2EncoderTrainer{}".format(class_postfix)
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    trainer = cls(encoder)
    return trainer


def load(encoder_version: int,
         n_person: int,
         trainer_module_postfix: str,
         trainer_class_postfix: str,
         device=None,
         map_location=None):
    encoder = construct_encoder(encoder_version, n_person)
    trainer = construct_trainer(trainer_module_postfix, trainer_class_postfix, encoder)
    load_ppes_v2_encoder(trainer, download=False, map_location=map_location)
    return trainer

