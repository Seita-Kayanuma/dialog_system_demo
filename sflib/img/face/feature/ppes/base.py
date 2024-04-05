from abc import ABCMeta, abstractmethod
from os import path
import torch.nn as nn
import torch
from ..... import config
from .....cloud.google import GoogleDriveInterface
from .....ext.torch.trainer import TorchTrainer


class PPESEncoder(nn.Module):
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
        個人性ベクトル，ポーズベクトル，表情ベクトルの次元
        を表す．
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
          [4] 再合成画像 (batch, 1, 64, 64)
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
             [0]は個人情報に対応するベクトル．(batch, 次元)
             [1]はポーズ情報に対応するベクトル．(batch, 次元)
             [2]は表情情報に対応するベクトル．(batch, 次元)
        """
        raise NotImplementedError()

    def decode(self, x_per: torch.tensor, x_pose: torch.tensor,
               x_expr: torch.tensor) -> tuple:
        """
        エンコードされた各種情報から，出力に変換する．
        
        Args:
          x_per: 個人情報に対応する入力ベクトル (batch, 次元)
          x_pose: ポーズ情報に対応する入力ベクトル (batch, 次元)
          x_expr: 表情ベクトルに対応する入力ベクトル (batch, 次元)
        Returns:
          [0] 個人情報の対数尤度（batch, n_person) 
          [1] ポーズ情報（アフィン変換情報） (batch, 6)
          [2] ポーズ情報（鼻・顎ライン情報） (batch, 52)
          [3] 表情情報 (batch, 84)
          [4] 再合成画像 (batch, 1, 64, 64)
        """
        raise NotImplementedError()

    def save_model(self, filename):
        self.eval()
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, map_location=None):
        self.eval()
        self.load_state_dict(torch.load(filename, map_location=map_location))


class TorchTrainerForPPESEncoder(TorchTrainer):
    def __init__(self, ppes_encoder: PPESEncoder, *args, **kwargs):
        self._ppes_encoder = ppes_encoder
        # kwargs.update({'automatic_input_transfer': False})
        super().__init__(self._ppes_encoder, *args, **kwargs)

    def _forward(self, batch, update=True):
        """
        batch[0]: 画像 (b, 1, 64, 64) 
        batch[1]: マスク (b, 1, 64, 64)
        batch[2]: 個人インデクス（b,) 
        batch[3]: ポーズ情報（アフィン変換情報） (b, 6)
        batch[4]: ポーズ情報（鼻・顎ライン情報） (b, 52)
        batch[5]: 表情情報 (b, 84)
        """
        x_image, x_mask, x_per, x_po1, x_po2, x_expr = batch
        y_per, y_po1, y_po2, y_expr, y_image = \
            self._ppes_encoder.forward(x_image)
        loss = self._criterion(y_image, x_image, x_mask, y_per, x_per, y_po1,
                               x_po1, y_po2, x_po2, y_expr, x_expr)
        if update:
            self._optimzier.zero_grad()
            loss.backward()
            self._callback_train_before_optimizer_step()
            self._optimzier.step()

        return loss


class PPESEncoderTrainer:
    def __init__(self, ppes_encoder: PPESEncoder):
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
    def build_torch_trainer(self, ppes_encoder: PPESEncoder
                            ) -> TorchTrainerForPPESEncoder:
        pass

    def train(self):
        self.torch_trainer = self.build_torch_trainer(self.ppes_encoder)
        self.torch_trainer.train()

    def upload_csv_log(self):
        filename = self.get_csv_log_filename()
        if path.exists(filename):
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename), mediaType='text/csv')


def save_ppes_encoder(trainer, upload=False):
    filename = trainer.get_model_filename()
    trainer.ppes_encoder.save_model(filename)
    if upload is True:
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename))


def load_ppes_encoder(trainer, download=False, map_location=None):
    filename = trainer.get_model_filename()
    if download is True or not path.exists(filename):
        g = GoogleDriveInterface()
        g.download(path.basename(filename), filename)
    trainer.ppes_encoder.load_model(filename, map_location=map_location)
