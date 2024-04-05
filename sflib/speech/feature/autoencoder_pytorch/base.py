import warnings
from os import path
from sflib import config
# from ....cloud.google import GoogleDriveInterface
from ....ext.torch.trainer import TorchTrainer
from ....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from ....ext.torch.callbacks.snapshot import Snapshot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import abstractmethod


class SpectrogramImageAutoEncoder(nn.Module):
    """スペクトログラム画像の自己符号化器
    """

    def __init__(self):
        super(SpectrogramImageAutoEncoder, self).__init__()
        self.__first_parameter = None

    def get_filename_base(self):
        """パラメータ保存などのためのファイル名のヒントを与える．
        クラス名をそのまま返す．
        """
        return self.__class__.__name__

    def __get_first_parameter(self):
        """モデルパラメタの最初のものを取得する．
        モデルがCPUかCUDAのどちらかを判定させるため"""
        if self.__first_parameter is None:
            self.__first_parameter = next(self.parameters())
        return self.__first_parameter

    @property
    def device(self) -> torch.device:
        """デバイス(CPU or CUDA)"""
        return self.__get_first_parameter().device

    @property
    def bottleneck_dim(self):
        """ボトルネック特徴量の次元数"""
        raise NotImplementedError

    @abstractmethod
    def encode(self, images: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """与えられた画像をエンコードし，ベクトル化する．
        
        Args:
          images (torch.Tensor): (バッチ, 1, 幅, 高さ)のテンソル．
            shape[1]はチャネルサイズで1カラー画像を入力にする予定は当面無いので
            1で固定．

        Returns:
          torch.Tensor: デコーディング結果である（バッチ, ベクトル次元）のテンソル
          torch.Tensor: 正規化係数である(バッチ, 1)のテンソル
        """
        raise NotImplementedError()

    @abstractmethod
    def decode(self, x: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        """与えられた特徴量でデコードを行う．

        Args:
          x (torch.Tensor): (バッチ, ベクトル次元）のテンソル．ボトルネック特徴量
          l2 (torch.Tensor): 正規化係数である（バッチ, 1）のテンソル

        oReturns:
          torch.Tensor: (バッチ, 1, 幅, 高さ)の画像状のテンソル．
        """
        raise NotImplementedError()

    def forward(self, imgs):
        """学習のためのフォワード計算
        """
        x, l2 = self.encode(imgs)
        return self.decode(x, l2)

    def save_weights(self, filename: str) -> None:
        """モデルパラメタの保存
        
        Args:
          filename (str): ファイル名
        """
        self.eval()
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename: str, map_location: str = None):
        """モデルパラメタの読み込み

        Args:
          filename (str): ファイル名
          map_location (str): パラメタを読み込む場所（CPU or CUDA）
        """
        self.eval()
        self.load_state_dict(torch.load(filename, map_location=map_location))


class SpectrogramImageAutoEncoderTrainer:
    """スペクトルオートエンコーダの学習器の基底クラス
    """

    def __init__(self, device=None):
        """
        Args:
          device (torch.device): cpu or cuda．
            ここで指定したデバイスで学習が行われることになる．
        """
        self._autoencoder = None
        self._device = device

    @property
    def autoencoder(self) -> SpectrogramImageAutoEncoder:
        """対応するオートエンコーダ"""
        return self._autoencoder

    @property
    def device(self) -> torch.device:
        """PyTorchを動作させるデバイス"""
        return self._device

    def get_filename_base(self):
        """ファイルの保存のヒントとなる文字列"""
        return self.__class__.__name__

    def set_autoencoder(self, autoencoder: SpectrogramImageAutoEncoder):
        """対応するオートエンコーダを設定．
        必要であればパラメタのGPUへの転送を行う．"""
        self._autoencoder = autoencoder
        if self._device is not None:
            self._autoencoder.to(self._device)

    @abstractmethod
    def get_criterion(self) -> nn.Module:
        """損失計算を行う関数オブジェクトの取得"""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self) -> optim.Optimizer:
        """オプティマイザの取得"""
        raise NotImplementedError

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        """学習用のデータローダの取得"""
        raise NotImplementedError

    def get_validation_loader(self) -> DataLoader:
        """検証用のデータローダの取得"""
        return None

    def train(self, epoch=20, additional_callbacks=None):
        """学習を実施する．

        Args:
          epoch (int): 最長エポック数
          additional_callbacks (list): 追加コールバック関数のリスト
        """
        criterion = self.get_criterion()
        if self.device is not None:
            criterion.to(self.device)
        optimizer = self.get_optimizer()
        callbacks = [
            StandardReporter(),
            CsvWriterReporter(self.get_csv_log_filename()),
            Snapshot(final_filename=self.get_model_filename()),
        ]
        if additional_callbacks is not None:
            callbacks.extend(additional_callbacks)
        trainer = TorchTrainer(
            self.autoencoder,
            criterion,
            optimizer,
            self.get_train_loader(),
            self.get_validation_loader(),
            epoch=epoch,
            callbacks=callbacks,
            device=self.device,
        )
        trainer.train()

    def get_model_filename(self, option=''):
        """モデル保存用のファイル名を取得"""
        filename = "%s+%s+%s.torch" % (self.autoencoder.get_filename_base(),
                                       self.get_filename_base(), option)
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    def get_csv_log_filename(self, option=''):
        """学習ログ用のCSVファイルのファイル名を取得"""
        filename = "%s+%s+%s.csv" % (self.autoencoder.get_filename_base(),
                                     self.get_filename_base(), option)
        fullpath = path.join(config.get_package_data_dir(__package__),
                             filename)
        return fullpath

    def save(self, upload=False):
        """モデルパラメタを保存する
        
        Args:
          upload (bool): 保存したファイルをGoogle DriveにアップロードするならTrue
        """
        filename = self.get_model_filename()
        self.autoencoder.save_weights(filename)
        if upload is True:
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename))

    def load(self, download=False, map_location=None):
        """モデルパラメタを読み込む．
        
        Args:
          download (bool): ファイルをGoogle Driveからダウンロードする場合はTrue
          map_location: モデルパラメタの読み込み先
        """
        filename = self.get_model_filename()
        if download is True or not path.exists(filename):
            g = GoogleDriveInterface()
            g.download(path.basename(filename), filename)
        self.autoencoder.load_weights(filename, map_location)

    def upload_csv_log(self):
        """CSVログをGoogle Driveにアップロードする"""
        filename = self.get_csv_log_filename()
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename), mediaType='text/csv')


def construct_autoencoder(version):
    """オートエンコーダをバージョン指定で構築する"""
    module_name = "sflib.speech.feature.autoencoder_pytorch.autoencoder%04d" \
        % version
    class_name = "SpectrogramImageAutoEncoder%04dPytorch" % version
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    autoencoder = cls()
    return autoencoder


def construct_trainer(module_postfix, class_postfix, device):
    """学習器をモジュール名の最後の部分ととクラス名の最後の部分で構築する"""
    module_name = "sflib.speech.feature.autoencoder_pytorch.trainer_%s" \
        % module_postfix
    class_name = "SpectrogramImageAutoEncoderTrainer%s" \
        % class_postfix
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    trainer = cls(device)
    return trainer


def load(autoencoder_version,
         trainer_module_postfix,
         trainer_class_postfix,
         device=None,
         map_location=None):
    """学習器の構築とパラメタの読み込みまで行う"""
    autoencoder = construct_autoencoder(autoencoder_version)
    autoencoder.to(device) #追加
    trainer = construct_trainer(trainer_module_postfix, trainer_class_postfix, device) # device追加
    trainer.set_autoencoder(autoencoder)
    trainer.load(map_location=map_location)
    return trainer
