import warnings
from os import path
import config
from cloud.google import GoogleDriveInterface
from ext.torch.trainer import TorchTrainer
from ext.torch.callbacks.reporters import StandardReporter, CsvWriterReporter
from ext.torch.callbacks.snapshot import Snapshot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import abstractmethod
import re
import glob


class SpectrogramImageAutoEncoder(nn.Module):
    """スペクトログラム画像の自己符号化器
    """

    DEFAULT_TRAINER_NUMBER = 6
    """int:
    学習器番号のデフォルト値．事情があって1からではなく6．
    """

    def __init__(self, trainer_number=DEFAULT_TRAINER_NUMBER):
        # class name check
        m = re.match(r'SpectrogramImageAutoEncoder(\d+)',
                     self.__class__.__name__)
        if m is None:
            raise RuntimeError('class name should match with' +
                               'r"SpectrogramImageAutoEncoder\\d+"')
        self.__number = int(m[1])

        # super(SpectrogramImageAutoEncoder, self).__init__()
        super().__init__()
        self.__first_parameter = None
        self.__trainer_number = trainer_number

    @property
    def filename_base(self):
        """パラメータ保存などのためのファイル名のヒント"""
        return 'SIAE{:02d}T{:02d}'.format(self.__number, self.__trainer_number)

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
            shape[1]はチャネルサイズで1（カラー画像を入力にする予定は当面無いので
            1で固定）

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

    def get_latest_model_version(self):
        """保存済の学習モデルの最新バージョン番号を取得する"""
        pattern = '{}.[0-9]*.torch'.format(self.filename_base)
        pattern = path.join(config.get_package_data_dir(__package__),
                            pattern)
        paths = glob.glob(pattern)
        version = None
        pat = re.compile(r'{}\.(\d+)\.torch'.format(self.filename_base))
        for p in paths:
            m = pat.match(path.basename(p))
            if m is None:
                continue
            v = int(m[1])
            if version is None or version < v:
                version = v
        return version

    def get_model_filename_base(self, version=None, overwrite=False):
        """学習モデルファイルの名前（拡張子を除く）を取得する.

        Args:
          version: 明示的にバージョンを指定する場合はその番号．
                   Noneの場合は最新のものになる．
          overwrite: version=Noneのとき，このオプションがFalseだと最新+1の
                   バージョンのファイル名となる
        """
        if version is None:
            version = self.get_latest_model_version()
            if version is None:
                version = 0
            elif not overwrite:
                version += 1
        filename_base = "{}.{:02d}".format(self.filename_base, version)
        return filename_base

    def get_csv_log_filename(self, version=None, overwrite=False):
        """学習ログを保存するファイル名を取得する"""
        filename = self.get_model_filename_base(version, overwrite) + ".csv"
        filename = path.join(config.get_package_data_dir(__package__),
                             filename)
        return filename

    def get_model_filename(self, version=None, overwrite=False):
        filename = self.get_model_filename_base(version, overwrite) + ".torch"
        filename = path.join(config.get_package_data_dir(__package__),
                             filename)
        return filename
    
    def save(self, version=None, overwrite=False, upload=True):
        """モデルパラメタの保存
        
        Args:
          version: バージョン番号．Noneの場合は最新版として保存する.
          overwrite: Trueの場合，最新バージョンのファイルに上書きする．
        """
        filename = self.get_model_filename(version, overwrite)
        self.eval()
        torch.save(self.state_dict(), filename)
        if upload is True:
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename))

    def load(self, version=None, download=False, download_overwrite=False):
        """モデルパラメタの読み込み
        
        Args:
          version: バージョン番号. Noneの場合は最新のものを読み込む.
        """
        if download is True:
            g = GoogleDriveInterface()
            g.download_with_filename_pattern(
                self.filename_base,
                r"{}.\d+.torch".format(self.filename_base),
                config.get_package_data_dir(__package__),
                overwrite=download_overwrite)
        if version is None:
            version = self.get_latest_model_version()
        if version is None:
            raise RuntimeError('file not found')
        filename = "{}.{:02d}.torch".format(self.filename_base, version)
        filename = path.join(config.get_package_data_dir(__package__),
                             filename)
        
        self.eval()
        self.load_state_dict(torch.load(filename, map_location=self.device))

    def force_load_model_file(self, filename):
        """強制的にモデルファイルを読み込む．
        同じ構造をもつ別のモデルファイルを読み込む場合などに使う．
        """
        self.load_state_dict(torch.load(filename, map_location=self.device))
        
    def upload_csv_log(self):
        """（既存の最新）CSVログをGoogle Driveにアップロードする"""
        filename = self.get_csv_log_filename(overwrite=True)
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename), mediaType='text/csv')

    def train_autoencoder(self, epoch=20, additional_callbacks=None):
        trainer = construct_trainer(self.__trainer_number)
        trainer.train(self, epoch, additional_callbacks)
        
        
class SpectrogramImageAutoEncoderTrainer:
    """スペクトルオートエンコーダの学習器の基底クラス
    """
    def __init__(self):
        self._device = None

    @abstractmethod
    def get_criterion(self) -> nn.Module:
        """損失計算を行う関数オブジェクトの取得"""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, model) -> optim.Optimizer:
        """オプティマイザの取得"""
        raise NotImplementedError

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        """学習用のデータローダの取得"""
        raise NotImplementedError

    def get_validation_loader(self) -> DataLoader:
        """検証用のデータローダの取得"""
        return None

    def get_additional_callbacks(self) -> list:
        return []

    def train(self, autoencoder: SpectrogramImageAutoEncoder,
              epoch=20, additional_callbacks=None):
        """学習を実施する．

        Args:
          autoencoder: 学習対象のオートエンコーダ
          epoch (int): 最長エポック数
          additional_callbacks (list): 追加コールバック関数のリスト
        """
        device = autoencoder.device
        self._device = device
        criterion = self.get_criterion()
        if device is not None:
            criterion.to(device)
        optimizer = self.get_optimizer(autoencoder)
        callbacks = [
            StandardReporter(),
            CsvWriterReporter(autoencoder.get_csv_log_filename()),
            # Snapshot(final_filename=autoencoder.get_model_filename()),
        ]
        callbacks.extend(self.get_additional_callbacks())
        if additional_callbacks is not None:
            callbacks.extend(additional_callbacks)

        trainer = TorchTrainer(
            autoencoder,
            criterion,
            optimizer,
            self.get_train_loader(),
            self.get_validation_loader(),
            epoch=epoch,
            callbacks=callbacks,
            device=device
        )
        trainer.train()

        
def construct_autoencoder(autoencoder_number, trainer_number) -> SpectrogramImageAutoEncoder:
    """オートエンコーダの構築

    Args:
      autoencoder_number: オートエンコーダ番号
      trainer_number: 学習器番号
    """
    autoencoder_module_name = "speech.feature.autoencoder_v2." + "autoencoder{:04d}".format(autoencoder_number)
    #autoencoder_module_name = "sflib.speech.feature.autoencoder_v2." + "autoencoder{:04d}".format(autoencoder_number)
    autoencoder_class_name = "SpectrogramImageAutoEncoder{:04d}".format(autoencoder_number)
    import importlib
    mod = importlib.import_module(autoencoder_module_name)
    cls = getattr(mod, autoencoder_class_name)
    autoencoder = cls(trainer_number=trainer_number)
    return autoencoder


def construct_trainer(trainer_number):
    """学習器を構築する

    Args:
      trainer_number: 学習器番号
    """
    trainer_module_name = "speech.feature.autoencoder_v2." + "trainer{:04d}".format(trainer_number)
    #trainer_module_name = "sflib.speech.feature.autoencoder_v2." + "trainer{:04d}".format(trainer_number)
    trainer_class_name = "SpectrogramImageAutoEncoderTrainer{:04d}".format(trainer_number)
    import importlib
    mod = importlib.import_module(trainer_module_name)
    cls = getattr(mod, trainer_class_name)
    trainer = cls()
    return trainer
