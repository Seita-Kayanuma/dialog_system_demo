# coding: utf-8
import warnings
from os import path
from ..... import config
from .....cloud.google import GoogleDriveInterface
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np


class FaceAutoEncoder(nn.Module):
    """
    FaceAutoEncoderの基底クラス
    """

    def __init__(self):
        super(FaceAutoEncoder, self).__init__()

    def get_filename_base(self):
        """
        パラメタ保存のためのファイル名の元となる文字列を返す．
        """
        return self.__class__.__name__

    @property
    def bottleneck_dim(self):
        raise NotImplementedError()

    def forward(self, x):
        """
        学習時のためのフォワード計算．
        入出力共に
        0.0〜1.0 の正規化されたピクセル値を持つ，
        shapeが (batch, 96, 96) の
        torch.Tensor型（torch.float32）の画像情報である．
        継承クラスで実装された _encode，_decode を通じて計算される．
        """
        # チャネル情報を付与する
        x = x.reshape((-1, 1, 96, 96))
        v = self._encode(x)
        y = self._decode(v)
        # チャネル情報を削除する
        y = y.reshape((-1, 96, 96))
        return y

    def get_device(self):
        """
        デバイスを取得する
        """
        return next(self.parameters()).device

    def encode(self, img):
        """
        画像をエンコードしてベクトルにする．
        入力は
        0〜255 のピクセル値を持つ，
        shape が (batch, 96, 96) で，
        numpy.array型（数値の型はなんでもよい）の画像情報．
        出力は，
        shape が (batch, bottleneck_dim) で，
        numpy.array型（np.float32型）のベクトル．
        """
        x = np.float32(img.reshape(-1, 1, 96, 96) / 255.0)
        x = torch.tensor(x)
        if self.get_device().type != 'cpu':
            x = x.to(self.get_device())
        y = self._encode(x)
        y = y.detach()
        if self.get_device().type != 'cpu':
            y = y.cpu()
        y = np.float32(y.numpy())
        y = y.reshape(-1, self.bottleneck_dim)
        return y

    def decode(self, x):
        """
        エンコードの結果得られたベクトルを元に画像を再構築する
        入力は
        shape が (batch, bottleneck_dim) で，
        numpy.array型（np.float32型）のベクトル．
        出力は，
        0〜255 のピクセル値を持つ，
        shape が (batch, 96, 96) で，
        numpy.array型（np.uint8型）の画像情報．
        """
        x = np.float32(x.reshape(-1, 1, self.bottleneck_dim))
        x = torch.tensor(x)
        if self.get_device().type != 'cpu':
            x = x.to(self.get_device())
        y = self._decode(x)
        y = y.detach()
        if self.get_device().type != 'cpu':
            y = y.cpu()
        y = y.numpy()
        y = np.uint8(y.reshape(-1, 96, 96) * 255.0)
        return y

    def save_weights(self, filename):
        self.eval()
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename, map_location=None):
        self.eval()
        self.load_state_dict(torch.load(filename, map_location=map_location))


class FaceAutoEncoderTrainer:
    def __init__(self):
        self.autoencoder = None

    def get_filename_base(self):
        return self.__class__.__name__

    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder

    def get_train_dataset(self):
        pass

    def get_validation_dataset(self):
        return None

    def train(self, epochs=20, batch_size=256, shuffle=True, device=None):
        train_dataset = self.get_train_dataset()
        validation_dataset = self.get_validation_dataset()

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
        num_train_loops = len(train_dataset) // train_loader.batch_size + 1
        if validation_dataset:
            validation_loader = DataLoader(validation_dataset,
                                           batch_size=batch_size)
            num_validation_loops = len(
                validation_dataset) // validation_loader.batch_size + 1

        if device:
            self.autoencoder = self.autoencoder.to(device)
            self.crit = self.crit.to(device)
        self.optimizer = optim.Adam(self.autoencoder.parameters())

        for epoch in range(epochs):
            print("Epoch %02d/%02d" % (epoch, epochs))

            train_total_loss = 0
            self.autoencoder.train()
            for i, x in enumerate(train_loader):
                print("\rTraining ... %03d/%03d" % (i, num_train_loops),
                      end="")

                if device:
                    x = x[0].to(device)
                else:
                    x = x[0]

                y = self.autoencoder(x)
                loss = self.crit(y, x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if device:
                    train_total_loss += loss.detach().cpu().numpy()
                else:
                    train_total_loss += loss.detach().numpy()
            print("")
            if validation_dataset:
                self.autoencoder.eval()
                validation_total_loss = 0
                for i, x in enumerate(validation_loader):
                    print("\rValidating ... %03d/%03d" %
                          (i, num_validation_loops),
                          end="")

                    if device:
                        x = x[0].to(device)
                    else:
                        x = x[0]

                    y = self.autoencoder(x)
                    loss = self.crit(y, x)

                    if device:
                        validation_total_loss += loss.detach().cpu().numpy()
                    else:
                        validation_total_loss += loss.detach().numpy()

            print("\nTrain Loss: %.3f" %
                  (train_total_loss / num_train_loops, ),
                  end="")
            if validation_dataset:
                print(", Validation Loss: %.3f" %
                      (validation_total_loss / num_validation_loops, ),
                      end="")
            print("")


def train_face_auto_encoder(autoencoder, trainer):
    trainer.set_autoencoder(autoencoder)
    trainer.train()


# ------------
def get_weights_filename(autoencoder, trainer, option=''):
    filename = "%s+%s+%s.h5" % (autoencoder.get_filename_base(),
                                trainer.get_filename_base(), option)
    fullpath = path.join(config.get_package_data_dir(__package__), filename)
    return fullpath


def save_face_auto_encoder_weights(autoencoder, trainer, upload=False):
    # if trainer.history is None:
    #    warnings.warn("autoencoder is not trained yet")
    filename = get_weights_filename(autoencoder, trainer)
    autoencoder.save_weights(filename)
    if upload is True:
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename))


def load_face_auto_encoder_weights(autoencoder,
                                   trainer,
                                   download=False,
                                   map_location=None):
    filename = get_weights_filename(autoencoder, trainer)
    if download is True:
        g = GoogleDriveInterface()
        g.download(path.basename(filename), filename)
    autoencoder.load_weights(filename, map_location=map_location)
