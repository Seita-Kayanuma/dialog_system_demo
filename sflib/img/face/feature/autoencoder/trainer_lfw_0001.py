# coding: utf-8
import numpy as np
from keras import backend as K
from sflib.img.face.alignment import FaceAligner
from sflib.corpus.lfw.process import AlignedFaces
from .base import FaceAutoEncoderTrainer

# 顔画像のマスクを初期化
mask = FaceAligner().mask
mask = K.variable((mask > 0).astype('float64'))


def masked_mean_squared_error(y_true, y_pred):
    """
    マスク上のみの平均二乗誤差を計算する
    """
    return K.sqrt(K.mean(K.pow((y_pred - y_true) * mask, 2.0), axis=[1, 2, 3]))


class FaceAutoEncoderTrainerLFW0001(FaceAutoEncoderTrainer):
    def __init__(self):
        super().__init__()

        self.x_train = None
        self.x_test = None

    def build_data(self):
        # 学習データ，テストデータの生成
        af = AlignedFaces()
        ap = np.float64(af.dataframe.iloc[:, (2 + 4 + 68 * 2):])
        ap = ap.reshape(-1, 96, 96)
        ap_flipped = np.flip(ap, 2)

        ap_train = np.stack([ap[:12000], ap_flipped[:12000]], axis=0)
        ap_test = np.stack([ap[12000:], ap_flipped[12000:]], axis=0)

        x_train = ap_train.reshape(-1, 96, 96, 1)
        x_train /= 255
        self.x_train = x_train
        x_test = ap_test.reshape(-1, 96, 96, 1)
        x_test /= 255
        self.x_test = x_test

    def compile(self, *args, **kwargs):
        super().compile(
            *args, optimizer='adam', loss=masked_mean_squared_error, **kwargs)

    def get_train_data(self):
        if self.x_train is None:
            self.build_data()
        return (self.x_train, self.x_train)

    def get_validation_data(self):
        if self.x_test is None:
            self.build_data()
        return (self.x_test, self.x_test)
