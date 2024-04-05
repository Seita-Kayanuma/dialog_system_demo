import numpy as np
from .base import FacialExpressionFeatureExtractor
# 顔画像オートエンコーダ（PyTorch版）関連
from ....img.face.feature.autoencoder_pytorch.autoencoder_0004 \
    import FaceAutoEncoder0004PyTorch as FaceAutoEncoder
from ....img.face.feature.autoencoder_pytorch.trainer_lfw_0001 \
    import FaceAutoEncoderTrainerLFW0001 as FaceAutoEncoderTrainer
from ....img.face.feature.autoencoder_pytorch.base \
    import load_face_auto_encoder_weights


def normalize_shape(shapes):
    """
    形状（68点のランドマーク情報）を，縦横 0.0 〜 1.0 の値に正規化する

    入力は（サンプル数，ランドマーク数，2）の形状をしている必要がある
    """
    # 最小値と最大値を求める
    s_min = shapes.min(axis=1, keepdims=True)
    s_max = shapes.max(axis=1, keepdims=True)
    # 正規化
    return (shapes - s_min) / (s_max - s_min + 1e-10)


class FacialExpressionFeatureExtractor0003(FacialExpressionFeatureExtractor):
    def __init__(self, use_image=True, use_shape=True):
        self.use_image = use_image
        self.use_shape = use_shape

        # 顔画像オートエンコーダのロード
        if self.use_image:
            self.face_autoencoder = FaceAutoEncoder()
            face_autoencoder_trainer = FaceAutoEncoderTrainer()
            load_face_auto_encoder_weights(self.face_autoencoder,
                                           face_autoencoder_trainer)

        if use_image is False and use_shape is False:
            raise ValueError("either use_image or use_shape must be True")

    def get_filename_base(self):
        # 画像を使うか形状を使うかの記号も付与しておく
        s = super().get_filename_base()
        s += '_'
        if self.use_image:
            s += 'I'
        if self.use_shape:
            s += 'S'
        return s

    def calc(self, images, shapes):
        xs = []
        if self.use_image:
            xs.append(self.face_autoencoder.encode(images))
        if self.use_shape:
            num_samples = shapes.shape[0]
            xs.append(normalize_shape(shapes).reshape((num_samples, -1)))
        return np.concatenate(xs, axis=1)

    def get_feature_dim(self):
        r = 0
        if self.use_image:
            r += self.face_autoencoder.encoded_dim
        if self.use_shape:
            r += 68 * 2
        return r
        
