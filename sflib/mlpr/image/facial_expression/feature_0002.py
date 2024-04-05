# HoG特徴量を抽出する
import numpy as np
from .base import FacialExpressionFeatureExtractor
import cv2


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


class FacialExpressionFeatureExtractor0002(FacialExpressionFeatureExtractor):
    def __init__(self, use_image=True, use_shape=True):
        self.use_image = use_image
        self.use_shape = use_shape

        # 顔画像オートエンコーダのロード
        if self.use_image:
            size = 8
            winSize = (64, 64)
            cellSize = (size, size)
            blockSize = (size, size)
            blockStride = (size, size)
            nbins = 9
            derivAperture = 1
            winSigma = 4.
            histogramNormType = 0
            L2HysThreshold = 2.0000000000000001e-01
            gammaCorrection = 0
            nlevels = 64
            self.hog = cv2.HOGDescriptor(
                winSize, blockSize, blockStride, cellSize, nbins,
                derivAperture, winSigma, histogramNormType, L2HysThreshold,
                gammaCorrection, nlevels)

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
            num_images = images.shape[0]
            xsi = []
            for i in range(num_images):
                x = self.hog.compute(images[i], (0, 0), (0, 0), ((0, 0),))
                if x is None:
                    xsi.append(np.zeros((1, self.hog.getDescriptorSize())))
                else:
                    xsi.append(x.T)
            xs.append(np.concatenate(xsi, axis=0))
        if self.use_shape:
            num_samples = shapes.shape[0]
            xs.append(normalize_shape(shapes).reshape((num_samples, -1)))
        return np.concatenate(xs, axis=1)

    def get_feature_dim(self):
        r = 0
        if self.use_image:
            r += self.hog.getDescriptorSize()
        if self.use_shape:
            r += 68 * 2
        return r
