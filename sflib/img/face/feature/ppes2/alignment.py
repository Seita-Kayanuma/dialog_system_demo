# coding: UTF-8
import pkgutil
import os
from os import path
from io import StringIO
import numpy as np
import pandas as pd
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from ..... import config

alignment_data_path = path.join(config.get_package_data_dir(__package__),
                                'data', 'alignment_data.npz')


def __read_csv(filename):
    """パッケージ内のCSVファイルを読み込みDataFrameとして返す"""
    data = pkgutil.get_data(__package__, filename)
    df = pd.read_csv(StringIO(data.decode('utf-8')), header=None)
    return df


def make_alignment_data():
    # 想定するテンプレート画像の大きさ
    DIM = 64

    # テンプレートの作成
    template = __read_csv('data/shape_template.csv').values.astype(np.float32)
    tmin = template.min(axis=0)
    tmax = template.max(axis=0)
    template = (template - tmin) / (tmax - tmin)

    # 三角形情報の読み込み
    triangles = __read_csv('data/shape_triangles.csv').values

    # 三角形画像の作成
    # 三角形画像の初期値
    triangle_img = np.ones((DIM, DIM), dtype=np.int64) * -1

    # xとyの点の組み合わせを生成させるため
    xx, yy = np.meshgrid(range(DIM), range(DIM))
    xx = xx.ravel()
    yy = yy.ravel()

    # 各三角形の番号で画像を埋めていく
    for i, t in enumerate(triangles):
        # print (i, t, '...', end='')
        polygon = Polygon(DIM * template[t])
        count = 0
        for x, y in zip(xx, yy):
            if Point(x, y).within(polygon):
                triangle_img[y, x] = i
                count += 1
        # print (count)

    # 保存
    if not path.exists(path.dirname(alignment_data_path)):
        os.makedirs(path.dirname(alignment_data_path),
                    mode=0o755,
                    exist_ok=True)
    np.savez(alignment_data_path, \
             template=template,
             triangles=triangles,
             triangle_img=triangle_img)


class FaceAligner:
    def __init__(self):
        if not path.exists(alignment_data_path):
            print("alignment data file is not found.")
            print("generating alignment data file (this takes a few moment)")
            make_alignment_data()

        with np.load(alignment_data_path) as f:
            self.template = f['template']
            self.triangles = f['triangles']
            self.triangle_img = f['triangle_img']
            self.dim = self.triangle_img.shape[0]
            self.mask = np.array(self.triangle_img >= 0, np.uint8) * 255

    def align(self, image, landmarks):
        '''
        画像を正規化する

        引数:
            image (numpy.ndarray): 入力画像
            landmarks (list or numpy.ndarray): 入力画像から検出されたランドマーク

        戻り値:
            numpy.ndarray: 出力画像
        '''
        if isinstance(landmarks, (tuple, list)):
            landmarks = np.array(landmarks)

        # 出力画像 (初期は真っ黒)
        # alignedImage = np.zeros((self.dim, self.dim, 3), image.dtype)
        alignedImage = np.zeros((self.dim, self.dim), image.dtype)

        # 各三角形（インデクス付き）でイテレート
        for i, triangle in enumerate(self.triangles):
            # マスク画像の生成（テンプレート画像上で対応する三角形のみが1）
            mask = np.uint8(self.triangle_img == i)
            # mask = np.stack((mask, ) * 3, -1)

            # 検出された三角形のバウンディングボックスの左上と右下
            minPosition = landmarks[triangle].min(0)
            maxPosition = landmarks[triangle].max(0)

            # if np.any(minPosition < 0) or \
            # maxPosition[0] >= image.shape[1] or \
            # maxPosition[1] >= image.shape[0]:
            # continue
            minPosition[0] = max(minPosition[0], 0)
            minPosition[1] = max(minPosition[1], 0)
            maxPosition[0] = min(maxPosition[0], image.shape[1])
            maxPosition[1] = min(maxPosition[1], image.shape[0])
            
            # 検出された三角形のバウンディングボックスの左上が原点となるように変換してから
            # 三角形間のアフィン変換を求める
            affineTransform = cv2.getAffineTransform(
                np.apply_along_axis(lambda position: position - minPosition, 1,
                                    landmarks[triangle]).astype(
                                        self.template.dtype),
                self.dim * self.template[triangle])

            # バウンディングボックス内のピクセルのみワープ
            warpedImage = cv2.warpAffine(
                image[minPosition[1]:(maxPosition[1] +
                                      1), minPosition[0]:(maxPosition[0] + 1)],
                affineTransform, (self.dim, self.dim))

            # マスクして出力画像に足す
            alignedImage += warpedImage * mask

        return alignedImage

    def get_alignment_info(self, landmarks):
        """64x64x2で，[:, :, 0]はx方向の変位，[:, :, 1]はy方向の変位を
        持つ行列を返す．
        全体が0.0〜1.0の大きさであるときの，元の位置からのずれを表す．
        dtypeはnp.float32
        """
        if isinstance(landmarks, (tuple, list)):
            landmarks = np.array(landmarks)

        # 出力情報
        result = np.zeros((self.dim, self.dim, 2), dtype=np.float32)
        # 元の位置の座標 [:, :, 0] が x，[:, :, 1] が y（のはず）
        orig_x = np.arange(0.0, 1.0, 1 / self.dim)
        orig = np.stack(np.meshgrid(orig_x, orig_x), -1)

        # 各三角形（インデクス付き）でイテレート
        for i, triangle in enumerate(self.triangles):
            # マスク画像の生成（テンプレート画像上で対応する三角形のみが1）
            mask = np.float32(self.triangle_img == i)
            mask = np.stack((mask, ) * 2, -1)

            affineTransform = cv2.getAffineTransform(
                self.template[triangle],
                landmarks[triangle])

            # 座標変換をする
            def warp(points, t):
                return np.matmul(
                    np.concatenate([points, np.ones((points.shape[0], 1))], axis=1),
                    t.T)
            
            tmp_result = \
                warp(orig.reshape(-1, 2), affineTransform).reshape(self.dim, self.dim, 2) \
                - orig

            # マスクして出力画像に足す
            result += tmp_result * mask

        return result

    def align_with_eq_hist(self, image, landmarks):
        img_out = self.align(image, landmarks)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        img_out = cv2.bitwise_and(img_out, self.mask)
        hist, bins = np.histogram(img_out, 256, [0, 255])
        hist[0] = 0
        cdf = hist.cumsum()
        tbl = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        tbl = tbl.astype('uint8')
        return tbl[img_out]
