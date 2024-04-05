# coding: utf-8
"""
全方位画像向けの背景差分処理による人のおおよその位置の推定
"""
__author__ = "Shinya FUJIE <shinya.fujie@p.chibakoudai.jp>"
__version__ = "0.0.0"
__date__ = "6 Sept 2018"

import math
import numpy as np
import cv2
from . import util


class BGSTracker:
    """
    BackGround Subtraction Tracker

    与えられた全方位画像をもとに，
    (1) 背景差分法によるマスク画像を生成
    (2) オープニング処理による細かい点の除去
    (3) マスク画像をlog-polar画像に直して，角度毎の最短点を見つける
    (4) 見つかった再短点の前後（左右?）一定の範囲にある点は対象から除き，
        (3) を繰り返す．
    """

    def __init__(self,
                 input_resize_ratio=1.0,
                 kernel_size=3,
                 bundle_angle=30.0,
                 dist_offset=50):
        """
        Parameters
        ----------
        input_resize_ratio : float
            入力画像を縮小して実行する場合はこの値を1.0未満の値にする．
            出力画像は逆に拡大される．
        kernel_size : int
            探索をする前に行うオープニング処理（細かい点の除去処理）における
            カーネルの大きさ（幅，高さ共通）．
        bundle_angle : float
            前後何度までを同一人物とみなすか
        dist_offset : float
            見つかった点より何ピクセル遠くを対象の点とするか．
            本当は顔の中心を見つけたいが，頭頂が見つかってしまうための措置．
        """
        # 背景差分器
        self.bgs = cv2.bgsegm.createBackgroundSubtractorCNT()
        # 入力画像の拡大率
        self.input_resize_ratio = input_resize_ratio
        # オープニング処理用のカーネル
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 前後何度までを同一人物とみなすか．
        self.bundle_angle = bundle_angle
        # 点のオフセット
        self.dist_offset = dist_offset

    @staticmethod
    def _search_nonzero_points(img, bundle_angle, dist_offset):
        """
        """
        # 画像の高さ，幅
        h, w = img.shape[0], img.shape[1]
        # 画像の中心座標
        cx, cy = w / 2, h / 2
        # log-polar画像のM，半径(w/2)が画像の幅と同じになるようにする
        M = w / math.log(math.sqrt(2) * w / 2.0)
        # log-polar画像の計算
        img_polar = cv2.logPolar(img, (cx, cy), M, cv2.WARP_FILL_OUTLIERS)
        # 左から見て非ゼロになる最小列を取得する
        # 画像は0か255なので，最大値を与えるインデクスのリストをとればよい．
        # 0の場合は見つからなかったということになるので，後で（角度情報とあわせて削除する）．
        min_indices = np.argmax(img_polar, axis=1)
        # 対応する角度のarrayを作成する
        angles = np.linspace(0, 360, len(min_indices) + 1)[:-1]

        # 0を取り除く
        cond = min_indices != 0
        min_indices = min_indices[cond]
        angles = angles[cond]

        results = []

        # anglesがなくなるまで繰り返す
        while len(angles) > 0:
            # もっとも距離が小さい位置を取得する
            min_index = np.argmin(min_indices)
            # その角度
            min_angle = angles[min_index]
            # 元画像上の距離に直す
            ind = min_indices[min_index]
            dist = math.exp(ind / M) + dist_offset
            # 元画像上の位置を求める
            x = int(cx + dist * math.cos(util.deg2rad(min_angle)))
            y = int(cx + dist * math.sin(util.deg2rad(min_angle)))

            results.append([min_angle, x, y, dist])

            # 不要なところを削除する
            # まわりこみなしで削除するところ
            cond = (angles >
                    (min_angle - bundle_angle)) & (angles <
                                                   (min_angle + bundle_angle))
            # 下側のまわりこみを考慮
            cond_lower = (angles - 360) > (min_angle - bundle_angle)
            # 上側のまわりこみを考慮
            cond_upper = (angles + 360) < min_angle + bundle_angle
            cond = ~(cond | cond_lower | cond_upper)

            min_indices = min_indices[cond]
            angles = angles[cond]
        return results

    def apply(self, img):
        """
        全方位画像を元に，処理を行う．
        
        Parameters
        ----------
        img : numpy.ndarray
            処理する画像
        
        Returns
        -------
        results : list
            検出された位置ごとに以下の要素を持つ．
            角度（ラジアン）, X座標, Y座標, 中心からの距離
        """
        # 入力画像の幅と高さ
        h, w = img.shape[:2]
        if self.input_resize_ratio != 1.0:
            # 必要であればリサイズ
            img = cv2.resize(img, None, None, self.input_resize_ratio,
                             self.input_resize_ratio)
        # (1) 背景差分によりマスク画像を計算
        self.mask = self.bgs.apply(img, None)
        # (2) オープニング処理による細かい点の除去
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, self.kernel)
        # (3) self.search_angle 度刻みで中心から端に向けて最初にマスクの値が
        #     非ゼロになる位置の探索
        results = self._search_nonzero_points(self.mask, self.bundle_angle,
                                              self.dist_offset)
        self.results = results

        # 必要があればリサイズ
        if self.input_resize_ratio != 1.0:
            self.mask = cv2.resize(self.mask, (w, h), None, -1.0, -1.0,
                                   cv2.INTER_NEAREST)
            for r in self.results:
                r[1] = int(r[1] / self.input_resize_ratio)
                r[2] = int(r[2] / self.input_resize_ratio)
                r[3] /= self.input_resize_ratio
        return self.results


if __name__ == "__main__":
    import sys

    # 第1引数を入力動画のファイル名とする
    if len(sys.argv) < 2:
        print("{} <filename>".format(sys.argv[0]))
        sys.exit(-1)
    filename = sys.argv[1]

    # 入力画像（元のサイズだと大きすぎるのでこのサイズに縮小する）
    W, H = 640, 640

    # 表示用画像
    image2show = np.zeros((H, W * 2, 3), np.uint8)

    # ビデオキャプチャ
    cap = cv2.VideoCapture(filename)

    # 背景差分によるマスク計算と，各角度の非ゼロ点の計算
    # bgst = BGSTracker(input_resize_ratio=0.5, dist_offset=25)
    bgst = BGSTracker()

    # 時間計測用
    timer = util.Timer()
    fpsc = util.FpsCalculator()

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    
    # 1枚ずつ処理して表示
    while cap.isOpened():
        # キャプチャ
        timer.start()
        ret, frame = cap.read()
        timer.report("capture ")

        # 画像の縮小
        frame = cv2.resize(frame, (W, H))

        # 表示用フレームの準備
        frame2show = frame.copy()

        # 計算
        timer.start()
        bgst.apply(frame)
        timer.report("bgstrack")

        # 表示用マスク
        bg2show = cv2.cvtColor(bgst.mask, cv2.COLOR_GRAY2BGR)

        # 入力画像，マスク画像にマーカーを描画
        for angle, x, y, dist in bgst.results:
            for img in (frame2show, bg2show):
                cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20,
                               3)

        # 表示用画像の更新
        image2show[:H, :W, :] = frame2show
        image2show[:H, W:2 * W, :] = bg2show

        cv2.putText(image2show, "%5.2ffps" % fpsc.tick(), (0, H),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 1)
        
        # 表示
        cv2.imshow("image", image2show)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    # 後処理
    cap.release
    cv2.destroyAllWindows()
