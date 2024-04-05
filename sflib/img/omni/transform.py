# coding: utf-8
"""
全方位画像から，特定の位置に注目した平面画像を抽出する
"""
__author__ = "Shinya FUJIE <shinya.fujie@p.chibakoudai.jp>"
__version__ = "0.0.0"
__date__ = "6 Sept 2018"

import cv2
import numpy as np
import math
from . import util


def transformBilinear(img, table):
    """
    バイリニア補間で画像変換をする．

    Parameters
    ----------
    img : numpy.array (Hi x Wi x C) uint8
        入力画像.
    table : numpy.array (Ho x Wo x 2) float
        出力画像サイズで，対応する入力画像のx, y座標が入っている行列
    
    Returns
    -------
    img_out : numpy.array (Ho x Wo x C)
    """
    hi, wi, c = img.shape
    out_shape = table.shape[:2] + (c,)

    # # xo, yo: 出力画像上のx, y座標の組み合わせを並べたベクトル
    # xo = np.array(range(out_shape[1]), np.int32)
    # yo = np.array(range(out_shape[0]), np.int32)
    # xo, yo = np.meshgrid(xo, yo)
    # xo = xo.ravel()
    # yo = yo.ravel()

    # xi, yi: 入力画像上のx, y座標の組み合わせを並べた行列
    xi = table[:, :, 0]
    yi = table[:, :, 1]

    xi0 = np.int32(xi)
    yi0 = np.int32(yi)
    xi1 = np.int32(xi) + 1
    yi1 = np.int32(yi) + 1

    # 重み
    dx = xi - xi0
    dy = yi - yi0
    w00 = (1 - dx) * (1 - dy)
    w01 = dx * (1 - dy)
    w10 = (1 - dx) * dy
    w11 = dx * dy

    # すべての座標が内側に入っている必要がある
    cond = ~((xi0 >= 0) & (xi0 < wi) & (yi0 >= 0) & (yi0 < hi) & (xi1 >= 0) & (
        xi1 < wi) & (yi1 >= 0) & (yi1 < hi))

    xi0[cond] = 0
    yi0[cond] = 0
    xi1[cond] = 0
    yi1[cond] = 0

    # コピーする
    # img_00 = np.zeros(out_shape, np.uint8)
    # img_00[yo, xo, :] = img[yi0, xi0, :]
    img_00 = img[yi0, xi0, :]
    # img_01 = np.zeros(out_shape, np.uint8)
    # img_01[yo, xo, :] = img[yi0, xi1, :]
    img_01 = img[yi0, xi1, :]
    # img_10 = np.zeros(out_shape, np.uint8)
    # img_10[yo, xo, :] = img[yi1, xi0, :]
    img_10 = img[yi1, xi0, :]
    # img_11 = np.zeros(out_shape, np.uint8)
    # img_11[yo, xo, :] = img[yi1, xi1, :]
    img_11 = img[yi1, xi1, :]

    img_o = img_00 * np.stack([w00.reshape(out_shape[:2])] * 3, axis=2) + \
        img_01 * np.stack([w01.reshape(out_shape[:2])] * 3, axis=2) + \
        img_10 * np.stack([w10.reshape(out_shape[:2])] * 3, axis=2) + \
        img_11 * np.stack([w11.reshape(out_shape[:2])] * 3, axis=2)
    img_o[cond] = 0
    
    return img_o.astype(np.uint8)


class OmniImageTransformer:
    """
    全方位画像から，特定の位置に注目した平面画像を抽出するためのクラス．
    """

    def __init__(self, R=320, w=160, h=160, near=100):
        # 最大角度[rad]（カメラとレンズのセッティングで決まる）
        self.theta_max = util.deg2rad(107)
        self.R = R
        self.w = w
        self.h = h
        self.near = near
        self.generateTable()

    def generateTable(self):
        # とりあえずtheta方向を30分割し，phi 0度の変換テーブルを用意する．
        # 実際の変換時は前後のstを線形補完した st を phi 度回転させることで
        # 近似変換テーブルとする作戦

        # テーブルを生成する角度
        self.theta_keys = np.linspace(0, self.theta_max, 30)
        # テーブルリスト（角度と実際のテーブルのタプルが要素）
        self.table_list = []

        # 各テーブルの共通事項はあらかじめ作成しておく
        # x と y はそれぞれ変換後の画像のX座標とY座標
        x = np.array(range(self.w))
        y = np.array(range(self.h))
        # xx とyy は x と y の組み合わせを考慮したベクトル
        xx, yy = np.meshgrid(x, y)
        xx = xx.ravel()
        yy = yy.ravel()
        # xy は xx と yy を縦に並べて，2xピクセル数の行列にしたもの
        self.xy = np.stack([xx, yy])

        # dxxとdyyは，画像中心を原点としたときの xx と yy
        # // ただし変換後のXは逆方向になっている（方向の整合性のため）
        dxx = xx - self.w / 2
        dyy = yy - self.h / 2
        # dは，視線方向をz方向の単位ベクトル[0, 0, -1]Tとして変換後の画像の中心にした
        # 場合の画像上の各点方向のベクトルを並べたもの．
        d = np.stack([dxx / self.near, dyy / self.near, -np.ones(dxx.shape)])

        for theta in self.theta_keys:
            # Mo（視線方向変換行列）
            Mo = self.generateMo(theta, 0)
            # d を全方位画像空間上の視線方向にあわせたもの．
            v = np.dot(Mo, d)
            # v の各成分のthetaと，それに対応した全方位画像の中心からの距離 r
            theta_v = np.arccos(-v[2] / np.linalg.norm(v, axis=0))
            r_v = theta_v / self.theta_max * self.R
            # vを正規化して全方位画像上の座標[s, t]を並べた行列 st にする
            norm = np.linalg.norm(v[:2], axis=0)
            coef = r_v / (norm + 1e-18)
            st = coef * v[:2]
            # リストに追加
            self.table_list.append((theta, st))

    @staticmethod
    def generateMo(theta, phi):
        """
        Z方向のベクトルを視線方向へ変換する回転行列Moを生成する．

        Parameters
        ----------
        theta : float
            X軸回りの回転角度（ラジアン）
        phi   : float
            Z軸回りの回転角度

        Returns
        -------
        Mo : numpy.ndarray
            3x3の3次元回転行列
        """
        Mo_theta = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)],
        ])
        Mo_phi = np.array([
            [math.cos(phi), -math.sin(phi), 0.0],
            [math.sin(phi), math.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ])
        Mo = np.dot(Mo_phi, Mo_theta)
        return Mo

    def transform(self, img, cx, cy):
        """
        変換を行う．

        Parameters
        ----------
        img : numpy.ndarray
            入力画像（全方位画像）
        cx, cy : int
            変換後の中心に持ってくる点の入力画像中の位置
        """
        # 出力画像
        # img_o = np.zeros((self.h, self.w, 3), np.uint8)
        # 入力画像のサイズ
        wi, hi = img.shape[1], img.shape[0]
        wh = np.array([[wi], [hi]])
        # 入力画像の半径
        R = wi / 2
        # 注目点の原点からの相対位置
        cxr, cyr = cx - wi / 2, cy - hi / 2
        # 視線方向のtheta（仰角に相当．天井方向が0）
        theta = math.sqrt(cxr * cxr + cyr * cyr) / R * self.theta_max
        # 視線方向のphi（入力画像のY軸方向が0で時計回りがプラス）
        phi = math.atan2(cyr, cxr) - util.deg2rad(90) 

        # テーブルを生成（該当する前後のテーブルの線形補完）
        # thetaキーのうち，今回のthetaを超えない最後のキーを取得する
        found_keys = np.where(self.theta_keys <= theta)[0]
        if len(found_keys) == 0 or len(found_keys) >= len(self.theta_keys):
            print("角度が不正 %f, %d, %d" % (theta / np.pi * 180, cxr, cyr))
            return None, None
        i = found_keys[-1]
        theta_1 = self.table_list[i][0]
        theta_2 = self.table_list[i + 1][0]
        weight = 1.0 - (theta - theta_1) / (theta_2 - theta_1)
        st = self.table_list[i][1] * weight + self.table_list[i + 1][1] * (
            1 - weight)

        # phiだけ回転させる
        Mphi = np.array([
            [math.cos(phi), -math.sin(phi)],
            [math.sin(phi), math.cos(phi)],
        ])
        st = np.dot(Mphi, st)
        
        # 出力画像と同じshapeにする．
        st = st.T.reshape((self.h, self.w, -1))

        # sとtの原点を画像の原点（左上）にあわせる
        wh = wh.reshape((1, 1, -1))
        st = st + wh / 2
        t  = st
        
        # # 整数型に直す
        # st = np.int32(st)
        
        # # 入力画像外の位置を指している場合は座標を0にする（取り急ぎ）
        # cond1 = np.any(~((st >= 0) & (st < wh)), axis=2)
        # cond2 = np.stack([cond1] * 2, axis=2)
        # st[cond2] = 0

        # # コピー
        # img_o = img[st[:, :, 1], st[:, :, 0], :]

        # # 範囲外のピクセルは0にする
        # cond3 = np.stack([cond1] * 3, axis=2)
        # img_o[cond3] = 0
        
        img_o = transformBilinear(img, t)
        
        # 終了
        return img_o, t


if __name__ == "__main__":
    from bgs import BGSTracker
    import sys

    # 第1引数を入力動画のファイル名とする
    if len(sys.argv) < 2:
        print("{} <filename>".format(sys.argv[0]))
        sys.exit(-1)
    filename = sys.argv[1]

    # ビデオキャプチャ
    cap = cv2.VideoCapture(filename)

    # 入力画像（元のサイズだと大きすぎるのでこのサイズに縮小する）
    W, H = 640, 640
    # 変換後の画像サイズ
    CW, CH = 320, 320
    # ズームの割合（単位は不明だがこれくらいが妥当？小さければ引いた画像になり，
    # 大きければよった画像になる）
    NEAR = 400

    # 表示用の大きな画像
    img2show = np.zeros((H, W + CW * 2, 3), np.uint8)

    # 画面表示用の変換後画像の原点．
    # 変換後画像は最大4(8)枚表示する
    convert_image_pos = (
        (W, 0),
        (W, CH),
        # (W, CH * 2),
        # (W, CH * 3),
        (W + CW, 0),
        (W + CW, CH),
        # (W + CW, CH * 2),
        # (W + CW, CH * 3),
    )

    # 背景差分〜おおまかな顔の位置までを推定するオブジェクト
    bgst = BGSTracker()

    # 全方位画像から特定の位置を中心とした平面画像を生成するオブジェクト
    cvtr = OmniImageTransformer(R=W / 2, w=CW, h=CH, near=NEAR)

    # 何枚のフレームを処理したか
    count = 0

    rt = util.ResultTracker()

    timer = util.Timer()
    fpsc = util.FpsCalculator()

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    
    # 1枚ずつ処理
    while cap.isOpened():
        # キャプチャする
        timer.start()
        ret, frame = cap.read()
        timer.report("cap")

        # 縮小する
        frame = cv2.resize(frame, (W, H))

        # 表示用にコピーする
        frame2show = frame.copy()

        # 背景差分〜おおまかな顔の位置までを推定
        timer.start()
        bgst.apply(frame)
        timer.report("bgs")

        # 処理枚数をインクリメント
        count += 1
        # 処理枚数が100枚を超えて入れば画像変換処理をする
        # （ある程度の枚数を処理しないと背景差分がまともに動作しないため）
        if count > 0:
            results = rt.track(bgst.results)

            # おおまかな顔の位置ごとに，表示用フレーム画像にマーカーを描画する
            for angle, x, y, dist in results:
                cv2.drawMarker(frame2show, (x, y), (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 3)
            img2show[:, :, :] = 0
            img2show[:H, :W, :] = frame2show

            # おおまかな顔の位置ごとに変換画像を生成する
            timer.start()
            i = 0
            for angle, x, y, dist in results:
                img_o, _ = cvtr.transform(frame, x, y)
                if img_o is None:
                    continue
                # 表示用画像にコピーする
                img2show[convert_image_pos[i][1]:convert_image_pos[i][1] + CH,
                         convert_image_pos[i][0]:convert_image_pos[i][0] + CW, :] \
                         = img_o
                i += 1
                if i >= len(convert_image_pos):
                    break
            timer.report("cnv")
        else:
            img2show[:H, :W, :] = frame2show

        cv2.putText(img2show, "%5.2ffps" % fpsc.tick(), (0, H),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 1)
        
        cv2.imshow("image", img2show)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
