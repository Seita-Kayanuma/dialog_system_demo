# coding: utf-8
import dlib
import cv2
import copy
from math import ceil, floor


class FaceDetector:
    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()

    def detect(self, img, resize_factor=1.0):
        # 検出．detsにはdlib.rectangleが入っている．
        # そのままでは扱いにくいので，(pt1, pt2) のリストにする
        # pt1は左上，pt2は右下の点の座標
        if resize_factor != 1.0:
            img = cv2.resize(img,
                             dsize=None,
                             fx=resize_factor,
                             fy=resize_factor)
        dets = self._detector(img, 1)
        results = [(
            (int(d.left() / resize_factor), int(d.top() / resize_factor)),
            (int(d.right() / resize_factor), int(d.bottom() / resize_factor)),
        ) for d in dets]
        return results


class FaceTracker:
    """ビデオなどの時間的に連続しているフレームにおいて
    軽量に顔の検出を行う"""

    def __init__(self,
                 resizing_face_width=50,
                 detection_window_ratio=3.0,
                 auto_retry=True):
        """
        Args:
          resizing_face_width (int): 直前に見つかった顔領域を合わせ込む幅
          detection_widow_ratio (float): 直前に見つかった顔領域を基準にした，
            探索領域の広さの比
          auto_retry (bool): 顔領域が見つからなかった場合に，全体に対して
            やり直しを行うかどうか．
        """
        self._prev_faces = []
        self._auto_retry = auto_retry
        self._detector = FaceDetector()
        self._resizing_face_width = resizing_face_width
        self._detection_window_ratio = detection_window_ratio
        self._detection_window_width = \
            self._resizing_face_width * self._detection_window_ratio

    def reset(self):
        self._prev_faces = []

    def detect(self, image):
        # 画像の幅，高さ
        ih, iw = image.shape[:2]

        r = []
        for ((px1, py1), (px2, py2)) in self._prev_faces:
            # 前回の中央
            pxc, pyc = (px1 + px2) / 2, (py1 + py2) / 2
            # 前回の幅と高さ
            pw, ph = px2 - px1, py2 - py1
            # 画像内での検索ウィンドウの幅，高さ
            dw = pw * self._detection_window_ratio
            dh = ph * self._detection_window_ratio
            # リサイズ後の検索ウィンドウの幅，比，高さ
            dwt = self._detection_window_width
            ratio = dwt / dw
            dht = dh * ratio
            # リサイズ後の方が大きい場合は変更無しとする
            if dwt > dw and dht > dh:
                dwt = dw
                dht = dh
                ratio = 1.0
            # 画像内の探索ウィンドウのX, Y座標
            dx1 = max(0, pxc - dw / 2)
            dy1 = max(0, pyc - dh / 2)
            dx2 = min(iw, pxc + dw / 2)
            dy2 = min(ih, pyc + dh / 2)
            # 本来のratioが違っている可能性があるのでここでもう一度補正
            dw = dx2 - dx1
            dh = dy2 - dy1
            # 小さい方に合わせて正方形にする
            if dw < dh:
                dh = dw
                dy1 = max(0, pyc - dh / 2)
                dy2 = min(ih, pyc + dh / 2)
            else:
                dw = dh
                dx1 = max(0, pxc - dw / 2)
                dx2 = min(iw, pxc + dw / 2)
            ratio = dwt / dw
            # 画像の切り出し
            ei = image[floor(dy1):floor(dy2), ceil(dx1):ceil(dx2)]
            print("({}, {}) - ({}, {}) -> ({}, {}) {}".format(
                ceil(dx1), floor(dy1), ceil(dx2), floor(dy2), dwt, dht, ratio))
            # リサイズ
            if dwt != dw or dht != dh:
                di = cv2.resize(ei, (ceil(dwt), ceil(dht)))
            else:
                di = ei
                ratio = 1.0
            # 顔を検出
            faces = self._detector.detect(di)
            # 位置，サイズを元に戻して結果リストに追加する
            for ((x1, y1), (x2, y2)) in faces:
                x1 = floor(x1 / ratio + dx1)
                y1 = floor(y1 / ratio + dy1)
                x2 = ceil(x2 / ratio + dx1)
                y2 = ceil(y2 / ratio + dy1)
                r.append(((x1, y1), (x2, y2)))
                # import ipdb; ipdb.set_trace()
        if len(self._prev_faces) == 0 or (len(r) == 0 and self._auto_retry):
            faces = self._detector.detect(image)
            r.extend(faces)
        self._prev_faces = copy.deepcopy(r)

        return r
