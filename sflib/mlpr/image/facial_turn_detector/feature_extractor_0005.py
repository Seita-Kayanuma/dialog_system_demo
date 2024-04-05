# 68点の変化量，重心の移動量，大きさの変化量に関わる3次元の特徴量のみ．
# なんかスケーリングが悪そうだったので，それぞれ100倍してます．
# 全特徴量に対してそうするならいらんやんというツッコミはなしで...
#  ・68点それぞれのdX, dY（68 x 2 = 136次元）
#  ・重心の差分 dX, dY （顔の幅で正規化）
#  ・顔の幅の変化率（変化がなければ0）
import cv2
from .base import FacialTurnDetectorFeatureExtractor
import numpy as np
from ....img.face.detection import FaceDetector
from ....img.face.shape_prediciton import FaceShapePredictor
import torch


class InputImageExtractor:
    def __init__(self, expand_ratio=2.0):
        """Args:
          expand_ratio (float): 前フレームの結果から何倍の範囲を探索するか
        """
        self._face_detector = FaceDetector()
        self._face_shape_predictor = FaceShapePredictor()
        self._expand_ratio = expand_ratio

        self._prev_landmarks = None
        self._prev_face = None
        self._prev_fx = None
        self._prev_w = None

    def reset(self):
        """
        """
        self._prev_face = None
        self._prev_landmarks = None
        self._prev_fx = None
        self._prev_w = None
        
    def extract(self, image: np.array):
        """
        """
        # カラー画像だったらグレイスケールに変換
        if image.ndim >= 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face = None
        if self._prev_face is not None:
            rect = self.expand_rect(self._prev_face, self._expand_ratio,
                                    image.shape)
            face = self.detect_face_in_rect(image, rect, self._face_detector)
        if face is None:
            faces = self._face_detector.detect(image)
            if len(faces) > 0:
                face = faces[0]
        if face is None:
            self._prev_face = None
            return None, np.array(([0.0] * 68 * 2) + [0.0, 0.0, 1.0], np.float32)
        self._prev_face = face
        landmarks = self._face_shape_predictor.predict(image, face[0], face[1])
        landmarks = np.array(landmarks, np.float32)
        
        w = (landmarks.max(axis=0) - landmarks.min(axis=0))[0]
        fx = np.mean(landmarks, axis=0)

        if self._prev_landmarks is not None:
            dlandmarks = ((landmarks - fx) - (self._prev_landmarks - self._prev_fx)) / w * 100.0
        else:
            dlandmarks = np.zeros((68, 2))
            
        if self._prev_fx is not None:
            dxy = ((fx - self._prev_fx) / w) * 100.0
            dw = np.log10(w / self._prev_w) * 100.0
        else:
            dxy = np.zeros((2,), np.float32)
            dw = 0.0
        self._prev_landmarks = landmarks
        self._prev_fx = fx
        self._prev_w = w
        
        img_out = None
        
        # import ipdb; ipdb.set_trace()
        #  print(dlandmarks)
        return img_out, np.float32(np.concatenate([dlandmarks.ravel(), dxy, [dw]]))

    @staticmethod
    def expand_rect(rect, ratio=1.5, img_size=None):
        """前のフレームの周辺に顔があることを前提に，
        前のフレームの顔検出結果を拡大して検出対象とするため．
        """
        ((ox1, oy1), (ox2, oy2)) = rect
        orig_width = ox2 - ox1
        orig_height = oy2 - oy1
        new_width = ratio * orig_width
        new_height = ratio * orig_height
        x1 = ox1 - (new_width - orig_width) / 2
        x2 = ox2 + (new_width - orig_width) / 2
        y1 = oy1 - (new_height - orig_height) / 2
        y2 = oy2 + (new_height - orig_height) / 2

        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        if img_size is not None:
            x2 = int(min(img_size[1], x2))
            y2 = int(min(img_size[0], y2))
        else:
            x2 = int(x2)
            y2 = int(y2)
        return ((x1, y1), (x2, y2))

    @staticmethod
    def detect_face_in_rect(image, rect, detector):
        """指定された枠内で顔を見つける．
        位置の情報は元の座標に戻される．
        """
        ((x1, y1), (x2, y2)) = rect
        subimage = image[y1:y2, x1:x2]
        faces = detector.detect(subimage)
        if len(faces) < 1:
            return None
        ((rx1, ry1), (rx2, ry2)) = faces[0]
        rx1 += x1
        rx2 += x1
        ry1 += y1
        ry2 += y1
        return ((rx1, ry1), (rx2, ry2))

    @staticmethod
    def equalize_hist(image, mask):
        img_out = image
        if img_out.ndim > 2:
            # assuming image is color in BGR
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        img_out_masked = cv2.bitwise_and(img_out, mask)
        hist, bins = np.histogram(img_out_masked, 256, [0, 255])
        hist[0] = 0
        cdf = hist.cumsum()
        tbl = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        tbl = tbl.astype('uint8')
        return tbl[img_out]


class FacialTurnDetectorFeatureExtractor0005(FacialTurnDetectorFeatureExtractor
                                             ):
    def __init__(self, device='cpu', auto_reset=True, dx_only=False):
        self._input_image_extractor = InputImageExtractor()
        self._auto_reset = auto_reset
        self._device = device
        self._dx_only = dx_only

    @property
    def device(self):
        return self._device

    def get_filename_base(self):
        """モデルファイルの名前のヒントに使う文字列を取得する
        """
        s = super().get_filename_base()
        if self._dx_only:
            s += 'D'
        return s

    @property
    def feature_dim(self):
        """特徴ベクトルの次元数
        """
        if self._dx_only:
            return 3
        return 68 * 2 + 3

    def reset(self):
        """状態をリセットする．
        新しいバッチを入力する際などに呼ぶ必要がある．
        """
        self._input_image_extractor.reset()

    def detach(self):
        """コンテクストをデタッチする．
        RNN系のニューラルネットを内部に持つ場合に，
        バックワード計算の対象から外すために呼ぶ必要がある．
        """
        pass

    def calc(self, img_seq_list: list) -> list:
        """画像系列データから特徴量を計算する．

        Args:
          img_seq_list: 画像シーケンスのリスト
             img_seq_list[i] は i 番目のバッチの画像のリスト
        
        Returns:
          torch.Tensor: 特徴ベクトル列のリスト
            リストのサイズは len(img_seq_list) と同じ．
            リストの中身は，形状が(length, feature_dim) である特徴ベクトルのテンソル．
            lengthは各画像列に応じて異なる．
        """
        zero_vec = torch.zeros((
            1, self.feature_dim,
        ), dtype=torch.float32)
        if self.device is not None:
            zero_vec = zero_vec.to(self.device)
        feat_list = []
        for img_seq in img_seq_list:
            if self._auto_reset:
                self._input_image_extractor.reset()
            vec_list = []
            for image in img_seq:
                in_image, vec = self._input_image_extractor.extract(image)
                if self._dx_only:
                    vec = vec[-3:]
                vec_list.append(vec)
            vec_tensor = torch.tensor(vec_list)
            if self.device is not None:
                vec_tensor = vec_tensor.to(self.device)
            feat_list.append(vec_tensor)
        return feat_list
