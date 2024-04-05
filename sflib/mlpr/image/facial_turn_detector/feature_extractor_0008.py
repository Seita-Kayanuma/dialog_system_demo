# 画像特徴量としてAutoEncoderの中間層を入力するタイプ
#   - PPESv2の個人性以外のベクトル48次元
#   - 顔全体の dX, dY, dW (dWは比率なので対数をとる） これらは値が小さいので 100 倍
#   - 68点それぞれのdX, dY（68 x 2 = 136次元）
import cv2
from .base import FacialTurnDetectorFeatureExtractor
import numpy as np
from ....img.face.detection import FaceDetector
from ....img.face.shape_prediciton import FaceShapePredictor
from ....img.face.alignment import FaceAligner
from ....img.face.feature.autoencoder_pytorch.base \
    import load_face_auto_encoder_weights
import torch


# 少しダサいですが，使うオートエンコーダを変更する際は
# 下記を書き直してください
def load_face_auto_encoder(map_location=None):
    from ....img.face.feature.autoencoder_pytorch.autoencoder_0001 \
        import FaceAutoEncoder0001PyTorch as Encoder
    from ....img.face.feature.autoencoder_pytorch.trainer_lfw_0001 \
        import FaceAutoEncoderTrainerLFW0001 as Trainer
    encoder = Encoder()
    if map_location is not None:
        encoder = encoder.to(map_location)
    trainer = Trainer()
    load_face_auto_encoder_weights(encoder, trainer, map_location=map_location)
    return encoder, trainer


class InputImageExtractor:
    def __init__(self, expand_ratio=2.0):
        """Args:
          expand_ratio (float): 前フレームの結果から何倍の範囲を探索するか
        """
        self._face_detector = FaceDetector()
        self._face_shape_predictor = FaceShapePredictor()
        self._face_aligner = FaceAligner()
        self._expand_ratio = expand_ratio

        self._prev_face = None
        self._prev_landmarks = None
        self._prev_fx = None
        self._prev_w = None

    def reset(self):
        """
        """
        # print("********* RESET ***********")
        self._prev_face = None
        self._prev_landmarks = None
        self._prev_fx = None
        self._prev_w = None

    def extract(self, image: np.array):
        """
        """
        # カラー画像だったらグレイスケールに変換
        # if image.ndim >= 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
            return None, np.array([0.0] * (68 * 2), np.float32), np.array([0.0] * 3, np.float32)
        self._prev_face = face
        landmarks = self._face_shape_predictor.predict(image, face[0], face[1])
        aligned_img = self._face_aligner.align_with_eq_hist(image, landmarks)

        landmarks = np.array(landmarks, np.float32)
        w = (landmarks.max(axis=0) - landmarks.min(axis=0))[0]
        fx = np.mean(landmarks, axis=0)
        # import ipdb; ipdb.set_trace()

        if self._prev_landmarks is not None:
            dlandmarks = \
                ((landmarks - fx) - (self._prev_landmarks - self._prev_fx)) \
                / w * 100.0
        else:
            dlandmarks = np.zeros((68, 2), dtype=np.float32)

        if self._prev_fx is not None:
            dxy = ((fx - self._prev_fx) / w) * 100.0
            dw = np.log10(w / self._prev_w) * 100.0
        else:
            dxy = np.zeros((2, ), np.float32)
            dw = 0.0
        self._prev_landmarks = landmarks
        self._prev_fx = fx
        self._prev_w = w

        return aligned_img, np.float32(dlandmarks.ravel()), np.float32(np.concatenate([dxy, [dw]]))
    
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


class FacialTurnDetectorFeatureExtractor0008(FacialTurnDetectorFeatureExtractor
                                             ):
    def __init__(self,
                 use_landmarks=False,
                 use_dxdydw=False,
                 device='cpu',
                 auto_reset=True):

        encoder, trainer = load_face_auto_encoder(map_location=device)
        self._encoder_filename_base = \
            encoder.get_filename_base() + '+' + \
            trainer.get_filename_base()

        self._encoder = encoder
        self._encoder.share_memory()

        self._input_image_extractor = InputImageExtractor()
        self._auto_reset = auto_reset

        self._device = device

        self._use_landmarks = use_landmarks
        self._use_dxdydw = use_dxdydw

    @property
    def device(self):
        return self._device

    def get_filename_base(self):
        """モデルファイルの名前のヒントに使う文字列を取得する
        """
        s = super().get_filename_base()
        if self._use_landmarks:
            s += 'L'
        if self._use_dxdydw:
            s += 'D'
        s += '+' + self._encoder_filename_base

        return s

    @property
    def feature_dim(self):
        """特徴ベクトルの次元数
        """
        r = np.sum(self._encoder.bottleneck_dim)
        if self._use_landmarks:
            r += 68 * 2
        if self._use_dxdydw:
            r += 3
        return r

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
        zero_vec = torch.zeros((1, self._encoder.bottleneck_dim),
                               dtype=torch.float32)
        if self.device is not None:
            zero_vec = zero_vec.to(self.device)
        feat_list = []
        for img_seq in img_seq_list:
            feat_seq = []
            prev_feat = None
            if self._auto_reset:
                self._input_image_extractor.reset()
            in_image_list = []
            dlandmarks_list = []
            dxdydw_list = []
            # --- FOR TEST ---
            # img_seq = img_seq[:100]
            # ----
            for image in img_seq:
                in_image, dlandmarks, dxdydw = \
                    self._input_image_extractor.extract(image)
                dlandmarks_list.append(dlandmarks)
                dxdydw_list.append(dxdydw)
                if in_image is None:
                    if len(in_image_list) > 0:
                        # in_image_tensor = torch.tensor(
                        #     np.concatenate(in_image_list, axis=0))
                        # if self.device is not None:
                        #     in_image_tensor = in_image_tensor.to(self.device)
                        # TODO: 255で割る処理はエンコード時に行われるので，
                        #       ここでのin_image_tesnsorは0〜255の値になっている
                        #       必要がある．そのチェック．
                        # import ipdb; ipdb.set_trace()
                        # feat = self._encoder.encode(in_image_tensor)
                        in_image_array = np.concatenate(in_image_list, axis=0)
                        feat = self._encoder.encode(in_image_array)
                        feat = torch.tensor(feat)
                        if self.device is not None:
                            feat = feat.to(self.device)
                        # feat = feat.clone().detach()
                        prev_feat = feat[-1:, :]
                        feat_seq.append(feat)
                        in_image_list = []
                    if prev_feat is None:
                        feat = zero_vec
                    else:
                        feat = prev_feat
                    feat_seq.append(feat)
                else:
                    in_image_list.append(in_image)
                    # import ipdb; ipdb.set_trace()
            if len(in_image_list) > 0:
                # in_image_tensor = torch.tensor(
                #     np.concatenate(in_image_list, axis=0))
                # if self.device is not None:
                #     in_image_tensor = in_image_tensor.to(self.device)
                # feat = self._encoder.encode(in_image_tensor)
                in_image_array = np.concatenate(in_image_list, axis=0)
                feat = self._encoder.encode(in_image_array)
                feat = torch.tensor(feat)
                if self.device is not None:
                    feat = feat.to(self.device)
                # feat = feat.clone().detach()
                feat_seq.append(feat)
                # import ipdb; ipdb.set_trace()
            # import ipdb; ipdb.set_trace()
            feat_tensor = torch.cat(feat_seq, dim=0)
            if self._use_landmarks:
                dlandmarks_tensor = torch.tensor(dlandmarks_list)
                if self.device is not None:
                    dlandmarks_tensor = dlandmarks_tensor.to(self.device)
                feat_tensor = \
                    torch.cat([feat_tensor, dlandmarks_tensor], dim=1)
            if self._use_dxdydw:
                dxdydw_tensor = torch.tensor(dxdydw_list)
                if self.device is not None:
                    dxdydw_tensor = dxdydw_tensor.to(self.device)
                feat_tensor = \
                    torch.cat([feat_tensor, dxdydw_tensor], dim=1)
            feat_list.append(feat_tensor)
        return feat_list
