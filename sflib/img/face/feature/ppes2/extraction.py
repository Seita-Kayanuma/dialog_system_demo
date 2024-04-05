import numpy as np
from ...detection import FaceDetector
from ...shape_prediciton import FaceShapePredictor
from .alignment import FaceAligner
import cv2


def _apply_affine_transform(points, t, inv=False):
    if inv:
        t = np.concatenate([t, np.array([0, 0, 1]).reshape(1, 3)], axis=0)
        t = np.linalg.inv(t)
        t = t[:2, :]
    x = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return np.matmul(x, t.T)


class PPESv2:
    """PPESは，
    Personality, Pose, Expression Separation の略．
    要は個人性と姿勢と表情を分離した特徴量を求めるための情報．
    このクラスは，学習データに使う一つのサンプルが持つ情報を
    保持するためのクラス"""

    # 共通のアライナ
    aligner = FaceAligner()

    # テンプレートのランドマーク
    template = aligner.template

    # アライメント画像のマスク
    aligned_mask = np.float32(aligner.mask > 0)

    # グローバルアフィン変換を決めるためのポイントのインデクス
    # 目の下の線と，唇の上の線
    affine_pts_indices = (36, 39, 40, 41) + (42, 45, 46, 47) \
        + (48, 49, 50, 51, 52, 53, 54)
    # 顎のラインと鼻のラインのポイントのインデクス
    pose2_pts = \
        (0, 1, 2, 3, 4 , 5, 6,  7, 8, 9, 10,  11, 12, 13,  14, 15, 16) + \
        (27, 28, 29, 30, 31, 32, 33, 34, 35)
    # 眉，目，口のポイントのインデクス
    expr_pts = \
        (17, 18, 19, 20, 21) + (22, 23, 24, 25, 26) + \
        (36, 37, 38, 39, 40, 41) + (42, 43, 44, 45, 46,  47) + \
        (48, 49, 50, 51, 52, 53, 54, 55, 56, 57,  58, 59, 60, 61, 62,  63, 64,  65, 66, 67)
    # マスクを計算するのポイントのインデクス
    mask_pts = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26,
                25, 24, 23, 22, 21, 20, 19, 18, 17)

    def __init__(self, id: str, image: np.array, landmarks: np.array):
        """
        Args:
          id : 個人性を表すID．同一人物に与えられるユニークなID．
            データのレベルでは番号で扱うと煩雑になってしまうので
            文字列とする．
          image : 画像情報．顔を切り出しただけのもの．
            shape は (64, 64)
            dtype は np.uint8
            必要な正規化（ヒストグラム正規化）は既に行われているものとする．
          landmarks : テンプレート空間への変換前のランドマーク情報．
            データとしては使わないが，表示用に使う． 
        """
        self.__id = id
        self.__image_orig = image
        self.__landmarks = landmarks
        self.__pose1 = None
        self.__pose2 = None
        self.__expression = None
        self.__alignment = None
        self.__aligned_face = None
        self.__mask = None
        self.__image_hist_eq = None
        self.__image_noised = None

    @property
    def id(self):
        return self.__id

    @property
    def image(self):
        if self.__image_noised is None:
            self.refresh_noise()
        return self.__image_noised

    @property
    def image_orig(self):
        return self.__image_orig

    @property
    def image_hist_eq(self):
        if self.__image_hist_eq is None:
            self.__image_hist_eq = equalize_hist(self.__image_orig,
                                                 np.uint8(self.mask * 255.0))
        return self.__image_hist_eq

    @property
    def landmarks(self):
        return self.__landmarks

    @property
    def pose1(self):
        """ランドマークからテンプレート空間へのアフィン変換のパラメタ．
        実際には pose1.reshape(2,3) したものがアフィン変換行列となる．
        """
        if self.__pose1 is None:
            pts = self.__class__.affine_pts_indices
            src = np.concatenate(
                [self.landmarks[pts, :],
                 np.ones((len(pts), 1))], axis=1)
            dst = self.template[pts, :]
            affineTransorm, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
            affineTransorm = affineTransorm.T
            self.__pose1 = affineTransorm.ravel()
        return self.__pose1

    @property
    def pose2(self):
        """鼻，アゴのラインの点の変位
        実際には pose2.reshape(-1, 2) したものが変位ベクトルを並べた行列になる．
        """
        if self.__pose2 is None:
            affineTransform = self.pose1.reshape(2, 3)
            pts = self.__class__.pose2_pts
            self.__pose2 = \
                _apply_affine_transform(self.landmarks[pts, :], affineTransform) \
                - self.template[pts, :]
            self.__pose2 = self.__pose2.ravel()
        return self.__pose2

    @property
    def expression(self):
        """眉，目，口の変位．
        実際には expression.reshape(-1, 2) したものが変位ベクトルを並べた行列になる．
        """
        if self.__expression is None:
            affineTransform = self.pose1.reshape(2, 3)
            pts = self.__class__.expr_pts
            self.__expression = \
                _apply_affine_transform(self.landmarks[pts, :], affineTransform) \
                - self.template[pts, :]
            self.__expression = self.__expression.ravel()
        return self.__expression

    @property
    def alignment(self):
        """アライメント情報を取得する．
        (64, 64, 2) で，[:, :, 0] にXの変位，[:, :, 1]にYの変位が入っている
        """
        if self.__alignment is None:
            self.__alignment = self.__class__.aligner.get_alignment_info(
                self.landmarks)
        return self.__alignment

    @property
    def aligned_image(self):
        """アライメントされた画像を取得する．
        """
        if self.__aligned_face is None:
            self.__aligned_face = \
                self.__class__.aligner.align(self.image_hist_eq,
                                             np.int32(self.landmarks * np.array(self.image.shape)))
        return self.__aligned_face

    @property
    def mask(self):
        """画像のマスク"""
        if self.__mask is None:
            mask = np.zeros(self.__image_orig.shape, np.uint8)
            pts = self.landmarks[self.__class__.mask_pts, :] * np.array(
                self.__image_orig.shape)
            mask = np.float32(
                cv2.fillConvexPoly(mask, np.int32(pts), color=(1, )))
            self.__mask = mask
        return self.__mask

    def refresh_noise(self):
        img_out = self.image_hist_eq.copy()
        height, width = img_out.shape

        mask_ = np.uint8(self.mask * 255)
        img_out = cv2.bitwise_and(img_out, mask_)

        rand_mask = np.uint8(np.random.rand(height, width) * 256)
        rand_mask = cv2.bitwise_and(cv2.bitwise_not(mask_), rand_mask)
        img_out = img_out + rand_mask

        self.__image_noised = img_out

    def get_info_for_draw(self):
        pts = self.__class__.pose2_pts
        pose2_pts = self.pose2.reshape(-1, 2) + self.template[pts, :]
        pose2_pts = _apply_affine_transform(pose2_pts,
                                            self.pose1.reshape(2, 3),
                                            inv=True)
        pts = self.__class__.expr_pts
        expr_pts = self.expression.reshape(-1, 2) + self.template[pts, :]
        return self.template, pose2_pts, expr_pts


def equalize_hist(image, mask, out_masked=True):
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
    if out_masked:
        return tbl[img_out_masked]
    else:
        return tbl[img_out]


def extract_image_with_landmarks(image: np.array, landmarks: np.array,
                                 out_size = (64, 64)):
    x1i = int(landmarks[:, 0].min())
    y1i = int(landmarks[:, 1].min())
    x2i = int(landmarks[:, 0].max())
    y2i = int(landmarks[:, 1].max())
    x1i = max(x1i, 0)
    x1i = min(x1i, image.shape[1])
    x2i = max(x2i, 0)
    x2i = min(x2i, image.shape[1])
    y1i = max(y1i, 0)
    y1i = min(y1i, image.shape[0])
    y2i = max(y2i, 0)
    y2i = min(y2i, image.shape[0])

    # 画像の切り出し，リサイズ，ヒストグラム正規化
    img_out = image[y1i:y2i, x1i:x2i]
    if img_out.ndim > 2:
        # assuming image is color in BGR
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    img_out = cv2.resize(img_out, out_size,
                         interpolation=cv2.INTER_LINEAR)
    
    landmarks_out = np.float32(landmarks.copy())
    landmarks_out[:, 0] = (landmarks[:, 0] - x1i) / (x2i - x1i)
    landmarks_out[:, 1] = (landmarks[:, 1] - y1i) / (y2i - y1i)
    pts = landmarks_out[PPESv2.mask_pts, :] * np.array(img_out.shape)
    mask = np.zeros(img_out.shape, np.uint8)
    mask = cv2.fillConvexPoly(mask, np.int32(pts), color=(255, ))
    img_out = equalize_hist(img_out, mask, out_masked=False)
    # import ipdb; ipdb.set_trace()
    
    return img_out


class PPESv2Extractor:
    def __init__(self):
        self.__face_detector = FaceDetector()
        self.__face_shape_predictor = FaceShapePredictor()

    def extract(self, image, id=None, verbose=False) -> list:
        # 出力画像のサイズ（将来パラメタ化するかもしれんので変数にしておく）
        out_width = 64
        out_height = 64

        faces = self.__face_detector.detect(image)
        if verbose:
            print("found {} faces.".format(len(faces)))

        result_list = []
        for face in faces:
            x1, y1 = face[0]
            x2, y2 = face[1]

            landmarks = self.__face_shape_predictor.predict(
                image, (x1, y1), (x2, y2))
            landmarks = np.array(landmarks, np.float32)

            x1i = int(landmarks[:, 0].min())
            y1i = int(landmarks[:, 1].min())
            x2i = int(landmarks[:, 0].max())
            y2i = int(landmarks[:, 1].max())
            x1i = max(x1i, 0)
            x1i = min(x1i, image.shape[1])
            x2i = max(x2i, 0)
            x2i = min(x2i, image.shape[1])
            y1i = max(y1i, 0)
            y1i = min(y1i, image.shape[0])
            y2i = max(y2i, 0)
            y2i = min(y2i, image.shape[0])

            # 画像の切り出し，リサイズ，ヒストグラム正規化
            img_out = image[y1i:y2i, x1i:x2i]
            if img_out.ndim > 2:
                # assuming image is color in BGR
                img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
            img_out = cv2.resize(img_out, (out_width, out_height),
                                 interpolation=cv2.INTER_LINEAR)

            landmarks_out = landmarks.copy()
            landmarks_out[:, 0] = (landmarks[:, 0] - x1i) / (x2i - x1i)
            landmarks_out[:, 1] = (landmarks[:, 1] - y1i) / (y2i - y1i)

            result_list.append(PPESv2(id, img_out, landmarks_out))

            # import ipdb; ipdb.set_trace()
        return result_list
