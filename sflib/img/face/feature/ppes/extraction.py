import numpy as np
from ...detection import FaceDetector
from ...shape_prediciton import FaceShapePredictor
from ...alignment import FaceAligner
import cv2


class PPES:
    """PPESは，
    Personality, Pose, Expression Separation の略．
    要は個人性と姿勢と表情を分離した特徴量を求めるための情報．
    このクラスは，学習データに使う一つのサンプルが持つ情報を
    保持するためのクラス"""

    def __init__(self,
                 id: str,
                 image: np.array,
                 pose: np.array,
                 expression: np.array,
                 landmarks: np.array = None,
                 mask: np.array = None):
        """
        Args:
          id : 個人性を表すID．同一人物に与えられるユニークなID．
            データのレベルでは番号で扱うと煩雑になってしまうので
            文字列とする．
          image : 画像情報．顔を切り出しただけのもの．
            shape は (64, 64)
            dtype は np.uint8
            必要な正規化（ヒストグラム正規化）は既に行われているものとする．
          pose : ランドマークのアフィン変換のパラメータ．
            [a11, a12, t1, a21, a22, t2] の6次元のベクトル．
            dtype は np.float32
            なお，アフィン変換の入力空間は，imageを切り出した窓を0.0〜1.0に
            正規化した空間．出力空間はテンプレート空間を0.0〜1.0に正規化した空間．
          expression : ランドマークをposeのアフィン変換でテンプレート空間に
            移した後のテンプレートとの差分．
            shape は (68, 2), dtype は np.float32
          landmarks : テンプレート空間への変換前のランドマーク情報．
            データとしては使わないが，表示用に使う． 
          mask : 顔領域が1になっているマスク画像．ロスを計算する際に利用
        """
        self.__id = id
        self.__image = image
        self.__pose = pose
        self.__expression = expression
        self.__landmarks = landmarks
        self.__mask = mask

    @property
    def id(self):
        return self.__id

    @property
    def image(self):
        return self.__image

    @property
    def pose(self):
        return self.__pose

    @property
    def expression(self):
        return self.__expression

    @property
    def landmarks(self):
        return self.__landmarks

    @property
    def mask(self):
        return self.__mask

    def refresh_noise(self):
        img_out = self.__image.copy()
        height, width = img_out.shape
        
        # landmarks_i = self.__landmarks.copy()
        # pts = landmarks_i[PPESExtractor.mask_pts, :]
        # pts[:, 0] = pts[:, 0] * width
        # pts[:, 1] = pts[:, 1] * height
        
        # mask = np.zeros(img_out.shape, np.uint8)
        # mask = cv2.fillConvexPoly(mask, np.int32(pts), color=(255, ))

        mask_ = np.uint8(self.mask * 255)
        img_out = cv2.bitwise_and(img_out, mask_)
        
        rand_mask = np.uint8(np.random.rand(height, width) * 256)
        rand_mask = cv2.bitwise_and(cv2.bitwise_not(mask_), rand_mask)
        img_out = img_out + rand_mask

        self.__image = img_out
        
        pass


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


class PPESExtractor:
    # ランドマークをアフィン変換する際の基準となる点のインデクス
    # affine_pts_indices = (36, 45, 33)
    # affine_pts_indices = list(range(68))
    # affine_pts_indices = (36, 39, 40, 41) + (42, 45, 46, 47) + (31, 32, 33, 34, 35)
    affine_pts_indices = (36, 39, 40, 41) + (42, 45, 46, 47) + (48, 49, 50, 51, 52, 53, 54)
    # ヒストグラム平滑化をする際のマスクを生成するための
    # 輪郭点のインデクス
    mask_pts = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                26, 25, 24, 23, 22, 21, 20, 19, 18, 17)

    def __init__(self):
        self.__face_detector = FaceDetector()
        self.__face_shape_predictor = FaceShapePredictor()
        self.__face_aligner = FaceAligner()
        self.__template = self.__face_aligner.template.copy()
        self.__margin = 0.2

    def extract(self, image, id=None, verbose=False) -> PPES:
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
            width = x2 - x1
            height = y2 - y1
            x1i = int(landmarks[:, 0].min())
            y1i = int(landmarks[:, 1].min())
            x2i = int(landmarks[:, 0].max())
            y2i = int(landmarks[:, 1].max())
            # x1i = int(x1 - width * self.__margin)
            # x2i = int(x2 + width * self.__margin)
            # y1i = int(y1 - height * self.__margin)
            # y2i = int(y2 + height * self.__margin)
            x1i = max(x1i, 0)
            x1i = min(x1i, image.shape[1])
            x2i = max(x2i, 0)
            x2i = min(x2i, image.shape[1])
            y1i = max(y1i, 0)
            y1i = min(y1i, image.shape[0])
            y2i = max(y2i, 0)
            y2i = min(y2i, image.shape[0])
            # widthi = x2i - x1i
            # heighti = y2i - y1i

            # 画像の切り出し，リサイズ，ヒストグラム正規化
            # img_out = image[x1:x2, y1:y2]
            img_out = image[y1i:y2i, x1i:x2i]
            if img_out.ndim > 2:
                # assuming image is color in BGR
                img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
            img_out = cv2.resize(
                img_out, (out_width, out_height),
                interpolation=cv2.INTER_LINEAR)

            # ランドマークからテンプレート空間へのアフィン変換を求める
            # まずランドマーク自体を単純に正規化（0.0〜1.0）にする．
            x1 = int(landmarks[:, 0].min())
            y1 = int(landmarks[:, 1].min())
            x2 = int(landmarks[:, 0].max())
            y2 = int(landmarks[:, 1].max())
            width = x2 - x1
            height = y2 - y1
            landmarks_ = landmarks.copy()
            landmarks_[:, 0] = (landmarks[:, 0] - x1) / width
            landmarks_[:, 1] = (landmarks[:, 1] - y1) / height
            # ランドマークの指定の点をテンプレート空間の位置に移動させる
            # アフィン変換行列を求める
            # affineTransorm = cv2.getAffineTransform(
            #    landmarks_[self.__class__.affine_pts_indices, :],
            #    self.__template[self.__class__.affine_pts_indices, :])
            pts = self.__class__.affine_pts_indices
            src = np.concatenate([landmarks_[pts, :], np.ones((len(pts), 1))], axis=1)
            dst = self.__template[pts, :]
            affineTransorm, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
            affineTransorm = affineTransorm.T            
            # ランドマークにアフィン変換を適用する
            landmarks_t = np.concatenate(
                [landmarks_, np.ones((68, 1))], axis=1)
            landmarks_t = np.matmul(landmarks_t, affineTransorm.T)
            # 差分を求める
            expression = landmarks_t - self.__template

            # 画像のヒストグラム平滑化．
            # 輪郭内のピクセルだけで平滑化する．
            landmarks_i = landmarks.copy()
            landmarks_i[:, 0] = (landmarks[:, 0] - x1i) / (x2i - x1i)
            landmarks_i[:, 1] = (landmarks[:, 1] - y1i) / (y2i - y1i)
            mask = np.zeros(img_out.shape, np.uint8)
            pts = landmarks_i[self.__class__.mask_pts, :]
            pts[:, 0] = pts[:, 0] * out_width
            pts[:, 1] = pts[:, 1] * out_height
            mask = cv2.fillConvexPoly(mask, np.int32(pts), color=(255, ))
            img_out = equalize_hist(img_out, mask)

            # rand_mask = np.uint8(np.random.rand(out_height, out_width) * 256)
            # rand_mask = cv2.bitwise_and(cv2.bitwise_not(mask), rand_mask)
            # img_out = img_out + rand_mask

            mask_ = np.float32(mask) / 255.0
            
            result_list.append(
                PPES(id, img_out, affineTransorm.ravel(), expression,
                     landmarks_i, mask_))

        # import ipdb; ipdb.set_trace()
        return result_list
