# coding: utf-8
from os import path
import cv2
import pkgutil
import matplotlib.pyplot as plt

from .detection import FaceDetector
from .shape_prediciton import FaceShapePredictor
from .alignment import FaceAligner


def download_file(url, path):
    import urllib
    urllib.request.urlretrieve(url, path)

    
def demo():
    # 各種変換器などの作成
    face_detector = FaceDetector()
    face_shape_predictor = FaceShapePredictor()
    face_aligner = FaceAligner()

    # 入力画像の読み込み
    lenna_url = 'http://optipng.sourceforge.net/pngtech/img/lena.png'
    lenna_path = path.join(
        path.dirname(pkgutil.get_loader(__package__).get_filename()), 'data/lenna.png')

    if not path.exists(lenna_path):
        download_file(lenna_url, lenna_path)

    img = cv2.imread(lenna_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 顔の検出
    results = face_detector.detect(img)
    if len(results) == 0:
        return

    # 検出結果の表示用画像の生成
    r = results[0]
    img_detection = img.copy()
    cv2.rectangle(img_detection, r[0], r[1], (255, 0, 0), 2)

    # ランドマークの検出（この関数は失敗しない）
    s = face_shape_predictor.predict(img, r[0], r[1])

    # ランドマークの検出結果の表示用画像の生成
    img_shape_prediction = img_detection.copy()
    for x, y in s:
        cv2.circle(img_shape_prediction, (x, y), 3, (0, 255, 0))

    # アラインメント
    img_alignment = face_aligner.align(img, s)
    
    # 表示
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('input image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 2)
    plt.imshow(img_detection)
    plt.title('face detection')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 3)
    plt.imshow(img_shape_prediction)
    plt.title('shape prediction')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 4)
    plt.imshow(img_alignment)
    plt.title('face alignment')
    plt.xticks([])
    plt.yticks([])

    plt.show()
