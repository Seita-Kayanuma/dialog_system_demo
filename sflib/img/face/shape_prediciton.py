# coding: utf-8
from os import path
import os
import pkgutil
import dlib
from ... import config

face_shape_predictor_model_path = path.join(
    config.get_package_data_dir(__package__), 'data',
    'shape_predictor_68_face_landmarks.dat')


def download_model_file():
    import urllib
    import bz2
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    bz2path = face_shape_predictor_model_path + '.bz2'
    if not path.exists(path.dirname(face_shape_predictor_model_path)):
        os.makedirs(
            path.dirname(face_shape_predictor_model_path),
            mode=0o755,
            exist_ok=True)
    urllib.request.urlretrieve(url, bz2path)
    with open(bz2path, 'br') as fi:
        with open(face_shape_predictor_model_path, 'bw') as fo:
            fo.write(bz2.decompress(fi.read()))
    os.unlink(bz2path)


class FaceShapePredictor:
    def __init__(self):
        if not path.exists(face_shape_predictor_model_path):
            print('model file is not found')
            print('download model file (this takes a few moment)')
            download_model_file()

        self._predictor = dlib.shape_predictor(face_shape_predictor_model_path)

    def predict(self, img, pt1, pt2):
        bb = dlib.rectangle(pt1[0], pt1[1], pt2[0], pt2[1])
        r = self._predictor(img, bb)
        result = [(p.x, p.y) for p in r.parts()]
        return result
