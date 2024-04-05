# utf-8
import numpy as np
import pandas as pd
from os import path
from . import LFW
from sflib.img.face.detection import FaceDetector
from sflib.img.face.shape_prediciton import FaceShapePredictor
from sflib.img.face.alignment import FaceAligner

from ...img.face.feature.ppes.extraction import PPES, PPESExtractor
from ...img.face.feature.ppes2.extraction import PPESv2, PPESv2Extractor


class AlignedFaces():
    """
    LFWの顔画像をアライメントして一つのDataFrameファイルにまとめたもの

    """
    DEFAULT_DF_PATH = path.join(path.dirname(LFW.DEFAULT_PATH),
                                'lfw_aligned.df.pkl')

    def __init__(self, filepath=None):
        """
        コンストラクタ
        
        Parameters
        ----------
        filepath : string
           読み込むデータフレームのパス．
           Noneの場合はDEFAULT_DF_PATHから読み込む．
        """
        if filepath is None:
            filepath = AlignedFaces.DEFAULT_DF_PATH
        self.dataframe = pd.read_pickle(filepath)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_item_by_index(key)
        elif isinstance(key, slice):
            r = [
                self.get_item_by_index(i)
                for i in range(*(key.indices(len(self))))
            ]
            return tuple(r)
        elif isinstance(key, str):
            return self.get_items_by_name(key)

    def get_item_by_index(self, i):
        r = self.dataframe.iloc[i, :]
        shape = np.array(r[2 + 4:2 + 4 + 68 * 2], 'int64')
        shape = shape.reshape((68, 2))
        img = np.array(r[2 + 4 + 68 * 2:], 'uint8')
        img = img.reshape((96, 96))
        return r['subject_name'], r['image_no'], shape, img

    @staticmethod
    def generate(filepath=None, refresh=False):
        """
        sflib.img.face.FaceAlignerの，align_with_eq_histでアライメントをとって
        データフレームにする

        Parameters
        ----------
        filepath : string
            データフレームを保存するファイルパス．
            Noneの場合はDEFAULT_DF_PATHに保存される．
        refresh : bool
            Trueの場合は既存のデータフレームを削除し，全ての画像について更新する．
            Falseの場合は，既に処理ずみの画像はスキップして未処理の画像のみ更新する．
        """
        if filepath is None:
            filepath = AlignedFaces.DEFAULT_DF_PATH
        # データフレームのカラム名，各カラムのデータタイプを生成
        column_names = [
            'subject_name', 'image_no', 'bb_x', 'bb_y', 'bb_w', 'bb_h'
        ]
        dtypes = ['object', 'int64', 'int32', 'int32', 'int32', 'int32']
        for i in range(68):
            column_names.extend([('x%02d' % i), ('y%02d' % i)])
            dtypes.extend(['int32', 'int32'])
        for y in range(96):
            for x in range(96):
                column_names.append('i%03d/%03d' % (x, y))
                dtypes.append('uint8')

        if path.exists(filepath) and refresh is False:
            df_out = pd.read_pickle(filepath)
        else:
            df_out = None

        face_detector = FaceDetector()
        face_shape_predictor = FaceShapePredictor()
        face_aligner = FaceAligner()

        lfw = LFW()
        for i in range(len(lfw)):
            subj, number, img = lfw[i]
            print("%s %s " % (subj, number), end='')
            # 既に対処済だったらパス
            if df_out is not None and \
               np.any((df_out['subject_name'] == subj) &
                      (df_out['image_no'] == int(number))):
                print("already processed")
                continue
            dets = face_detector.detect(img)
            if len(dets) == 0:
                print("face not found.")
                continue
            # 最初の結果（もっとも大きい四角）のみ相手にする
            det = dets[0]
            landmarks = face_shape_predictor.predict(img, det[0], det[1])
            aligned_img = face_aligner.align_with_eq_hist(img, landmarks)

            data = [subj, int(number)]
            data.extend([
                det[0][0], det[0][1], det[1][0] - det[0][0],
                det[1][1] - det[0][1]
            ])
            data.extend(np.array(landmarks, 'int32').ravel().tolist())
            data.extend(aligned_img.ravel().tolist())
            ldata = [[d] for d in data]
            df = pd.DataFrame(dict(zip(column_names, ldata)))
            df = df.astype(dict(zip(column_names, dtypes)))
            if df_out is None:
                df_out = df
            else:
                df_out = df_out.append(df)
            print("OK")
            # 念のため10回に1回は保存しておく
            if i % 10 == 0:
                df_out.to_pickle(filepath)
        df_out.to_pickle(filepath)


class PPESData:
    """
    LFWの顔画像からPPESを抽出したもの．

    """
    DEFAULT_DF_PATH = path.join(path.dirname(LFW.DEFAULT_PATH),
                                'lfw_ppes.df.pkl')

    def __init__(self, filepath=None):
        if filepath is None:
            filepath = PPESData.DEFAULT_DF_PATH
        self.dataframe = pd.read_pickle(filepath)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_item_by_index(key)
        elif isinstance(key, slice):
            r = [
                self.get_item_by_index(i)
                for i in range(*(key.indices(len(self))))
            ]
            return tuple(r)
        elif isinstance(key, str):
            return self.get_items_by_name(key)

    def get_item_by_index(self, i) -> PPES:
        r = self.dataframe.iloc[i, :]
        id = r['id']
        s = 2
        e = s + 64 * 64
        image = np.array(r[s:e], 'uint8')
        image = image.reshape((64, 64))
        s = e
        e = s + 6
        pose = np.array(r[s:e], 'float32')
        s = e
        e = s + 68 * 2
        expression = np.array(r[s:e], 'float32').reshape(68, -1)
        s = e
        e = s + 68 * 2
        landmarks = np.array(r[s:e], 'float32').reshape(68, -1)
        s = e
        e = s + 64 * 64
        mask = np.float32(np.array(r[s:e], 'uint8').reshape((64, 64)))
        return PPES(id, image, pose, expression, landmarks, mask)

    @staticmethod
    def generate(filepath=None, refresh=False):
        if filepath is None:
            filepath = PPESData.DEFAULT_DF_PATH
        # カラム名を生成
        # (1) IDと通し番号
        column_names = ['id', 'image_no']
        dtypes = ['object', 'int64']
        # (2) 画像（64x64）
        for y in range(64):
            for x in range(64):
                column_names.append('i%03d/%03d' % (x, y))
                dtypes.append('uint8')
        # (3) ポーズパラメタ（2x3)
        for i in range(6):
            column_names.append('p%d' % i)
            dtypes.append('float32')
        # (4) 表情パラメタ（68x2)
        for i in range(68):
            column_names.extend([('e%02dx' % i), ('e%02dy' % i)])
            dtypes.extend(['float32', 'float32'])
        # (5) ランドマーク
        for i in range(68):
            column_names.extend([('l%02dx' % i), ('l%02dy' % i)])
            dtypes.extend(['float32', 'float32'])
        # (6) マスク
        for y in range(64):
            for x in range(64):
                column_names.append('m%03d/%03d' % (x, y))
                dtypes.append('uint8')

        if path.exists(filepath) and refresh is False:
            df_out = pd.read_pickle(filepath)
        else:
            df_out = None

        extractor = PPESExtractor()

        lfw = LFW()
        for i in range(len(lfw)):
            subj, number, img = lfw[i]
            print("%s %s " % (subj, number), end='', flush=True)
            if df_out is not None and \
               np.any((df_out['id'] == subj) &
                      (df_out['image_no'] == int(number))):
                print("already processed")
                continue
            ppes_list = extractor.extract(img, id=subj)
            # 最初の結果をのみを相手にする
            if len(ppes_list) == 0:
                print("any face is not found")
                continue
            ppes = ppes_list[0]
            data = [subj, int(number)]
            data.extend(ppes.image.ravel().tolist())
            data.extend(ppes.pose.ravel().tolist())
            data.extend(ppes.expression.ravel().tolist())
            data.extend(ppes.landmarks.ravel().tolist())
            data.extend(np.uint8(ppes.mask).ravel().tolist())
            ldata = [[d] for d in data]
            # import ipdb; ipdb.set_trace()
            df = pd.DataFrame(dict(zip(column_names, ldata)))
            # import ipdb; ipdb.set_trace()
            # df = df.astype(dict(zip(column_names, dtypes)))
            if df_out is None:
                df_out = df
            else:
                df_out = df_out.append(df)
            print("OK")
            # 念のため100回に1回は保存しておく
            if i % 100 == 0:
                df_out = df_out.astype(dict(zip(column_names, dtypes)))
                df_out.to_pickle(filepath)
        df_out.to_pickle(filepath)


class PPESv2Data:
    """
    LFWの顔画像からPPESv2を抽出したもの．

    """
    DEFAULT_DF_PATH = path.join(path.dirname(LFW.DEFAULT_PATH),
                                'lfw_ppes_v2.df.pkl')

    def __init__(self, filepath=None):
        if filepath is None:
            filepath = PPESv2Data.DEFAULT_DF_PATH
        self.dataframe = pd.read_pickle(filepath)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_item_by_index(key)
        elif isinstance(key, slice):
            r = [
                self.get_item_by_index(i)
                for i in range(*(key.indices(len(self))))
            ]
            return tuple(r)
        elif isinstance(key, str):
            return self.get_items_by_name(key)

    def get_item_by_index(self, i) -> PPESv2:
        r = self.dataframe.iloc[i, :]
        id = r['id']
        s = 2
        e = s + 64 * 64
        image = np.array(r[s:e], 'uint8')
        image = image.reshape((64, 64))
        s = e
        e = s + 68 * 2
        landmarks = np.array(r[s:e], 'float32').reshape(-1, 2)
        return PPESv2(id, image, landmarks)

    @staticmethod
    def generate(filepath=None, refresh=False):
        if filepath is None:
            filepath = PPESv2Data.DEFAULT_DF_PATH
        # カラム名を生成
        # (1) IDと通し番号
        column_names = ['id', 'image_no']
        dtypes = ['object', 'int64']
        # (2) 画像（64x64）
        for y in range(64):
            for x in range(64):
                column_names.append('i%03d/%03d' % (x, y))
                dtypes.append('uint8')
        # (3) ランドマーク
        for i in range(68):
            column_names.extend([('l%02dx' % i), ('l%02dy' % i)])
            dtypes.extend(['float32', 'float32'])

        if path.exists(filepath) and refresh is False:
            df_out = pd.read_pickle(filepath)
        else:
            df_out = None

        extractor = PPESv2Extractor()

        lfw = LFW()
        for i in range(len(lfw)):
            subj, number, img = lfw[i]
            print("%s %s " % (subj, number), end='', flush=True)
            if df_out is not None and \
               np.any((df_out['id'] == subj) &
                      (df_out['image_no'] == int(number))):
                print("already processed")
                continue
            ppes_list = extractor.extract(img, id=subj)
            # 最初の結果をのみを相手にする
            if len(ppes_list) == 0:
                print("any face is not found")
                continue
            ppes = ppes_list[0]
            data = [subj, int(number)]
            data.extend(ppes.image_orig.ravel().tolist())
            data.extend(ppes.landmarks.ravel().tolist())
            ldata = [[d] for d in data]
            # import ipdb; ipdb.set_trace()
            df = pd.DataFrame(dict(zip(column_names, ldata)))
            # import ipdb; ipdb.set_trace()
            # df = df.astype(dict(zip(column_names, dtypes)))
            if df_out is None:
                df_out = df
            else:
                df_out = df_out.append(df)
            print("OK")
            # 念のため100回に1回は保存しておく
            if i % 100 == 0:
                df_out = df_out.astype(dict(zip(column_names, dtypes)))
                df_out.to_pickle(filepath)
        df_out.to_pickle(filepath)
