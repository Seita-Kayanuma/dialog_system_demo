# utf-8
import numpy as np
import pandas as pd
from os import path
from . import CKplus
from sflib.img.face.detection import FaceDetector
from sflib.img.face.shape_prediciton import FaceShapePredictor
from sflib.img.face.alignment import FaceAligner


class AlignedFaces():
    """
    CK+の顔画像をアライメントして一つのDataFrameファイルにまとめたもの

    """
    DEFAULT_DF_PATH = path.join(
        path.dirname(CKplus.DEFAULT_PATH), 'ckp_aligned.df.pkl')

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
        """
        i番目のデータを取得する．

        Parameters
        ----------
        i : int
            取得したいデータのインデクス

        Returns
        -------
        subject : string
                  被験者ID
        seq : int
              被験者内のseq番号
        emotion : float or None
                  ラベルづけされた表情番号．ラベルがつけられていないものは None
        strength : float
                   表情の強さ（シーケンス内での相対位置）
        image_no : int
                   画像番号
        shape : np.array float (68, 2)
                形状
        image : np.array uint8 (96, 96)
                アライメントされた画像
        """
        r = self.dataframe.iloc[i, :]
        shape = np.array(r[(5 + 4):(5 + 4 + 68 * 2)], 'int64')
        shape = shape.reshape((68, 2))
        img = np.array(r[5 + 4 + 68 * 2:], 'uint8')
        img = img.reshape((96, 96))
        return r['subject'], r['seq'], r['emotion'], r['strength'], r['image_no'], shape, img

    def get_data_for_learning(self):
        """
        機械学習用のデータを取得する.

        Returns
        -------
        下記の説明で，Nはサンプル数
        shapes : np.array float (N, 68, 2)
                 形状情報
        images : np.array uint8 (N, 96, 96)
                 画像
        targets : np.array int (N,)
                 ラベル（無表情が0）
        groups : list （N,)
                 グループ（被験者ID）
        """
        def extract_data_for_learning(df):
            shapes = np.array(df.iloc[:, 9:(9 + 68 * 2)]).reshape((-1, 68, 2))
            images = np.array(df.iloc[:, (9 + 68 * 2):]).reshape((-1, 96, 96))
            targets = np.array(df['emotion'], 'int')
            groups = df['subject'].tolist()
            return shapes, images, targets, groups
        
        # 強度が1.0（ピーク画像）かつ，感情ありの行のみ取得
        cond = (self.dataframe['strength'] == 1.0) & \
            (np.isnan(self.dataframe['emotion']) == 0)
        extracted_on = self.dataframe[cond]
        extracted_on = extracted_on.copy()
        
        # 強度が0.0（平静画像）かつ，感情あり，かつ被験者重複の場合は最初のみ
        cond = (self.dataframe['strength'] == 0.0) & \
            (np.isnan(self.dataframe['emotion']) == 0)
        extracted = self.dataframe[cond]
        extracted_off = extracted[extracted.duplicated(subset='subject') == 0]
        extracted_off = extracted_off.copy()
        extracted_off['emotion'] = 0

        df = pd.concat([extracted_on, extracted_off], axis=0, ignore_index=True)
        df = df.sort_values(by='subject')
        # import ipdb; ipdb.set_trace()
        return extract_data_for_learning(df)

    def get_data_for_learning_n_frames(self, n=5):
        """
        機械学習用のデータを取得する.
        平静（０）は無し．

        Parameters
        ----------
        n : int
            ピーク画像から過去 n フレーム目まで遡って，
            画像リストやshapeリストを取得する（ピーク画像を含む）．

        Returns
        -------
        以下の説明で，Nはサンプル数，nは引数のnを表す
        shapes : array (N, 68, 2, n)
                 形状．
        images : array (N, 96, 96, n)
                 画像．
        targets : int or None (N,)
                 正解ラベル．
        groups : list of string (N,)
                 被検者ID
        """
        # DataFrameのメソッドで適当なものがあるかもしれないが，
        # ぱっと思いつかないので，愚直に順番に見て行くことにする．

        shapes = []
        images = []
        targets = []
        groups = []
        
        # 被験者リスト
        subjects = sorted(self.dataframe['subject'].unique().tolist())
        for subj in subjects:
            # シーケンスナンバー
            seqs = sorted(self.dataframe[self.dataframe['subject'] == subj]['seq'].unique().tolist())
            # 感情無しならスキップ
            for seq in seqs:
                cond = ((self.dataframe['subject'] == subj) & (self.dataframe['seq'] == seq))
                ror = self.dataframe[cond]
                if np.all(np.isnan(ror['emotion'])):
                    continue
                if len(ror) < n:
                    print("image sequense length is less than n(=%d)." % n)
                ror = ror[-n:]
                shape = np.array(ror.iloc[:, 9:(9 + 68 * 2)]).reshape((-1, 68, 2))
                shape = shape.transpose((1, 2, 0)) # 時間軸を最終軸にする
                image = np.array(ror.iloc[:, (9 + 68 * 2):]).reshape((-1, 96, 96))
                image = image.transpose((1, 2, 0))
                
                shapes.append(shape)
                images.append(image)
                targets.append(ror['emotion'].iloc[0])
                groups.append(ror['subject'].iloc[0])
        return np.array(shapes), np.array(images), np.array(targets), groups
                
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
            'subject', 'seq', 'emotion', 'strength', 'image_no', 'bb_x',
            'bb_y', 'bb_w', 'bb_h'
        ]
        dtypes = [
            'object', 'int32', 'float64', 'float64', 'int32', 'int32', 'int32',
            'int32', 'int32'
        ]
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

        ckp = CKplus()
        for subj in ckp.subjects:
            for seq in ckp.get_seq_list(subj):
                emotion = ckp.get_emotion(subj, seq)
                images = ckp.get_images(subj, seq)

                for i in range(len(images)):
                    if len(images) == 1:
                        strength = 1
                    else:
                        strength = i / (len(images) - 1)

                    print(
                        "%s %d %d (%s, %.2f) " % (subj, seq, i + 1, emotion,
                                                  strength),
                        end='')
                    # 既に対処済だったらパス
                    if df_out is not None and \
                       np.any((df_out['subject'] == subj) &
                              (df_out['seq'] == seq) &
                              (df_out['image_no'] == i + 1)):
                        print("already processed")
                        continue
                    img = images[i]
                    dets = face_detector.detect(img)
                    if len(dets) == 0:
                        print("face not found.")
                        continue
                    # 最初の結果（もっとも大きい四角）のみ相手にする
                    det = dets[0]
                    landmarks = face_shape_predictor.predict(
                        img, det[0], det[1])
                    aligned_img = face_aligner.align_with_eq_hist(
                        img, landmarks)

                    data = [subj, seq, emotion, strength, i + 1]
                    data.extend([
                        det[0][0],
                        det[0][1],
                        det[1][0] - det[0][0],
                        det[1][1] - det[0][1],
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
