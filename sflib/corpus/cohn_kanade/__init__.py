# coding: utf-8
from os import path
from glob import glob
import pandas as pd
import numpy as np
import re
import cv2
from ... import config


class CKplus:
    """Cohn-Kanade Extended Data Setアクセスするためのクラス．
    """

    EMOTION_NAMES = [
        'neutral',
        'anger',
        'contempt',
        'disgust',
        'fear',
        'happy',
        'sadness',
        'surprise',
    ]
    """list[str]:
    感情ラベルのリスト．ラベルはこのインデクスでつけられる（neutralなら0）
    """

    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__), 'CK+')
    """str:
    デフォルトのデータセットパス
    """

    IMAGE_DIR = 'cohn-kanade-images'
    """str:
    画像ファイルのフォルダ（トップディレクトリからの相対パス）
    """

    EMOTION_DIR = 'Emotion'
    """str:
    表情ファイルのフォルダ（トップディレクトリからの相対パス）
    """

    # カタログ情報:
    CATALOG_FILE_PATH = path.join(
        config.get_package_data_dir(__package__), 'ck+catalog.pkl')
    """str:
    | カタログ情報を保存するファイルのパス（フルパス）．
    | pandas.DataFrameをpickleで書き出したもの．
    | 被検者ID(S000), シーケンス番号（int），画像枚数（int），表情ラベル（int）
      欠番あり．表情ラベルはついていないものも（多数）ある．
    """
    
    def __init__(self, topdir_path=None):
        """
        Args:
          topdir_path(str): データセットのトップディレクトリ．
            | 指定しない場合はデフォルトのパス（
            | :py:const:`DEFAULT_PATH`
            | ）になる．
        """        
        if topdir_path is None:
            self.__path = self.__class__.DEFAULT_PATH
        else:
            self.__path = topdir_path

        if not path.exists(self.__path):
            raise Exception("データディレクトリがありません．")

        self.__image_dir = path.join(self.__path, self.__class__.IMAGE_DIR)
        self.__emotion_dir = path.join(self.__path, self.__class__.EMOTION_DIR)

        self.__read_catalog_file()

    def __read_catalog_file(self):
        if not path.exists(self.__class__.CATALOG_FILE_PATH):
            self.update_catalog()
        self._catalog = pd.read_pickle(self.__class__.CATALOG_FILE_PATH)

    @staticmethod
    def get_image_path(subject, seq, num):
        """画像のパスを取得する．

        Args:
          subject(str): 被験者ID（S000など）．
          seq(int): シーケンス番号．
          num(int): 画像の番号．

        Returns:
          str: 画像ファイルのパス（フルパス）．
        """
        return path.join(subject, "%03d" % seq,
                         "%s_%03d_%08d.png" % (subject, seq, num))

    @property
    def subjects(self):
        """int: 全被験者数
        """
        return self._catalog['subject'].unique().tolist()

    def get_seq_list(self, subject):
        """被験者に対応するシーケンスのリストを取得する．

        Args:
          subject(str): 被験者ID（S000など）．

        Returns:
          list[int]: シーケンス番号のリスト．
        """
        cond = self._catalog['subject'] == subject
        return self._catalog[cond]['seq'].tolist()

    def get_num_images(self, subject, seq):
        cond = ((self._catalog['subject'] == subject) &
                (self._catalog['seq'] == seq))
        num_images = self._catalog[cond]['num_images']
        if len(num_images) == 1:
            num_images = int(num_images)
        else:
            num_images = 0
        return num_images

    def get_images(self, subject, seq):
        num_images = self.get_num_images(subject, seq)
        result = []
        for i in range(num_images):
            filepath = path.join(self.__image_dir,
                                 CKplus.get_image_path(subject, seq, i + 1))
            result.append(cv2.imread(filepath))
        return result

    def get_first_image(self, subject, seq):
        filepath = path.join(self.__image_dir,
                             CKplus.get_image_path(subject, seq, 1))
        return cv2.imread(filepath)

    def get_peak_image(self, subject, seq):
        num_images = self.get_num_images(subject, seq)
        filepath = path.join(self.__image_dir,
                             CKplus.get_image_path(subject, seq, num_images))
        return cv2.imread(filepath)

    def get_emotion(self, subject, seq):
        cond = ((self._catalog['subject'] == subject) &
                (self._catalog['seq'] == seq))
        result = self._catalog[cond]['emotion']
        if len(result) == 0:
            return None
        result = result.iloc[0]
        if np.isnan(result):
            return None
        return int(result)

    def update_catalog(self):
        dat_subject = []
        dat_seq = []
        dat_num_images = []
        dat_em = []

        subject_list = sorted(
            [path.basename(d) for d in glob(path.join(self.__image_dir, '*'))])
        for subject in subject_list:
            seq_list = sorted([
                path.basename(d)
                for d in glob(path.join(self.__image_dir, subject, '*'))
            ])
            for seq in seq_list:
                filelist = sorted([
                    path.basename(d) for d in glob(
                        path.join(self.__image_dir, subject, seq, '*.png'))
                ])
                num_images = len(filelist)
                # 感情ファイル
                em_files = glob(
                    path.join(self.__emotion_dir, subject, seq, '*.txt'))
                if len(em_files) == 1:
                    em = int(np.loadtxt(em_files[0]))
                else:
                    em = None
                #
                dat_subject.append(subject)
                dat_seq.append(int(seq))
                dat_num_images.append(num_images)
                dat_em.append(em)
        df = pd.DataFrame({
            'subject': dat_subject,
            'seq': dat_seq,
            'num_images': dat_num_images,
            'emotion': dat_em
        })
        df.to_pickle(self.__class__.CATALOG_FILE_PATH)
