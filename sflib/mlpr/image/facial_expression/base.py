# coding: utf-8
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report
from ....cloud.google import GoogleDriveInterface
from os import path
from .... import config


class FacialExpressionFeatureExtractor:
    """
    特徴抽出器
    """

    def get_filename_base(self):
        """
        モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    def calc(self, images, shapes):
        """
        画像と形状から特徴量を抽出する．

        images と shapes の最初の次元はサンプル数で共通でなければならない．
        images の形状は(サンプル数, 高さ, 幅)，
        shapes の形状は(サンプル数, 点数, 2)
        である必要がある．
        ただし，手法によっては追加の次元（時間軸など）がある可能性がある

        
        戻り値は，(サンプル数, 特徴次元数) のnp.array
        """
        pass

    def get_feature_dim(self):
        """
        特徴ベクトルの次元数を取得する
        """
        pass


class FacialExpressionRecognizer:
    """
    FacialExpressionRecognizer（顔表情識別器）の基底クラス
    """

    def __init__(self, feature_extractor=None):
        self.feature_extractor = feature_extractor

    def get_filename_base(self):
        """
        モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    def fit_with_features(self, x, y, *args, validation_data=None, **kwargs):
        """
        特徴抽出済のデータでモデルを学習する．
        """
        pass

    def predict_with_features(self, x):
        """
        特徴抽出済のベクトルで予測をする．
        """
        pass

    def predict(self, img, shape):
        """
        画像，形状情報から予測をする
        """
        x = self.feature_extractor.calc(img, shape)
        return self.predict_with_features(x)

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass


class FacialExpressionRecognizerTrainer:
    """
    FacialExpressionRecognizerTrainer（顔表情認識器学習器）の基底クラス．
    """

    def __init__(self):
        self.recognizer = None

    def get_filename_base(self):
        """
        モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    def set_recognizer(self, recognizer):
        self.recognizer = recognizer

    def get_train_data(self):
        """
        標準的な学習データを取得する
        
        Returns
        -------
        下記のタプル
        x : np.array
            特徴ベクトルが並んだもの
        y : np.array
            ターゲット（正解）が並んだもの
        """
        pass

    def get_validation_data(self):
        """
        標準的な検証データを取得する
        
        Returns
        -------
        下記のタプル
        x : np.array
            特徴ベクトルが並んだもの
        y : np.array
            正解ラベルが並んだもの
        """
        pass

    def get_all_data_with_groups(self):
        """
        クロスバリデーションを行うために，
        全ての学習データとクロスバリデーションをするためのグループラベルを取得する．
        
        Returns
        -------
        下記のタプル
        x : np.array
            特徴ベクトルが並んだもの
        y : np.array
            正解ラベルが並んだもの
        g : np.array（listでも良い）
            グループラベルが並んだもの
        """
        pass

    def train(self, **kwargs):
        """
        実際に学習を行う．
        標準的な学習を行う（原則的には持ってる全てのデータを使って学習）．
        """
        x_train, y_train = self.get_train_data()
        validation_data = self.get_validation_data()
        self.recognizer.fit_with_features(
            x_train, y_train, validation_data=validation_data, **kwargs)

    def perform_cross_validation(self):
        """
        クロスバリデーションをする．
        識別結果を返す．

        TODO 結果のビジュアライゼーション
        """
        x, y, g = self.get_all_data_with_groups()
        logo = LeaveOneGroupOut()
        results = np.zeros(y.shape)
        for idx_train, idx_test in logo.split(x, y, g):
            print(idx_test)
            x_train = x[idx_train]
            y_train = y[idx_train]
            x_test = x[idx_test]
            y_test = y[idx_test]
            self.recognizer.fit_with_features(
                x_train, y_train, validation_data=(x_test, y_test))
            results[idx_test] = self.recognizer.predict_with_features(x_test)
        print(classification_report(y, results))
        print(confusion_matrix(y, results))
        return results


def get_recognizer_filename(trainer):
    filename = trainer.recognizer.get_filename_base()
    filename += '+' + trainer.recognizer.feature_extractor.get_filename_base()
    filename += '+' + trainer.get_filename_base()
    fullpath = path.join(config.get_package_data_dir(__package__), filename)
    return fullpath


def save_facial_expression_recognizer(trainer, upload=False):
    filename = get_recognizer_filename(trainer)
    trainer.recognizer.save_model(filename)
    if upload is True:
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename))


def load_facial_expression_recognizer(trainer, download=False):
    filename = get_recognizer_filename(trainer)
    if download is True:
        g = GoogleDriveInterface()
        g.download(path.basename(filename), filename)
    trainer.recognizer.load_model(filename)
