# coding: utf-8
import numpy as np
from ....cloud.google import GoogleDriveInterface
from os import path
from .... import config
from abc import ABCMeta, abstractmethod

# 音素のリスト（最後をブランクラベルに設定する必要がある）
phone_list = [
    'N', 'N:', 'a', 'a:', 'b', 'by', 'ch', 'd', 'dy', 'e', 'e:', 'f',
    'g', 'gy', 'h', 'hy', 'i', 'i:', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o',
    'o:', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 'sp', 't', 'ts', 'ty', 'u',
    'u:', 'w', 'y', 'z', 'zy', 'nil'
]

# 音素記号から音素番号へ変換するディクショナリ
phone2id = dict([(p, i) for (i, p) in enumerate(phone_list)])


def convert_phone_to_id(phone_list):
    """
    音素リスト（文字列のリスト）を音素ID（整数のリスト）に変換する．
    """
    return [phone2id[phone] for phone in phone_list]


def convert_id_to_phone(label_list):
    """
    音素ID（整数のリスト）を音素リスト（文字列のリスト）に変換する．
    """
    return [phone_list[i] for i in label_list]


class PhoneTypeWriterFeatureExtractor(metaclass=ABCMeta):
    """
    特徴量抽出器
    """

    def get_filename_base(self):
        """
        モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    @abstractmethod
    def calc(self, x):
        """
        入力から特徴量を計算する．
        
        入力の形状は (サンプル数, 入力次元数)である必要がある．
        実際には特徴抽出器ごとに入力次元の拡張を行ってよい．
        例えばスペクトル画像を入力とする場合は，（サンプル数，高さ，幅）
        となることもある．
        
        戻り値は，(サンプル数, 特徴次元数) のnp.array
        """
        pass

    @abstractmethod
    def get_feature_dim(self):
        """
        特徴ベクトルの次元数を取得する．
        """
        pass


class PhoneTypeWriter(metaclass=ABCMeta):
    """
    PhoneTypeWriter（音素タイプライタ）の基底クラス
    """

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def get_filename_base(self):
        """
        モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    @abstractmethod
    def predict(self, x):
        """
        入力は(時系列, 入力次元数)．
        戻り値は音素文字列のリスト．
        """
        pass

    @abstractmethod
    def get_input_layer(self):
        """
        学習器のための入力層を取得する．

        戻り値は (the_input, input_length)．
        the_input は keras.Input で，名前が the_input
        input_length は keras.Input で，名前が input_length
        """
        pass

    @abstractmethod
    def get_output_layer(self):
        """
        学習器のための出力層を取得する．

        戻り値は keras.Layer （softmax 済の出力層）
        """
        pass

    @abstractmethod
    def save_model(self, filename):
        pass

    @abstractmethod
    def load_model(self, filename):
        pass


class PhoneTypeWriterTrainer(metaclass=ABCMeta):
    """
    PhoneTypeWriterTrainer（音素タイプライタ学習器）の基底クラス．
    """

    def __init__(self, phone_type_writer):
        self.phone_type_writer = phone_type_writer

    def get_filename_base(self):
        """
        モデルファイルの名前のヒントに使う文字列を取得する
        """
        return self.__class__.__name__

    def get_csv_log_filename(self):
        filename = self.phone_type_writer.get_filename_base()
        filename += '+' + self.phone_type_writer.feature_extractor.get_filename_base(
        )
        filename += '+' + self.get_filename_base() + '.csv'
        fullpath = path.join(
            config.get_package_data_dir(__package__), filename)
        return fullpath

    @abstractmethod
    def train(self, **kwargs):
        pass

    def upload_csv_log(self):
        filename = self.get_csv_log_filename()
        if path.exists(filename):
            g = GoogleDriveInterface(read_only=False)
            g.upload(filename, path.basename(filename), mediaType='text/csv')


def get_model_filename(trainer):
    filename = trainer.phone_type_writer.get_filename_base()
    filename += '+' + trainer.phone_type_writer.feature_extractor.get_filename_base(
    )
    filename += '+' + trainer.get_filename_base()
    fullpath = path.join(config.get_package_data_dir(__package__), filename)
    return fullpath


def save_phone_type_writer(trainer, upload=False):
    filename = get_model_filename(trainer)
    trainer.phone_type_writer.save_model(filename)
    if upload is True:
        g = GoogleDriveInterface(read_only=False)
        g.upload(filename, path.basename(filename))


def load_phone_type_writer(trainer, download=False):
    filename = get_model_filename(trainer)
    if download is True:
        g = GoogleDriveInterface()
        g.download(path.basename(filename), filename)
    trainer.phone_type_writer.load_model(filename)


# ----
def construct_feature_extractor_with_autoencoder(
        feature_extractor_version, autoencoder_version,
        autoencoder_trainer_module_postfix, autoencoder_trainer_class_postfix):
    module_name = "sflib.mlpr.speech.phone_type_writer.feature_autoencoder_%04d" \
        % feature_extractor_version
    class_name = "PhoneTypeWriterFeatureExtractorAutoEncoder%04d" \
        % feature_extractor_version
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    feature_extractor = cls(autoencoder_version,
                            autoencoder_trainer_module_postfix,
                            autoencoder_trainer_class_postfix)
    return feature_extractor


def construct_phone_type_writer(phone_type_writer_version, feature_extractor):
    module_name = "sflib.mlpr.speech.phone_type_writer.phone_type_writer_%04d" \
        % phone_type_writer_version
    class_name = "PhoneTypeWriter%04d" % phone_type_writer_version
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    phone_type_writer = cls(feature_extractor)
    return phone_type_writer


def construct_phone_type_writer_trainer(
        trainer_module_postfix, trainer_class_postfix, phone_type_writer):
    module_name = "sflib.mlpr.speech.phone_type_writer.trainer_%s" \
        % trainer_module_postfix
    class_name = "PhoneTypeWriterTrainer%s" % trainer_class_postfix
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    trainer = cls(phone_type_writer)
    return trainer
