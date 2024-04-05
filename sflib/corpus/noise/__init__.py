from sflib import config
from os import path
from glob import glob
import pandas as pd
import re


class NoiseBase:
    """各種ノイズコーパスの基底クラス
    """

    def __init__(self, wav_dir_path, catalog_path):
        self._wav_dir_path = wav_dir_path
        self._catalog_path = catalog_path
        self.__read_catalog_file()

    def get_wav_dir_path(self):
        return self._wav_dir_path

    def get_id_list(self):
        return self._catalog['id'].tolist()

    def get_wav_path(self, id):
        return path.join(self.get_wav_dir_path(), id + '.wav')

    def get_wav_path_list(self):
        return [self.get_wav_path(id) for id in self.get_id_list()]

    def __read_catalog_file(self):
        # print(self.__catalog_path)
        if not path.exists(self._catalog_path):
            self.update_catalog()
        self._catalog = pd.read_pickle(self._catalog_path)

    def update_catalog(self):
        id_list = []
        for fullpath in glob(path.join(self.get_wav_dir_path(), '*.wav')):
            filename = path.basename(fullpath).replace('.wav', '')
            id_list.append(filename)
        df = pd.DataFrame({'id': sorted(id_list)})
        df.to_pickle(self._catalog_path)


class JEIDA(NoiseBase):
    """JEIDA Noise Database (JEIDA-NOISE)にアクセスするためのクラス
    http://research.nii.ac.jp/src/en/JEIDA-NOISE.html
    """

    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # 実際のWAVファイルがフラットに入っているディレクトリ．
    WAV_DIR_NAME = path.join('JEIDA_NOISE', 'wav,mono,16kHz')

    # カタログファイル名
    CATALOG_FILE_NAME = 'jeida_noise_catalog.df.pkl'

    def __init__(self, topdir_path=None):
        if topdir_path is None:
            topdir_path = self.__class__.DEFAULT_PATH
        wav_dir_path = path.join(topdir_path, self.__class__.WAV_DIR_NAME)
        catalog_path = path.join(topdir_path, self.__class__.CATALOG_FILE_NAME)
        super().__init__(wav_dir_path, catalog_path)


class SoundffectLab(NoiseBase):
    """フリーの効果音素材．
    https://soundeffect-lab.info/
    """

    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # 実際のWAVファイルがフラットに入っているディレクトリ．
    WAV_DIR_NAME = path.join('soundeffect-lab.info', 'orig')

    # カタログファイル名
    CATALOG_FILE_NAME = 'soundeffect_lab_catalog.df.pkl'

    def __init__(self, topdir_path=None):
        if topdir_path is None:
            topdir_path = self.__class__.DEFAULT_PATH
        wav_dir_path = path.join(topdir_path, self.__class__.WAV_DIR_NAME)
        catalog_path = path.join(topdir_path, self.__class__.CATALOG_FILE_NAME)
        super().__init__(wav_dir_path, catalog_path)


class Fujie(NoiseBase):
    """藤江が独自に集めた素材（と言っても大したものじゃないが）
    """

    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # 実際のWAVファイルがフラットに入っているディレクトリ．
    WAV_DIR_NAME = 'fujie'

    # カタログファイル名
    CATALOG_FILE_NAME = 'fujie.df.pkl'

    def __init__(self, topdir_path=None):
        if topdir_path is None:
            topdir_path = self.__class__.DEFAULT_PATH
        wav_dir_path = path.join(topdir_path, self.__class__.WAV_DIR_NAME)
        catalog_path = path.join(topdir_path, self.__class__.CATALOG_FILE_NAME)
        super().__init__(wav_dir_path, catalog_path)


class Silent(NoiseBase):
    """無音（もはやノイズじゃないけど...）
    """
    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # 実際のWAVファイルがフラットに入っているディレクトリ．
    WAV_DIR_NAME = 'silent'
    
    # カタログファイル名
    CATALOG_FILE_NAME = 'silent.df.pkl'

    def __init__(self, topdir_path=None):
        if topdir_path is None:
            topdir_path = self.__class__.DEFAULT_PATH
        wav_dir_path = path.join(topdir_path, self.__class__.WAV_DIR_NAME)
        catalog_path = path.join(topdir_path, self.__class__.CATALOG_FILE_NAME)
        super().__init__(wav_dir_path, catalog_path)
    
