# coding: utf-8
from ... import config
from os import path
from glob import glob
import pandas as pd
import re


class RWCP_SPXX:
    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # コーパスが実際にあるディレクトリの名前(96年度版と97年度版があるので二つ用意）
    CORPUS_DIR_NAMES = ['RWCP-SP96', 'RWCP-SP97']

    # カタログファイル名
    CATALOG_FILE_NAME = 'catalog.df.pkl'

    def __init__(self, topdir_path=None):
        if topdir_path is None:
            self.__path = self.__class__.DEFAULT_PATH
        else:
            self.__path = topdir_path
        self.__corpus_paths = [
            path.join(self.__path, corpus_dir)
            for corpus_dir in self.__class__.CORPUS_DIR_NAMES
        ]
        self.__catalog_path = path.join(self.__path,
                                        self.__class__.CATALOG_FILE_NAME)
        self.__read_catalog_file()

    def __read_catalog_file(self):
        # print(self.__catalog_path)
        if not path.exists(self.__catalog_path):
            self.update_catalog()
        self._catalog = pd.read_pickle(self.__catalog_path)

    def get_data_path(self):
        return self.__path

    def get_id_list(self, cond=None):
        if not cond:
            return self._catalog['id'].tolist()
        else:
            pat = re.compile(cond)
            c = self._catalog['id'].apply(lambda x: pat.match(x) is not None)
            return self._catalog[c]['id'].tolist()

    def get_wav_path(self, id):
        return self._catalog[self._catalog['id'] == id]['wav'].iloc[0]

    def get_dat_path(self, id):
        return self._catalog[self._catalog['id'] == id]['dat'].iloc[0]

    def update_catalog(self):
        # 音声ファイル名のリスト（フルパス）を取得しソート
        wavfile_list = []
        for corpus_path in self.__corpus_paths:
            wavfile_list.extend(
                glob(path.join(corpus_path, '*', '??_?_??.wav')))
        wavfile_list = sorted(wavfile_list)
        # 音声ファイル名の拡張子以外をファイルのIDとする
        id_list = [
            path.basename(name).replace('.wav', '') for name in wavfile_list
        ]
        # 音声ファイル名の拡張子をdatに変えたものがdatファイル名
        datfile_list = [name.replace('.wav', '.dat') for name in wavfile_list]

        df = pd.DataFrame({
            'id': id_list,
            'wav': wavfile_list,
            'dat': datfile_list
        })
        df.to_pickle(self.__catalog_path)
