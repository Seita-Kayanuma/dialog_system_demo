from ... import config
from os import path
from glob import glob
import pandas as pd
import re


class CSJ:
    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # コーパスが実際にあるディレクトリの名前（データディレクトリ内の相対パス）
    # この中を書き換えるのは禁止
    CORPUS_DIR_NAME = 'CSJ'

    # カタログファイル名
    CATALOG_FILE_NAME = "catalog.df.pkl"

    def __init__(self, topdir_path=None):
        if topdir_path is None:
            self.__path = self.__class__.DEFAULT_PATH
        else:
            self.__path = topdir_path
        self.__corpus_path = path.join(self.__path,
                                       self.__class__.CORPUS_DIR_NAME)
        self.__catalog_path = path.join(self.__path,
                                        self.__class__.CATALOG_FILE_NAME)
        self.__read_catalog_file()

    def __get_trn_form1_dir_path(self, core=False):
        core_dir = 'noncore'
        if core:
            core_dir = 'core'
        return path.join(self.__corpus_path, 'TRN', 'Form1', core_dir)

    def __get_trn_form2_dir_path(self, core=False):
        core_dir = 'noncore'
        if core:
            core_dir = 'core'
        return path.join(self.__corpus_path, 'TRN', 'Form2', core_dir)

    def __get_wav_dir_path(self, core=False):
        core_dir = 'noncore'
        if core:
            core_dir = 'core'
        return path.join(self.__corpus_path, 'WAV', core_dir)

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

    def is_core(self, id):
        r = self._catalog[self._catalog['id'] == id]['is_core']
        if len(r) == 0:
            return False
        return r.iloc[0]

    def get_wav_path(self, id):
        return path.join(
            self.__get_wav_dir_path(core=self.is_core(id)), id + ".wav")

    def get_trn_form1_path(self, id):
        return path.join(
            self.__get_trn_form1_dir_path(core=self.is_core(id)), id + ".trn")

    def get_trn_form2_path(self, id):
        return path.join(
            self.__get_trn_form2_dir_path(core=self.is_core(id)), id + ".trn")

    def update_catalog(self):
        core_list = []
        for fullpath in glob(
                path.join(self.__get_trn_form2_dir_path(core=True), '*.trn')):
            filename = path.basename(fullpath).replace('.trn', '')
            core_list.append(filename)
        # print(core_list)
        noncore_list = []
        for fullpath in glob(
                path.join(self.__get_trn_form2_dir_path(core=False), '*.trn')):
            filename = path.basename(fullpath).replace('.trn', '')
            noncore_list.append(filename)
        # print(noncore_list)

        id_list = core_list + noncore_list
        is_core_list = ([
            True,
        ] * len(core_list)) + ([
            False,
        ] * len(noncore_list))
        df = pd.DataFrame({'id': id_list, 'is_core': is_core_list})
        df.to_pickle(self.__catalog_path)
