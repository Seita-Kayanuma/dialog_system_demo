from os import path
import re
from glob import glob
import pandas as pd

from .... import config


class Rinus:
    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # コーパスが実際にあるディレクトリの名前（データディレクトリ内の相対パス）
    # この中を書き換えるのは禁止
    CORPUS_DIR_NAME = 'rinus'

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

    def get_id_list(self, cond=None):
        if not cond:
            return self._catalog['id'].tolist()
        else:
            pat = re.compile(cond)
            c = self._catalog['id'].apply(lambda x: pat.match(x) is not None)
            return self._catalog[c]['id'].tolist()

    def get_mp4_dir_path(self):
        return path.join(self.__corpus_path, 'mp4')
        
    def get_mp4_path(self, id):
        return path.join(
            self.get_mp4_dir_path(), id + ".MP4")

    def get_eaf_kobayashi_dir_path(self):
        return path.join(self.__corpus_path, 'eaf_kobayashi')
    
    def get_eaf_kobayashi_path(self, id):
        return path.join(
            self.get_eaf_kobayashi_dir_path(), id + ".eaf")
        
    def __read_catalog_file(self):
        # print(self.__catalog_path)
        if not path.exists(self.__catalog_path):
            self.update_catalog()
        self._catalog = pd.read_pickle(self.__catalog_path)
    
    def update_catalog(self):
        id_list = []
        for fullpath in sorted(glob(
                path.join(self.get_mp4_dir_path(), '*.MP4'))):
            filename = path.basename(fullpath).replace('.MP4', '')
            id_list.append(filename)
        df = pd.DataFrame({'id': id_list})
        df.to_pickle(self.__catalog_path)

