# coding: utf-8
from os import path
from glob import glob
import pandas as pd
import re
import cv2
from ... import config


class LFW:
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__), 'lfw')

    CATALOG_FILE_PATH = path.join(
        config.get_package_data_dir(__package__), 'catalog.pkl')

    def __init__(self, path=None):
        if path is None:
            self.__path = self.__class__.DEFAULT_PATH
        else:
            self.__path = path
        self.__read_catalog_file__()

    def __len__(self):
        return len(self._catalog)

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
        r = self._catalog.iloc[i, :]
        image_fullpath = path.join(self.__path, r['path'])
        img = cv2.imread(image_fullpath)
        return r['name'], r['num'], img

    def get_items_by_name(self, name):
        index_list = self._catalog[self._catalog['name'] ==
                                   name].index.tolist()
        return tuple([self.get_item_by_index(i) for i in index_list])

    def get_name_list(self):
        return self._catalog['name'].unique().tolist()

    def __read_catalog_file__(self):
        """カタログ情報を読み込む．必要であれば生成をする"""
        if not path.exists(self.__class__.CATALOG_FILE_PATH):
            self.__class__.update_catalog()
        self._catalog = pd.read_pickle(self.__class__.CATALOG_FILE_PATH)

    @classmethod
    def download(cls, topdir_path=None):
        """コーパスをダウンロードして展開する
        
        Parameters
        ----------
        topdir_path string ダウンロード・展開先のディレクトリ
        """
        # 出力ディレクトリが指定されていなければデフォルトパスにする
        if topdir_path is None:
            topdir_path = cls.DEFAULT_PATH
        # 出力ディレクトリの名前は lfw でなければならない
        assert path.basename(topdir_path) == 'lfw'
        # ファイルのダウンロード
        import urllib
        url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
        tgzpath = topdir_path + '.tgz'

        # 進陟状況報告の為のコールバック
        def progress(bc, bs, ts):
            ratio = 100.0 * bc * bs / ts
            print("\r%.2f%%" % ratio, end='')

        print("Downloading %s" % url)
        urllib.request.urlretrieve(url, tgzpath, reporthook=progress)
        print(" Done.")
        # 展開
        print("Extracting %s" % path.basename(tgzpath))
        import tarfile
        outdir_path = path.dirname(topdir_path)
        t = tarfile.open(tgzpath)
        t.extractall(outdir_path)
        print("Done.")

    @classmethod
    def update_catalog(cls):
        """コーパスのカタログファイルを作成する"""
        # フルパスのリスト
        filelist = glob(cls.DEFAULT_PATH + '/*/*.jpg')
        # DEFAULT_PATH からの相対パスにする
        filelist = sorted(
            [p.replace(cls.DEFAULT_PATH + '/', '') for p in filelist])
        # DataFrame作成用の人物名（namelist），番号（numlist）を作成する
        namelist = []
        numlist = []
        patt = re.compile('([^/]+)/.+_(\d+)\.jpg$')
        for f in filelist:
            m = patt.match(f)
            namelist.append(m.group(1))
            numlist.append(int(m.group(2)))
        # DataFrameの作成
        df = pd.DataFrame({'name': namelist, 'num': numlist, 'path': filelist})
        # 保存
        df.to_pickle(cls.CATALOG_FILE_PATH)
