# coding: utf-8
"""sflibの設定ファイルの読み込みを行うためのモジュール
"""
import os
from os import path
from configparser import ConfigParser

__DEFAULT_CONFIG_FILE_PATH = path.join(path.expanduser("~"), ".sflib.ini")
"""str:
設定ファイルのパス
"""


def _generate_default_config_file(filename):
    """デフォルト設定ファイルの生成
    
    Args:
      filename(str): ファイル名 
    """
    __config = ConfigParser()
    __config['common'] = {'topdir': path.join(path.expanduser('~'), 'sflib')}
    with open(filename, 'w') as f:
        __config.write(f)


# 設定ファイルを読み込む．無ければデフォルト設定ファイルを生成する．
if not path.exists(__DEFAULT_CONFIG_FILE_PATH):
    _generate_default_config_file(__DEFAULT_CONFIG_FILE_PATH)

# 読み込み
__config = ConfigParser()
__config.read([__DEFAULT_CONFIG_FILE_PATH], encoding='utf-8')


def get(section, name, default=None):
    """設定値を読み込む

    Args:
      section(str): 設定値のセクション名
      name(str): 設定値の名前
      default(str): 設定値の読み出しに失敗した場合のデフォルト値

    Returns:
      str :
      | 読み込んだ設定値．
      | 設定値が存在しなく，パラメータdefaultがNoneの場合はNone（または動作停止？）．
      | 設定値が存在しなく，パラメータdefaultがNone出ない場合はdefaultの値．
    """
    if default is None:
        return __config.get(section, name)
    else:
        return __config.get(section, name, default)


# well known options
# TOPDIR = get('common', 'topdir')
TOPDIR = "/Users/user/desktop/授業/lab/code/ResponseTimingEstimator_demo/sflib"
"""
str:
  | sflibの各種ファイルを保存するためのデータディレクトのトップディレクトリ．
  | デフォルトではホームディレクトリ直下のsflibフォルダ．
"""


# get package oriented data directory
def get_package_data_dir(package):
    """パッケージ名に応じたデータディレクトリを取得する．
      
    ディレクトリが存在しない場合は新規に作成される．

    Args:
      package (str): パッケージ名

    Returns:
      str: 
      | 対応するディレクトリ名．
      | デフォルトでは，{ホームディレクトリ}/sflib/{パッケージ名}
      | ただしパッケージ名はサブパッケージ名も含めて，.（ピリオド）が
        サブディレクトリの区切れ目になって掘られる．
    """
    import re
    d = re.sub('^sflib\.', '', package)
    d = d.replace('.', path.sep)
    r = path.join(TOPDIR, d)
    if not path.exists(r):
        os.makedirs(r, mode=0o755, exist_ok=True)
    return r
