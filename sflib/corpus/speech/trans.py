# coding: utf-8
from os import path


class TransInfo:
    """
    音声の転記情報．
    音声区間情報と，その間の発話内容（あれば）を保持する．
    """

    def __init__(self, start, end, trans='', pron=''):
        """
        コンストラクタ
        
        Parameters
        ----------
        start : int
          音声開始時刻（ミリ秒）
        end : int
          音声終了時刻（ミリ秒）
        trans : str
          転記情報（仮名漢字混じり文）
        pron : str
          読み（音素列）
        """
        self.__start = start
        self.__end = end
        self.__trans = trans
        self.__pron = pron

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def trans(self):
        return self.__trans

    @property
    def pron(self):
        return self.__pron

    def __repr__(self):
        return "TransInfo(%d, %d, %s, %s)" % (self.__start, self.__end,
                                              self.__trans, self.__pron)


class TransInfoManager:
    """
    音声ファイルの転記情報を管理するクラス
    """

    def __init__(self):
        pass

    def get_trans_info_dir_path(self):
        """
        転記情報を保存するディレクトリのパス（フルパス）を取得する．
        継承クラスで実装する必要がある．
        また，このメソッドが呼ばれた以降は当該ディレクトリは存在する
        ものとして実行される（したがって存在しなければ作成する必要がある）．
        """
        raise NotImplementedError()

    def build_trans_info(self, id):
        """
        IDに対応する転記情報を構築する．
        継承クラスで実装する必要がある．
        転記情報は，音声のチャネル毎のTransInfoのリストとする．
        例えば，ステレオ音声の場合は
        info[0]がチャネル0に対応するTransInfoのリスト，
        info[1]がチャネル1に対応するTransInfoのリストとなる．
        """
        raise NotImplementedError()

    def get_trans_info_path(self, id):
        """
        IDに対応する転記情報ファイルのパスを取得する．
        """
        return path.join(self.get_trans_info_dir_path(), id + '.trans.txt')

    def get_trans_info(self, id):
        filename = self.get_trans_info_path(id)
        if not path.exists(filename):
            info = self.build_trans_info(id)
            self._write_trans(info, filename)
        else:
            info = self._read_trans(filename)
        return info

    def _write_trans(self, info, filename):
        with open(filename, 'w') as f:
            f.write("%d\n" % len(info))
            for ch, t_list in enumerate(info):
                for t in t_list:
                    f.write("%d,%d,%d,%s,%s\n" % (
                        ch,
                        t.start,
                        t.end,
                        t.trans,
                        t.pron,
                    ))

    def _read_trans(self, filename):
        with open(filename, 'r') as f:
            num_channels = int(f.readline())
            info = [[] for i in range(num_channels)]
            while True:
                line = f.readline()
                if line is None or len(line) == 0:
                    break
                ch, s, e, t, p = line.rstrip().split(',')
                info[int(ch)].append(TransInfo(int(s), int(e), t, p))
        return info
