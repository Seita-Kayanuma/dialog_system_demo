# coding: utf-8
from .trans import TransInfo, TransInfoManager
import numpy as np
import wave


class WavPool:
    def __init__(self):
        self._filename2data = {}
        self._filenames = []

    def get_wav_data(self, filename):
        # print(len(self._filename2data))
        if filename in self._filename2data:
            return self._filename2data[filename]
        else:
            wf = wave.open(filename, 'r')
            channels = wf.getnchannels()
            rawdata = wf.readframes(wf.getnframes())
            wf.close()
            data = np.frombuffer(rawdata, 'int16')
            data = data.reshape(-1, channels)
            self._filename2data[filename] = data
            self._filenames.append(filename)
            while len(self._filenames) > 10:
                fn = self._filenames.pop(0)
                del self._filename2data[fn]
            return data

        
wav_pool = WavPool()


class WavDataWithTrans:
    """
    音声ファイル名，チャネル，転記情報を保持し，
    要求に応じて音声を切り出して取得するためのクラス．
    Speech Auto Encoder や Phone Type Writer などの
    発話単位の音声や転記情報などが必要な場合に利用する．
    """

    def __init__(self, filename, trans, channel=0):
        self.__filename = filename
        self.__trans = trans
        self.__channel = channel
        # 切り出した音声ファイルのキャッシュ
        self.__cache = None
        # 切り出すサンプルの位置
        self.__start = trans.start * 16000 // 1000
        self.__end = trans.end * 16000 // 1000

    @property
    def wav(self):
        return self.get_wav_data()

    @property
    def num_samples(self):
        return self.__end - self.__start

    @property
    def trans(self):
        return self.__trans

    def get_wav_data(self):
        """
        音声データを取得する．
        一度呼ぶとデータはキャッシュされ，clearが呼ばれるまで保持される．
        """
        if self.__cache is not None:
            return self.__cache
        # # 音声データの読み込み（最終データのところまで読み込めば十分）
        # wf = wave.open(self.__filename, 'r')
        # channels = wf.getnchannels()
        # # マージンの関係で実際の音声ファイルの長さよりも後ろを取ろうと
        # # する場合があるので，その場合は最後まで読み込むように変更
        # end = self.__end
        # if end > wf.getnframes():
        #     end = wf.getnframes()
        # rawdata = wf.readframes(end)
        # wf.close()
        # # 変換と切り出し
        # data = np.frombuffer(rawdata, 'int16')
        # data = data.reshape(-1, channels)
        # print(self.__filename)
        data = wav_pool.get_wav_data(self.__filename)
        end = min(self.__end, data.shape[0])
        # 開始時間もマージンの関係で負になることがあるので，
        # その場合は0に調整
        start = self.__start
        if start < 0:
            start = 0        
        extracted = data[start:end, self.__channel].copy()
        wav_data = np.concatenate([np.zeros(start - self.__start, np.int16),
                                   extracted,
                                   np.zeros(self.__end - end, np.int16)])
        self.__cache = wav_data
        return self.__cache

    def clear(self):
        self.__cache = None

    def __repr__(self):
        return "WavDataWithTrans(cached=%s, length=%d, trans=%s)" % (
            self.__cache is not None, self.__end - self.__start,
            self.__trans.trans)


class WavDataWithTransManager:
    def __init__(self, trans_info_manager):
        self.__trans_info_manager = trans_info_manager
        self.__id2data = {}

    def get_wav_filename(self, id):
        raise NotImplementedError()

    def get(self, id):
        if id in self.__id2data:
            return self.__id2data[id]

        filename = self.get_wav_filename(id)
        trans_info = self.__trans_info_manager.get_trans_info(id)
        result = []
        for ch, trans_list in enumerate(trans_info):
            data_list = []
            for trans in trans_list:
                data_list.append(WavDataWithTrans(filename, trans, ch))
            result.append(data_list)
        self.__id2data[id] = result
        return result

    def clear_wav_cache(self):
        """
        保持する全てのWavDataWithTransのキャッシュをクリアする
        """
        for datas in self.__id2data.values():
            for data_list in datas:
                for data in data_list:
                    data.clear()

    def clear(self):
        """
        全てのデータをクリアする
        """
        self.__id2data = {}
