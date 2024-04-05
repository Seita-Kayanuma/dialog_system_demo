# coding: utf-8
from . import WASEDA_SOMA
import os
from os import path
import numpy as np
import wave
from ....sound.safia import apply_safia


class SafiaWavMaker:
    """WASEDA-SOMAのSAFIA化されたWAVファイルを生成する為のクラス．
    転記情報を作るのに先立ってSAFIA化されたファイルが必要なため．
    """

    def __init__(self, waseda_soma=WASEDA_SOMA()):
        self.__waseda_soma = waseda_soma

    def make_dest_dir(self):
        """出力先のディレクトリを作成"""
        os.mkdir(self.__waseda_soma.get_safia_dir_path())

    def get_filename_for_id(self, id):
        """idに対応したSAFIA化ファイルのパスを取得する"""
        return os.path.join(self.__waseda_soma.get_safia_dir_path(),
                            id +'.wav')

    def exists(self, id):
        """既にSAFIA化されたファイルが存在するかどうか"""
        return path.exists(self.get_filename_for_id(id))

    def make(self, id):
        in_filename = self.__waseda_soma.get_wav_path(id)
        out_filename = self.get_filename_for_id(id)

        if not path.exists(self.__waseda_soma.get_safia_dir_path()):
            self.make_dest_dir()
        
        wf = wave.open(in_filename, 'r')
        channels = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
        wf.close()

        x = np.frombuffer(data, 'int16')
        x = x.reshape(-1, channels).T
        x_safia = apply_safia(x)
        data_safia = x_safia.T.ravel().tobytes()

        wf = wave.open(out_filename, 'w')
        wf.setnchannels(channels)
        wf.setframerate(16000)
        wf.setsampwidth(2)
        wf.writeframes(data_safia)
        wf.close()

        return
