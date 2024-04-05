# coding: utf-8
from .... import config
from os import path
import os
from glob import glob
import numpy as np
import pandas as pd
import re
import wave

from ...speech.trans import TransInfoManager
from ...speech.wav import WavDataWithTransManager
from ...speech.spec_image import SpecImageDataManager
from ...speech.duration import DurationInfoManager
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
from ....sound.sigproc.noise import NoiseAdder
from ....sound.safia import apply_safia
from .trans_util import read_dat_file

from .. import CorpusSpeech


class RWCP_SPXX(CorpusSpeech):
    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = '/mnt/aoni04/fujie/sflib/corpus/speech/rwcp_spxx'
#     DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

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

    def get_safia_wav_path(self, id):
        return path.join(self.get_data_path(), 'wav_safia', id + '.wav')

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

    def get_trans_info_manager(self) -> TransInfoManager:
        return TransInfoManagerRWCP(self)

    def get_wav_data_with_trans_manager(self) -> WavDataWithTransManager:
        return WavDataWithTransManagerRWCP(self)

    def get_spec_image_data_manager(
            self,
            generator: SpectrogramImageGenerator = None,
            noise_adder: NoiseAdder = None) -> SpecImageDataManager:
        return SpecImageDataManagerRWCP(self, generator, noise_adder)

    def get_duration_info_manager(self):
        return DurationInfoManagerRWCP(self)


class TransInfoManagerRWCP(TransInfoManager):
    def __init__(self, rwcp=None):
        if rwcp is not None:
            self._rwcp = rwcp
        else:
            self._rwcp = RWCP_SPXX()
        self.__trans_info_dir_path = path.join(self._rwcp.get_data_path(),
                                               'trans')
        if not path.exists(self.__trans_info_dir_path):
            os.makedirs(self.__trans_info_dir_path, mode=0o755, exist_ok=True)

    def get_trans_info_dir_path(self):
        return self.__trans_info_dir_path

    def build_trans_info(self, id):
        filename = self._rwcp.get_dat_path(id)
        return read_dat_file(filename)


class WavDataWithTransManagerRWCP(WavDataWithTransManager):
    WAV_SAFIA_DIR_NAME = 'wav_safia'

    def __init__(self, rwcp=None):
        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp
        self.__wav_safia_dir_path = path.join(
            self._rwcp.get_data_path(), self.__class__.WAV_SAFIA_DIR_NAME)
        if not path.exists(self.__wav_safia_dir_path):
            os.makedirs(self.__wav_safia_dir_path, mode=0o755, exist_ok=True)

        super().__init__(TransInfoManagerRWCP(self._rwcp))

    def get_wav_filename(self, id):
        filename = path.join(self.__wav_safia_dir_path, id + '.wav')
        if not path.exists(filename):
            # オリジナル音声データの読み込み
            wf = wave.open(self._rwcp.get_wav_path(id), 'r')
            channels = wf.getnchannels()
            data = wf.readframes(wf.getnframes())
            wf.close()
            # SAFIAをかける
            x = np.frombuffer(data, 'int16')
            x = x.reshape(-1, channels).T
            x_safia = apply_safia(x)
            data_safia = x_safia.T.ravel().tobytes()
            # 書き込み
            wf = wave.open(filename, 'w')
            wf.setnchannels(channels)
            wf.setframerate(16000)
            wf.setsampwidth(2)
            wf.writeframes(data_safia)
            wf.close()
        return filename


class SpecImageDataManagerRWCP(SpecImageDataManager):
    def __init__(self,
                 rwcp=None,
                 generator: SpectrogramImageGenerator = None,
                 noise_adder: NoiseAdder = None):
        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp

        if generator is None:
            self._generator = SpectrogramImageGenerator()
        else:
            self._generator = generator

        self._noise_adder = noise_adder

        super().__init__(
            WavDataWithTransManagerRWCP(self._rwcp), self._generator,
            self._noise_adder)


class DurationInfoManagerRWCP(DurationInfoManager):
    def __init__(self, rwcp=None):
        super().__init__()

        if rwcp is None:
            self._rwcp = RWCP_SPXX()
        else:
            self._rwcp = rwcp

        self._duration_info_dir_path = path.join(self._rwcp.get_data_path(),
                                                 'duration')
        if not path.exists(self._duration_info_dir_path):
            os.makedirs(
                self._duration_info_dir_path, mode=0o755, exist_ok=True)

        self._duration_info_eaf_dir_path = path.join(
            self._rwcp.get_data_path(), 'duration_eaf')
        if not path.exists(self._duration_info_eaf_dir_path):
            os.makedirs(
                self._duration_info_eaf_dir_path, mode=0o755, exist_ok=True)

    def get_duration_info_dir_path(self):
        return self._duration_info_dir_path

    def get_wav_filename(self, id):
        return self._rwcp.get_safia_wav_path(id)

    def get_vad_info(self, id):
        filename = self._rwcp.get_dat_path(id)
        trans_info = read_dat_file(filename, max_gap=0)
        result = []
        for trans_info_list_for_channel in trans_info:
            result_for_channel = []
            for info in trans_info_list_for_channel:
                result_for_channel.append((
                    info.start,
                    info.end,
                ))
            result.append(result_for_channel)
        return result

    def get_duration_info_eaf_dir_path(self):
        return self._duration_info_eaf_dir_path

    def get_media_info_for_eaf(self, id):
        fullpath = self._rwcp.get_wav_path(id)
        media_type = "audio/wav"
        relative_path = path.join('..', 'wav_safia', id + '.wav')
        return fullpath, media_type, relative_path
