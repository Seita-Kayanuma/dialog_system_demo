# codig: utf-8
from .... import config
import os
from os import path
from glob import glob
import pandas as pd
import re

from ....speech.nict_vad import run_vad

from ..trans import TransInfoManager, TransInfo
from ..duration import DurationInfoManager
from ..duration_v2 import DurationInfoV2Manager
from ..wav import WavDataWithTransManager
from ..spec_image import SpecImageDataManager
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
from ....sound.sigproc.noise import NoiseAdder

from .. import CorpusSpeech


class WASEDA_SOMA(CorpusSpeech):
    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # コーパスが実際にあるディレクトリの名前
    CORPUS_DIR_NAME = 'WASEDA-SOMA'

    # カタログファイル名
    CATALOG_FILE_NAME = 'catalog.df.pkl'

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

    def __read_catalog_file(self):
        if not path.exists(self.__catalog_path):
            self.update_catalog()
        self._catalog = pd.read_pickle(self.__catalog_path)

    def get_id_list(self, cond=None):
        if not cond:
            return self._catalog['id'].tolist()
        else:
            pat = re.compile(cond)
            c = self._catalog['id'].apply(lambda x: pat.match(x) is not None)
            return self._catalog[c]['id'].tolist()

    def get_wav_path(self, id):
        return self._catalog[self._catalog['id'] == id]['wav'].iloc[0]

    def get_data_path(self):
        return self.__path

    def get_safia_dir_path(self):
        return path.join(self.get_data_path(), 'wav_safia')

    def get_safia_wav_path(self, id):
        return path.join(self.get_safia_dir_path(), id + '.wav')

    def update_catalog(self):
        # 音声ファイル名のリスト（フルパス）を取得しソート
        wavfile_list = glob(path.join(self.__corpus_path, 'wav', '*.wav'))
        wavfile_list = sorted(wavfile_list)
        # 音声ファイル名の拡張子以外をファイルのIDとする
        id_list = [
            path.basename(name).replace('.wav', '') for name in wavfile_list
        ]
        df = pd.DataFrame({
            'id': id_list,
            'wav': wavfile_list,
        })
        df.to_pickle(self.__catalog_path)

    def get_trans_info_manager(self) -> TransInfoManager:
        return TransInfoManagerWasedaSoma(self)

    def get_wav_data_with_trans_manager(self) -> WavDataWithTransManager:
        return None

    def get_spec_image_data_manager(
            self,
            generator: SpectrogramImageGenerator = None,
            noise_adder: NoiseAdder = None) -> SpecImageDataManager:
        return None

    def get_duration_info_manager(self):
        return DurationInfoManagerWasedaSoma(self)


class TransInfoManagerWasedaSoma(TransInfoManager):
    def __init__(self, waseda_soma=None, verbose=True):
        if waseda_soma is not None:
            self._waseda_soma = waseda_soma
        else:
            self._waseda_soma = WASEDA_SOMA()
        self.__trans_info_dir_path = path.join(
            self._waseda_soma.get_data_path(), 'trans')
        self._verbose = verbose
        if not path.exists(self.__trans_info_dir_path):
            os.makedirs(self.__trans_info_dir_path, mode=0o755, exist_ok=True)

        # 特に必要ない場合もあるので，必要になるまでは
        # Noneにしておく
        self.__safia_wav_maker = None

    def get_trans_info_dir_path(self):
        return self.__trans_info_dir_path

    def build_trans_info(self, id):
        from .safia import SafiaWavMaker
        if self.__safia_wav_maker is None:
            self.__safia_wav_maker = SafiaWavMaker(self._waseda_soma)
        wav_file_path = self.__safia_wav_maker.get_filename_for_id(id)
        if not path.exists(wav_file_path):
            if self._verbose:
                print("Run SAFIA on {} ... ".format(id), end="", flush=True)
            self.__safia_wav_maker.make(id)
            if self._verbose:
                print("done")
        if self._verbose:
            print("Run VAD on {} ... ".format(id), end="", flush=True)
        vad_result = run_vad(wav_file_path)
        if self._verbose:
            print("done")
        info = []
        for r in vad_result:
            info.append([TransInfo(x[0], x[1]) for x in r])
        return info


class DurationInfoManagerWasedaSoma(DurationInfoManager):
    def __init__(self, waseda_soma=None):
        super().__init__()

        if waseda_soma is None:
            self._waseda_soma = WASEDA_SOMA()
        else:
            self._waseda_soma = waseda_soma

        self._trans_info_manager = TransInfoManagerWasedaSoma(
            self._waseda_soma)

        self._duration_info_dir_path = path.join(
            self._waseda_soma.get_data_path(), 'duration')
        if not path.exists(self._duration_info_dir_path):
            os.makedirs(
                self._duration_info_dir_path, mode=0o755, exist_ok=True)

        self._duration_info_eaf_dir_path = path.join(
            self._waseda_soma.get_data_path(), 'duration_eaf')
        if not path.exists(self._duration_info_eaf_dir_path):
            os.makedirs(
                self._duration_info_eaf_dir_path, mode=0o755, exist_ok=True)

    def get_duration_info_dir_path(self):
        return self._duration_info_dir_path

    def get_wav_filename(self, id):
        return self._waseda_soma.get_safia_wav_path(id)

    def get_vad_info(self, id):
        trans_info = self._trans_info_manager.get_trans_info(id)
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
        fullpath = self._waseda_soma.get_wav_path(id)
        media_type = "audio/wav"
        relative_path = path.join('..', 'wav_safia', id + '.wav')
        return fullpath, media_type, relative_path

    
class DurationInfoV2ManagerWasedaSoma(DurationInfoV2Manager):
    def __init__(self, waseda_soma=None):
        super().__init__()

        if waseda_soma is None:
            self._waseda_soma = WASEDA_SOMA()
        else:
            self._waseda_soma = waseda_soma

        self._trans_info_manager = TransInfoManagerWasedaSoma(
            self._waseda_soma)

        self._duration_info_dir_path = path.join(
            self._waseda_soma.get_data_path(), 'duration_v2')
        if not path.exists(self._duration_info_dir_path):
            os.makedirs(
                self._duration_info_dir_path, mode=0o755, exist_ok=True)

        self._duration_info_eaf_dir_path = path.join(
            self._waseda_soma.get_data_path(), 'duration_v2_eaf')
        if not path.exists(self._duration_info_eaf_dir_path):
            os.makedirs(
                self._duration_info_eaf_dir_path, mode=0o755, exist_ok=True)

    def get_duration_info_dir_path(self):
        return self._duration_info_dir_path

    def get_wav_filename(self, id):
        return self._waseda_soma.get_safia_wav_path(id)

    def get_vad_info(self, id):
        trans_info = self._trans_info_manager.get_trans_info(id)
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
        fullpath = self._waseda_soma.get_wav_path(id)
        media_type = "audio/wav"
        relative_path = path.join('..', 'wav_safia', id + '.wav')
        return fullpath, media_type, relative_path
    
