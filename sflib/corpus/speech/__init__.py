from abc import abstractmethod, ABCMeta
from sflib.corpus.speech.trans import TransInfoManager
from sflib.corpus.speech.wav import WavDataWithTransManager
from sflib.corpus.speech.spec_image import SpecImageDataManager
from sflib.corpus.speech.duration import DurationInfoManager
from sflib.sound.sigproc.spec_image import SpectrogramImageGenerator
from sflib.sound.sigproc.noise import NoiseAdder


class CorpusSpeech(metaclass=ABCMeta):
    @abstractmethod
    def get_trans_info_manager(self) -> TransInfoManager:
        raise NotImplementedError()

    @abstractmethod
    def get_wav_data_with_trans_manager(self) -> WavDataWithTransManager:
        raise NotImplementedError()

    @abstractmethod
    def get_spec_image_data_manager(
            self,
            generator: SpectrogramImageGenerator = None,
            noise_adder: NoiseAdder = None) -> SpecImageDataManager:
        raise NotImplementedError()

    def get_duration_info_manager(self):
        return None
