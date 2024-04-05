from corpus.noise import NoiseBase
from sound.sigproc.spec_image \
    import SpectrogramImageGenerator
from sound.sigproc.noise import NoiseAdder
import wave
from os import path
from corpus.speech.trans import TransInfo
from corpus.speech.wav import WavDataWithTrans, WavDataWithTransManager
from corpus.speech.spec_image import SpecImageDataManager


class SpecImageDataManagerForNoise(SpecImageDataManager):
    def __init__(self,
                 noise_base: NoiseBase,
                 generator: SpectrogramImageGenerator,
                 noise_adder: NoiseAdder = None):
        self.__noise_base = noise_base
        super().__init__(
            WavDataWithTransInfoManagerForNoise(noise_base), generator,
            noise_adder)


# -----------------------------------------------------------------------------
class WavDataWithTransInfoManagerForNoise(WavDataWithTransManager):
    def __init__(self, noise_base: NoiseBase):
        self._noise_base = noise_base
        self.__id2data = {}

    def get_wav_filename(self, id):
        return path.basename(self._noise_base.get_wav_path(id))

    def get(self, id):
        if id in self.__id2data:
            return self.__id2data[id]

        wav_path = self._noise_base.get_wav_path(id)
        f = wave.open(wav_path, 'r')
        if f.getframerate() != 16000 \
           or f.getsampwidth() != 2 or f.getnchannels() != 1:
            raise RuntimeError("{} is not 16kHz, 16bit, mono".format(wav_path))
        num_samples = f.getnframes()
        f.close()

        trans_data = TransInfo(0, num_samples * 1000 // 16000)
        wav_data = WavDataWithTrans(wav_path, trans_data)
        result = [[wav_data]]
        self.__id2data = result
        return result

    def clear_wav_cache(self):
        for datas in self.__id2data.values():
            for data_list in datas:
                for data in data_list:
                    data.clear()

    def clear(self):
        self.__id2data = {}
