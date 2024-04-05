from corpus.noise import NoiseBase
from sound.sigproc.spec_image_torch \
    import SpectrogramImageGeneratorTorch
from corpus.speech.spec_image_torch import SpecImageDataManagerTorch
from corpus.noise.spec_image import WavDataWithTransInfoManagerForNoise
from sound.sigproc.noise import NoiseAdder


class SpecImageDataManagerForNoiseTorch(SpecImageDataManagerTorch):
    def __init__(self,
                 noise_base: NoiseBase,
                 generator: SpectrogramImageGeneratorTorch,
                 noise_adder: NoiseAdder = None):
        self.__noise_base = noise_base
        super().__init__(
            WavDataWithTransInfoManagerForNoise(noise_base), generator,
            noise_adder)
