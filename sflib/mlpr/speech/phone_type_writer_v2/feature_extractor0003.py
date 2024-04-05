# SpectrogramImageAutoEncoderを使うもの
from .base import PhoneTypeWriterFeatureExtractor
from ....speech.feature.autoencoder_v2.base import construct_autoencoder
from ....sound.sigproc.spec_image_torch import SpectrogramImageGeneratorTorch
import numpy as np
import torch


class PhoneTypeWriterFeatureExtractor0003(
        PhoneTypeWriterFeatureExtractor):
    def __init__(self,
                 autoencoder_number,
                 autoencoder_trainer_number,
                 autoencoder_model_version=None,
                 spec_image_shift=2,
                 batch_size=1):
        super().__init__()
        
        autoencoder = construct_autoencoder(autoencoder_number,
                                            autoencoder_trainer_number)
        autoencoder.load(autoencoder_model_version, download=True,
                         download_overwrite=False)
        if autoencoder_model_version is None:
            autoencoder_model_version = autoencoder.get_latest_model_version()
            
        self._autoencoder = autoencoder
        self._autoencoder_model_version = autoencoder_model_version
        # self._autoencoder.share_memory()

        self._batch_size = batch_size
        self._spec_image_generators = []
        for i in range(self._batch_size):
            gen = SpectrogramImageGeneratorTorch(image_shift=spec_image_shift)
            gen.to(self._autoencoder.device)
            self._spec_image_generators.append(gen)
        self._spec_image_shift = spec_image_shift

    @property
    def filename_base(self):
        return super().filename_base + \
            "{}V{:02d}".format(self._autoencoder.filename_base,
                               self._autoencoder_model_version)

    @property
    def feature_dim(self):
        return self._autoencoder.bottleneck_dim

    def to(self, device):
        self._autoencoder.to(device)
        for gen in self._spec_image_generators:
            gen.to(device)
            
    def reset(self):
        for gen in self._spec_image_generators:
            gen.reset()

    def calc(self, wav_list: list) -> list:
        batch = len(wav_list)

        # スペクトル画像生成器の足りない分は自動的に拡張する
        if batch > len(self._spec_image_generators):
            num = batch - len(self._spec_image_generators)
            for i in range(num):
                gen = SpectrogramImageGeneratorTorch(image_shift=self._spec_image_shift)
                gen.to(self._autoencoder.device)
                self._spec_image_generators.append(gen)
                
        spec_images = []
        spec_image_nums = []
        for i, wav in enumerate(wav_list):
            imgs = self._spec_image_generators[i].input_wave(wav)
            spec_images.append(imgs)
            spec_image_nums.append(imgs.shape[0])
        spec_images = torch.cat(spec_images, dim=0)
        if len(spec_images) == 0:
            return None
        
        # spec_images = torch.tensor(np.float32(np.stack(spec_images)))
        # b, w, h = spec_images.shape
        # spec_images = spec_images.reshape(-1, 1, w, h)
        spec_images = spec_images.unsqueeze(1)
        # import ipdb; ipdb.set_trace()
        # if self._autoencoder.device is not None:
        #     spec_images = spec_images.to(self._autoencoder.device)
        x, l2 = self._autoencoder.encode(spec_images)
        x_list = []
        st = 0
        for i, l in enumerate(spec_image_nums):
            en = st + l
            x_list.append(x[st:en])
            st = en
            
        # import ipdb; ipdb.set_trace()
        return x_list
    
    @property
    def feature_rate(self):
        gen = self._spec_image_generators[0]
        return 16000 / gen.num_samples_per_image_shift
