# coding: utf-8
# 0001のTorch版
from .base import VoiceActivityDetectorFeatureExtractor
from ....speech.feature.autoencoder_v2.base \
    import construct_autoencoder
from ....sound.sigproc.spec_image_torch import SpectrogramImageGeneratorTorch
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence


class VoiceActivityDetectorFeatureExtractor0003(
        VoiceActivityDetectorFeatureExtractor):
    def __init__(self,
                 autoencoder_number,
                 autoencoder_trainer_number,
                 autoencoder_model_version=None,
                 spec_image_shift=2,
                 batch_size=1):
        super().__init__()
        
        # オートエンコーダの作成とロード
        autoencoder = construct_autoencoder(autoencoder_number,
                                            autoencoder_trainer_number)
        autoencoder.load(autoencoder_model_version, download=True,
                         download_overwrite=False)
        if autoencoder_model_version is None:
            autoencoder_model_version = autoencoder.get_latest_model_version()

        self._autoencoder = autoencoder
        self._autoencoder_model_version = autoencoder_model_version

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
    def device(self):
        return self._autoencoder.device

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
        for i, (wav, gen) in enumerate(
                zip(wav_list, self._spec_image_generators)):
            if len(wav) == 0:
                spec_images.append(gen._null_image)
                spec_image_nums.append(0)
            else:
                imgs = gen.input_wave(wav)
                spec_images.append(imgs)
                spec_image_nums.append(imgs.shape[0])
        spec_images = torch.cat(spec_images, dim=0)
        if len(spec_images) == 0:
            return None
        spec_images = spec_images.unsqueeze(1)
        x, l2 = self._autoencoder.encode(spec_images)
        x = x.clone().detach()
        
        feat_list = []
        for idx, sub_len in zip(np.cumsum([0] + spec_image_nums[:-1]),
                                spec_image_nums):
            feat_list.append(x[idx:(idx + sub_len), :])
        return feat_list

    @property
    def feature_dim(self):
        return self._autoencoder.bottleneck_dim

    @property
    def feature_rate(self):
        gen = self._spec_image_generators[0]
        return 16000 / gen.num_samples_per_image_shift
