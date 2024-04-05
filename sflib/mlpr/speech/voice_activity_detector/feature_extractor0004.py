# coding: utf-8
# 0003で，Delta L2 を特徴量として入れるバージョン
from .base import VoiceActivityDetectorFeatureExtractor
from ....speech.feature.autoencoder_v2.base \
    import construct_autoencoder
from ....sound.sigproc.spec_image_torch import SpectrogramImageGeneratorTorch
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence


class VoiceActivityDetectorFeatureExtractor0004(
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

        self._last_l2 = [0.0] * self._batch_size

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
        self._last_l2 = [0.0] * self._batch_size

    def calc(self, wav_list: list) -> list:
        batch = len(wav_list)

        # スペクトル画像生成器の足りない分は自動的に拡張する
        if batch > len(self._spec_image_generators):
            num = batch - len(self._spec_image_generators)
            for i in range(num):
                gen = SpectrogramImageGeneratorTorch(
                    image_shift=self._spec_image_shift)
                gen.to(self._autoencoder.device)
                self._spec_image_generators.append(gen)
            self._last_l2.extend([0.0] * num)
            self._batch_size = batch

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
        l2 = l2.clone().detach().cpu().numpy()
        
        feat_list = []
        for i, (idx, sub_len) in enumerate(
                zip(np.cumsum([0] + spec_image_nums[:-1]),
                    spec_image_nums)):
            x_sub = x[idx:(idx + sub_len), :]
            l2_sub = l2[idx:(idx + sub_len), :]
            if sub_len > 0:
                l2_sub_pre = np.concatenate([[[self._last_l2[i]]],
                                             l2_sub[:-1, :]], axis=0)
                dl2 = l2_sub - l2_sub_pre
                dl2 = torch.tensor(dl2, dtype=torch.float32)
                dl2 = dl2.to(self._autoencoder.device)
                x_sub = torch.cat([x_sub, dl2], dim=1)
                self._last_l2[i] = l2_sub[-1, 0]
            else:
                z = torch.zeros(x_sub.shape[0], 1)
                z = z.to(self._autoencoder.device)
                x_sub = torch.cat(
                    [x_sub, z], dim=1)
                
            feat_list.append(x_sub)
            
        # import ipdb; ipdb.set_trace()
        return feat_list

    @property
    def feature_dim(self):
        return self._autoencoder.bottleneck_dim + 1

    @property
    def feature_rate(self):
        gen = self._spec_image_generators[0]
        return 16000 / gen.num_samples_per_image_shift
