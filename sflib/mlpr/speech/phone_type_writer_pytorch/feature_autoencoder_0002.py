from .base import PhoneTypeWriterFeatureExtractor
from ....speech.feature.autoencoder_pytorch.base \
    import load as load_autoencoder_trainer
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence


class PhoneTypeWriterFeatureExtractorAutoEncoder0002(
        PhoneTypeWriterFeatureExtractor):
    def __init__(self,
                 autoencoder_version,
                 autoencoder_trainer_module_postfix,
                 autoencoder_trainer_class_postfix,
                 spec_image_shift=2,
                 batch_size=1,
                 device=None):
        # オートエンコーダの作成（とりあえずはCPUに読み込む）
        trainer = load_autoencoder_trainer(autoencoder_version,
                                           autoencoder_trainer_module_postfix,
                                           autoencoder_trainer_class_postfix,
                                           map_location='cpu')
        # オートエンコーダの取得
        autoencoder = trainer.autoencoder
        # AutoEncoderのファイル名（のベースを取得する）
        self._autoencoder_filename_base = \
            autoencoder.get_filename_base() + '+' + \
            trainer.get_filename_base()

        self._autoencoder = autoencoder
        if device is not None:
            self._autoencoder.to(device)
        self._autoencoder.share_memory()

        self._batch_size = batch_size
        self._spec_image_generators = []
        for i in range(self._batch_size):
            self._spec_image_generators.append(
                SpectrogramImageGenerator(image_shift=spec_image_shift))
        self._spec_image_shift = spec_image_shift
        self._device = device

    def get_filename_base(self):
        s = super().get_filename_base()
        s += '+' + self._autoencoder_filename_base
        s += '+' + "{}".format(self._spec_image_generators[0].image_shift)
        return s

    @property
    def feature_dim(self):
        return self._autoencoder.bottleneck_dim

    def reset(self):
        for gen in self._spec_image_generators:
            gen.reset()

    def calc(self, wav_list: list):
        batch = len(wav_list)

        # スペクトル画像生成器の足りない分は自動的に拡張する
        if batch > len(self._spec_image_generators):
            num = batch - len(self._spec_image_generators)
            for i in range(num):
                self._spec_image_generators.append(
                    SpectrogramImageGenerator(
                        image_shift=self._spec_image_shift))

        spec_images = []
        spec_image_nums = []
        for i, wav in enumerate(wav_list):
            img_list = self._spec_image_generators[i].input_wave(wav)
            spec_images.extend(img_list)
            spec_image_nums.append(len(img_list))
        if len(spec_images) == 0:
            return None
            
        spec_images = torch.tensor(np.float32(np.stack(spec_images)))
        b, w, h = spec_images.shape
        spec_images = spec_images.reshape(-1, 1, w, h)
        # import ipdb; ipdb.set_trace()
        if self._device is not None:
            spec_images = spec_images.to(self._device)
        x, l2 = self._autoencoder.encode(spec_images)

        x_list = []
        st = 0
        for i, l in enumerate(spec_image_nums):
            en = st + l
            x_list.append(x[st:en, :])
            st = en

        result = pack_sequence(x_list, enforce_sorted=False)
        return result

