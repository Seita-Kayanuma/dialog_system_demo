import numpy as np
from .base import PhoneTypeWriterFeatureExtractor
from ....speech.feature.autoencoder.base import load as load_autoencoder_trainer


class PhoneTypeWriterFeatureExtractorAutoEncoder0002(
        PhoneTypeWriterFeatureExtractor):
    def __init__(self,
                 autoencoder_version,
                 autoencoder_trainer_module_postfix,
                 autoencoder_trainer_class_postfix):
        # パラメータの読み込み
        trainer = load_autoencoder_trainer(autoencoder_version,
                                           autoencoder_trainer_module_postfix,
                                           autoencoder_trainer_class_postfix)
        # オートエンコーダの取得
        autoencoder = trainer.autoencoder
        # AutoEncoderのファイル名（のベースを取得する）
        self._autoencoder_filename_base = \
            autoencoder.get_filename_base() + '+' + \
            trainer.get_filename_base()

        self._autoencoder = autoencoder
        
    def get_filename_base(self):
        s = super().get_filename_base()
        s += '+' + self._autoencoder_filename_base
        return s

    def calc(self, x):
        x, l2 = self._autoencoder.encode(x)
        return x

    def get_feature_dim(self):
        return self._autoencoder.encoded_dim
