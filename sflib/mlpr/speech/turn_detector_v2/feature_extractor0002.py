# coding: utf-8
# Spectrogram Auto Encoder と Phone Type Writer を用いた特徴抽出器
from .base import TurnDetectorFeatureExtractor
from ....speech.feature.autoencoder_v2.base \
    import construct_autoencoder
from ..phone_type_writer_v2.base \
    import construct_phone_type_writer
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class TurnDetectorFeatureExtractor0002(TurnDetectorFeatureExtractor):
    def __init__(self,
                 phone_type_writer_number,
                 phone_type_writer_trainer_number,
                 phone_type_writer_feature_extractor_number,
                 phone_type_writer_model_version,
                 autoencoder_number,
                 autoencoder_trainer_number,
                 autoencoder_model_version,
                 spec_image_shift=2,
                 batch_size=1):
        super().__init__()

        # 音素タイプライタの作成とロード
        phone_type_writer = construct_phone_type_writer(
            phone_type_writer_number, phone_type_writer_trainer_number,
            phone_type_writer_feature_extractor_number,
            ([], {
                'autoencoder_number': autoencoder_number,
                'autoencoder_trainer_number': autoencoder_trainer_number,
                'autoencoder_model_version': autoencoder_model_version,
                'spec_image_shift': spec_image_shift,
                'batch_size': batch_size
            }))
        phone_type_writer.load(phone_type_writer_model_version,
                               download=True,
                               download_overwrite=False)
        if phone_type_writer_model_version is None:
            phone_type_writer_model_version = \
                phone_type_writer.get_latest_model_version()
        phone_type_writer.torch_model.eval()
        self._phone_type_writer = phone_type_writer
        self._phone_type_writer_model_version = phone_type_writer_model_version

    @property
    def filename_base(self):
        return super().filename_base + \
            "{}V{:02d}".format(self._phone_type_writer.filename_base,
                               self._phone_type_writer_model_version)

    @property
    def device(self):
        return self._phone_type_writer.device

    def to(self, device):
        self._phone_type_writer.to(device)
    
    def reset(self):
        self._phone_type_writer.reset()

    def detach(self):
        self._phone_type_writer.detach()

    def calc(self, wav_list: list) -> list:
        ae_feat_list = self._phone_type_writer.calc_feature(wav_list)
        if ae_feat_list is None:
            return None
        ptw_feat_list = self._phone_type_writer.calc_hidden_feature(ae_feat_list)
        feat_list = [torch.cat([af, pf], dim=1).clone().detach() 
                     for af, pf in zip(ae_feat_list, ptw_feat_list)]
        return feat_list
        
    @property
    def feature_dim(self):
        return self._phone_type_writer.feature_extractor.feature_dim + \
            self._phone_type_writer.hidden_feature_dim
    
    @property
    def feature_rate(self):
        return self._phone_type_writer.feature_extractor.feature_rate
