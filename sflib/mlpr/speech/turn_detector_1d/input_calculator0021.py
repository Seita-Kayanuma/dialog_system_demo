# 0001 -> 0021
#   Voice Activity Detector の10番台を使うように変更．
#   u(t)が2次元になり，1次元目が非音声らしさ（従来のu(t)），
#   2次元目が音声区間終了までの予測時間に対応したu(t)となる
from .base import InputCalculator
from ..voice_activity_detector.base import construct_voice_activity_detector
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import Tuple


class InputCalculator0021(InputCalculator):
    def __init__(self,
                 voice_activity_detector_number,
                 voice_activity_detector_trainer_number,
                 voice_activity_detector_feature_extractor_number,
                 voice_activity_detector_feature_extractor_construct_args,
                 voice_activity_detector_model_version=None,
                 spec_image_shift=2,
                 batch_size=1):
        super().__init__()

        voice_activity_detector = construct_voice_activity_detector(
            voice_activity_detector_number,
            voice_activity_detector_trainer_number,
            voice_activity_detector_feature_extractor_number,
            voice_activity_detector_feature_extractor_construct_args)
        voice_activity_detector.load(voice_activity_detector_model_version,
                                     download=True, download_overwrite=False)
        if voice_activity_detector_model_version is None:
            voice_activity_detector_model_version = \
                voice_activity_detector.get_latest_model_version()

        self._voice_activity_detector = voice_activity_detector
        self._voice_activity_detector_model_version = \
            voice_activity_detector_model_version

    @property
    def filename_base(self):
        return super().filename_base + \
            "{}V{:02d}".format(self._voice_activity_detector.filename_base,
                               self._voice_activity_detector_model_version)

    @property
    def device(self):
        return self._voice_activity_detector.device

    def to(self, device):
        self._voice_activity_detector.to(device)

    def reset(self):
        self._voice_activity_detector.reset()

    def detach(self):
        self._voice_activity_detector.detach()

    @property
    def feature_dim(self):
        return self._voice_activity_detector.feature_extractor.feature_dim

    @property
    def feature_rate(self):
        return self._voice_activity_detector.feature_extractor.feature_rate

    @property
    def ut_dim(self):
        return 2

    def calc(self, wav_list: list) -> Tuple[list, list]:
        feat_list = \
            self._voice_activity_detector.feature_extractor.calc(wav_list)
        if feat_list is None:
            return None, None
        feat = pad_sequence(feat_list)
        if feat.shape[0] == 0:
            return None
        out = self._voice_activity_detector.forward_core_padded(feat)

        feat = feat.detach()
        out = out.detach()
        
        # outのサイズは，(L, B, 2)
        # sigmoidをかけて0〜1にする．
        # 最初の次元は0〜1を逆にする．
        out = torch.sigmoid(out)
        out[:, :, 0] = 1.0 - out[:, :, 0]
        # import ipdb; ipdb.set_trace()
        ut_list = []
        for i, length in enumerate([len(f) for f in feat_list]):
            ut_list.append(out[:length, i, :])
        return ut_list, feat_list
