# 0001 -> 0011
#   VADのON/OFFに加えて，VAD区間終了までの予測時間を出力するようにした
from .base import VoiceActivityDetectorTorchModel, VoiceActivityDetector
from .base import VoiceActivityDetectorFeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class VoiceActivityDetectorModel(VoiceActivityDetectorTorchModel):
    def __init__(self, in_feature_dim):
        super().__init__()
        self._feature_dim = in_feature_dim

        self.linear_in1 = nn.Linear(self._feature_dim, 128)
        self.lstm1 = nn.LSTM(128, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear_out_vad = nn.Linear(64, 1)  # voice probability
        self.linear_out_end = nn.Linear(64, 1)  # time to end (with sigmoid)
        
        self._context = None

    def reset_context(self):
        self._context = None

    def detach_context(self):
        if self._context is None:
            return
        self._context = tuple([c.clone().detach() for c in self._context])

    def forward(self, feat: PackedSequence) -> PackedSequence:
        h_linear_in1 = torch.tanh(self.linear_in1(feat.data))
        h_linear_in1_packed = PackedSequence(h_linear_in1, feat.batch_sizes,
                                             feat.sorted_indices,
                                             feat.unsorted_indices)
        h_lstm1, context = self.lstm1(h_linear_in1_packed, self._context)
        h_linear1 = torch.tanh(self.linear1(h_lstm1.data))
        h_vad = self.linear_out_vad(h_linear1)
        h_end = self.linear_out_end(h_linear1)
        h_all = torch.cat([h_vad, h_end], dim=1)
        h_all_packed = PackedSequence(h_all, feat.batch_sizes,
                                      feat.sorted_indices,
                                      feat.unsorted_indices)
        self._context = context

        # import ipdb; ipdb.set_trace()
        return h_all_packed

    
class VoiceActivityDetector0011(VoiceActivityDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = VoiceActivityDetectorModel(
            self.feature_extractor.feature_dim)
        
    @property
    def torch_model(self) -> VoiceActivityDetectorTorchModel:
        return self._model

    def convert_result_to_task_softmax(self, result: PackedSequence) -> list:
        padded_result, result_len = pad_packed_sequence(result)
        result = []
        for b in range(padded_result.shape[1]):
            batch_result = []
            batch_result_len = result_len[b]
            for i in range(2):
                batch_result.append(
                    torch.sigmoid(padded_result[:batch_result_len, b, i]))
            result.append(batch_result)
        return result
