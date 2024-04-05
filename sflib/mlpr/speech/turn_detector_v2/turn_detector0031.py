# 0021が基本．alphaを削除
from .base import TurnDetectorTorchModel, TurnDetector
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class TurnDetectorModel(TurnDetectorTorchModel):
    def __init__(self, in_feature_dim):
        super().__init__()
        self._feature_dim = in_feature_dim

        self.linear_in1 = nn.Linear(self._feature_dim, 512)
        self.lstm1 = nn.LSTM(512, 512)
        self.linear_1 = nn.Linear(512, 128)
        self.linear_out = nn.Linear(128, 2)

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
        h_lstm1 = torch.tanh(h_lstm1.data)
        h_linear1 = torch.tanh(self.linear_1(h_lstm1))
        h_all = self.linear_out(h_linear1)
        h_all_packed = PackedSequence(h_all, feat.batch_sizes,
                                      feat.sorted_indices,
                                      feat.unsorted_indices)
        self._context = context

        # import ipdb; ipdb.set_trace()
        return h_all_packed

    
class TurnDetector0031(TurnDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = TurnDetectorModel(self.feature_extractor.feature_dim)

    @property
    def torch_model(self) -> TurnDetectorTorchModel:
        return self._model

    def convert_result_to_task_softmax(self, result: PackedSequence) -> list:
        padded_result, result_len = pad_packed_sequence(result)
        result = []
        for b in range(padded_result.shape[1]):
            batch_result = []
            batch_result_len = result_len[b]

            y_dt = padded_result[:batch_result_len, b, 0:1]
            y_vad = padded_result[:batch_result_len, b, 1:2]

            batch_result.append(y_dt)
            batch_result.append(torch.sigmoid(y_vad))
            
            result.append(batch_result)
        return result
