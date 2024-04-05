# テスト用テンポラリ
from .base import TurnDetectorTorchModel, TurnDetector
from .base import TurnDetectorFeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class TurnDetectorModel(TurnDetectorTorchModel):
    def __init__(self, in_feature_dim):
        super().__init__()
        self._feature_dim = in_feature_dim

        self.linear_in1 = nn.Linear(self._feature_dim, 512)
        self.linear_in2 = nn.Linear(512, 512)
        self.lstm1 = nn.LSTM(512, 512)
        self.linear_1 = nn.Linear(512, 512)
        self.linear_out_turn = nn.Linear(512, 3)  # 0, 1(TURN), 2(SHORT)
        self.linear_out_utterance = nn.Linear(512, 2)  # 0, 1(ON)
        self.linear_out_vad = nn.Linear(512, 2)  # 0, 1 (ON)

        self._context = None

    def reset_context(self):
        self._context = None

    def detach_context(self):
        if self._context is None:
            return
        # デタッチしなかったら -> 二重にbackwardしようとするので怒られた
        # clone()しないでデタッチしたら？-> なんとも無い
        self._context = tuple([c.detach() for c in self._context])

    def forward(self, feat: PackedSequence) -> PackedSequence:
        h_linear_in1 = torch.tanh(self.linear_in1(feat.data))
        h_linear_in2 = torch.tanh(self.linear_in2(h_linear_in1))
        h_linear_in2_packed = PackedSequence(h_linear_in2, feat.batch_sizes,
                                             feat.sorted_indices,
                                             feat.unsorted_indices)
        h_lstm1, context = self.lstm1(h_linear_in2_packed, self._context)
        h_lstm1 = torch.tanh(h_lstm1.data)
        h_linear1 = torch.tanh(self.linear_1(h_lstm1))
        h_turn = self.linear_out_turn(h_linear1)
        h_utterance = self.linear_out_utterance(h_linear1)
        h_vad = self.linear_out_vad(h_linear1)
        h_all = torch.cat([h_turn, h_utterance, h_vad], dim=1)
        h_all_packed = PackedSequence(h_all, feat.batch_sizes,
                                      feat.sorted_indices,
                                      feat.unsorted_indices)
        self._context = context

        # import ipdb; ipdb.set_trace()
        return h_all_packed

    
class TurnDetector0099(TurnDetector):
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
            for s, e in ((0, 3), (3, 5), (5, 7)):
                batch_result.append(
                    torch.softmax(padded_result[:batch_result_len, b, s:e],
                                  dim=1))
            result.append(batch_result)
        return result
