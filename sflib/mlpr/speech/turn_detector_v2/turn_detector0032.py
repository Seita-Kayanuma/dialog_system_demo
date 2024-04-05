# 22が基本．alphaを削除
from .base import TurnDetectorTorchModel, TurnDetector
import torch
import torch.nn as nn
from torch.nn.utils.rnn \
    import PackedSequence, pad_packed_sequence, pack_padded_sequence


def calc_weight_delay(x, alpha, context=None):
    # x (L, B, D)
    # alpha (L, B, 1)
    # context (1, B, 1)
    length, batch, dim = x.shape
    y_list = []
    if context is None:
        xp = x[0, :, :]
    else:
        xp = x[0, :, :] * alpha[0, :, :] + \
             context[0, :, :] * (1 - alpha[0, :, :])
    y_list.append(xp)
    for i in range(1, length):
        xc = x[i, :, :]
        a = alpha[i, :, :]
        xp = xc * a + xp * (1 - a)
        y_list.append(xp)
    return torch.stack(y_list, dim=0), xp.unsqueeze(0)


class TurnDetectorModel(TurnDetectorTorchModel):
    def __init__(self, in_feature_dim):
        super().__init__()
        self._feature_dim = in_feature_dim

        self.linear_in1 = nn.Linear(self._feature_dim, 512)
        self.lstm1 = nn.LSTM(512, 512)
        self.linear_vad_in = nn.Linear(512, 256)
        self.linear_vad_out = nn.Linear(256, 1)  # VAD
        self.linear_attention = nn.Linear(256, 1)
        self.linear_turn_out = nn.Linear(256, 1)  # TURN

        self._context = None
        self._lstm_out_context = None

    def reset_context(self):
        self._context = None
        self._lstm_out_context = None

    def detach_context(self):
        if self._context is None:
            return
        self._context = tuple([c.clone().detach() for c in self._context])
        self._lstm_out_context = self._lstm_out_context.clone().detach()

    def forward(self, feat: PackedSequence) -> PackedSequence:
        h_linear_in1 = torch.tanh(self.linear_in1(feat.data))
        h_linear_in1_packed = PackedSequence(h_linear_in1, feat.batch_sizes,
                                             feat.sorted_indices,
                                             feat.unsorted_indices)
        h_lstm1, context = self.lstm1(h_linear_in1_packed, self._context)
        h_lstm1 = torch.tanh(h_lstm1.data)
        h_linear_vad_in = torch.tanh(self.linear_vad_in(h_lstm1))
        h_vad = self.linear_vad_out(h_linear_vad_in)
        h_attention = torch.sigmoid(self.linear_attention(h_linear_vad_in))
        h_linear_vad_in_packed = PackedSequence(h_linear_vad_in,
                                                feat.batch_sizes,
                                                feat.sorted_indices,
                                                feat.unsorted_indices)
        h_linear_vad_in_padded, lenghts = pad_packed_sequence(
            h_linear_vad_in_packed)
        h_attention_packed = PackedSequence(h_attention, feat.batch_sizes,
                                            feat.sorted_indices,
                                            feat.unsorted_indices)
        h_attention_padded, _ = pad_packed_sequence(h_attention_packed)
        h_linear_vad_in_weighted, lstm_out_context = \
            calc_weight_delay(h_linear_vad_in_padded, h_attention_padded,
                              self._lstm_out_context)
        h_linear_vad_in_weighted_packed = pack_padded_sequence(
            h_linear_vad_in_weighted, lenghts, enforce_sorted=False)
        h_turn = self.linear_turn_out(h_linear_vad_in_weighted_packed.data)
        h_all = torch.cat([h_turn, h_vad], dim=1)
        h_all_packed = PackedSequence(h_all, feat.batch_sizes,
                                      feat.sorted_indices,
                                      feat.unsorted_indices)
        self._context = context
        self._lstm_out_context = lstm_out_context

        # import ipdb; ipdb.set_trace()
        return h_all_packed


class TurnDetector0032(TurnDetector):
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

