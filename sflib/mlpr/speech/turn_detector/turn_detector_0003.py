from .base import TurnDetectorTorchModel, TurnDetector
from .base import TurnDetectorFeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


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


def apply_to_packed_tensor(fn, x):
    y_data = x.data
    if isinstance(fn, list):
        for fnc in fn:
            y_data = fnc(y_data)
    else:
        y_data = fn(y_data)
    packed_y = PackedSequence(y_data,
                              x.batch_sizes,
                              x.sorted_indices,
                              x.unsorted_indices)
    return packed_y


def apply_to_padded_tensor(fn, x, lengths):
    max_length, num_batches, dim = x.shape
    flag_lengths_is_nonzero = lengths > 0
    extracted_lengths = lengths[flag_lengths_is_nonzero]
    extracted_x = x[:, flag_lengths_is_nonzero, :]
    packed_x = pack_padded_sequence(extracted_x, extracted_lengths, enforce_sorted=False)
    packed_y = apply_to_packed_tensor(fn, packed_x)
    padded_y, _ = pad_packed_sequence(packed_y)
    y = torch.zeros(max_length, num_batches, padded_y.shape[-1])
    y = y.to(device=padded_y.device)
    y[:, flag_lengths_is_nonzero, :] = padded_y
    return y


class TurnDetectorModel(TurnDetectorTorchModel):
    def __init__(self, in_feature_dim):
        super().__init__()
        self._feature_dim = in_feature_dim

        self.linear_in1 = nn.Linear(self._feature_dim, 128)
        self.linear_in1_bn = nn.BatchNorm1d(128)
        self.lstm1 = nn.LSTM(128, 128)
        self.lstm1_bn = nn.BatchNorm1d(128)
        self.linear_vad_in = nn.Linear(128, 64)
        self.linear_vad_in_bn = nn.BatchNorm1d(64)
        self.linear_vad_out = nn.Linear(64, 2)  # 0, 1 (ON)
        self.linear_attention = nn.Linear(64, 1)
        self.linear_turn_out = nn.Linear(64, 3)  # 0, 1(TURN), 2(SHORT)
        self.linear_utterance_out = nn.Linear(64, 2)  # 0, 1(ON)

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
        # h_linear_in1 = torch.tanh(self.linear_in1(feat.data))
        # h_linear_in1_packed = PackedSequence(h_linear_in1, feat.batch_sizes,
        #                                      feat.sorted_indices,
        #                                      feat.unsorted_indices)
        # h_linear_in1_packed = apply_to_packed_tensor(
        #     self.linear_in1_bn, h_linear_in1_packed)
        h_linear_in1_packed = apply_to_packed_tensor(
            [self.linear_in1, torch.tanh, self.linear_in1_bn], feat)
        h_lstm1_packed, context = self.lstm1(h_linear_in1_packed, self._context)
        # h_lstm1 = torch.tanh(h_lstm1.data)
        # h_lstm1_packed = PackedSequence(h_lstm1,
        #                                 feat.batch_sizes,
        #                                 feat.sorted_indices,
        #                                 feat.unsorted_indices)
        # h_lstm1_packed = apply_to_packed_tensor(self.lstm1_bn, h_lstm1_packed)
        # h_linear_vad_in = torch.tanh(self.linear_vad_in(h_lstm1_packed.data))
        # h_linear_vad_in = apply_to_packed_tensor(
        #     self.linear_vad_in_bn, h_linear_vad_in)
        h_linear_vad_in_packed = apply_to_packed_tensor(
            [torch.tanh, self.lstm1_bn, self.linear_vad_in, self.linear_vad_in_bn],
            h_lstm1_packed)
        h_vad_data = self.linear_vad_out(h_linear_vad_in_packed.data)
        # h_attention = torch.sigmoid(self.linear_attention(h_linear_vad_in_packed))
        # h_linear_vad_in_packed = PackedSequence(h_linear_vad_in,
        #                                         feat.batch_sizes,
        #                                         feat.sorted_indices,
        #                                         feat.unsorted_indices)
        # h_linear_vad_in_padded, lenghts = pad_packed_sequence(
        #     h_linear_vad_in_packed)
        # h_attention_packed = PackedSequence(h_attention, feat.batch_sizes,
        #                                     feat.sorted_indices,
        #                                     feat.unsorted_indices)
        # h_attention_padded, _ = pad_packed_sequence(h_attention_packed)
        h_attention_packed = apply_to_packed_tensor(
            [self.linear_attention, torch.sigmoid], h_linear_vad_in_packed)
        h_linear_vad_in_padded, lengths = pad_packed_sequence(h_linear_vad_in_packed)
        h_attention_padded, _ = pad_packed_sequence(h_attention_packed)
        h_linear_vad_in_weighted, lstm_out_context = \
            calc_weight_delay(h_linear_vad_in_padded, h_attention_padded,
                              self._lstm_out_context)
        h_linear_vad_in_weighted_packed = pack_padded_sequence(
            h_linear_vad_in_weighted, lengths)
        h_turn_data = self.linear_turn_out(h_linear_vad_in_weighted_packed.data)
        h_utterance_data = self.linear_utterance_out(
            h_linear_vad_in_weighted_packed.data)
        h_all_data = torch.cat([h_turn_data, h_utterance_data, h_vad_data], dim=1)
        h_all_packed = PackedSequence(h_all_data, feat.batch_sizes,
                                      feat.sorted_indices,
                                      feat.unsorted_indices)
        self._context = context
        self._lstm_out_context = lstm_out_context

        return h_all_packed

    
    def forward_padded(self, feat: torch.Tensor, lengths: torch.Tensor=None) -> torch.Tensor:
        length, batch, dim = feat.shape
        if lengths is None:
            lengths = torch.tensor([length] * batch)
            lengths = lengths.to(device=feat.device)
        # feat_flat = feat.reshape(-1, dim)
        # h_linear_in1 = torch.tanh(self.linear_in1(feat_flat))
        # h_linear_in1_series = h_linear_in1.reshape(length, batch, -1)
        h_linear_in1_padded = apply_to_padded_tensor(
            [self.linear_in1, torch.tanh, self.linear_in1_bn], feat, lengths)
        h_lstm1_padded, context = self.lstm1(h_linear_in1_padded, self._context)
        # h_lstm1_flat = h_lstm1.reshape(-1, h_lstm1.shape[-1])
        # h_lstm1_flat = torch.tanh(h_lstm1_flat)
        # h_linear_vad_in_flat = torch.tanh(self.linear_vad_in(h_lstm1_flat))
        # h_vad_flat = self.linear_vad_out(h_linear_vad_in_flat)
        h_linear_vad_in_padded = apply_to_padded_tensor(
            [torch.tanh, self.lstm1_bn, self.linear_vad_in, self.linear_vad_in_bn],
            h_lstm1_padded, lengths)
        # h_attention_flat = torch.sigmoid(self.linear_attention(h_linear_vad_in_flat))
        # h_linear_vad_in_padded = h_linear_vad_in_flat.reshape(
        #     length, batch, h_linear_vad_in_flat.shape[-1])
        # h_attention_padded = h_attention_flat.reshape(
        #     length, batch, h_attention_flat.shape[-1])
        h_attention_padded = apply_to_padded_tensor(
            [self.linear_attention, torch.sigmoid], h_linear_vad_in_padded, lengths)
        h_linear_vad_in_weighted_padded, lstm_out_context = \
            calc_weight_delay(h_linear_vad_in_padded, h_attention_padded,
                              self._lstm_out_context)
        # h_linear_vad_in_weighted_flat = \
        #     h_linear_vad_in_weighted_padded.reshape(
        #         -1, h_linear_vad_in_weighted_padded.shape[-1])
        # h_turn_flat = self.linear_turn_out(h_linear_vad_in_weighted_flat)
        # h_utterance_flat = self.linear_utterance_out(
        #     h_linear_vad_in_weighted_flat)
        h_turn_padded = apply_to_padded_tensor(self.linear_turn_out,
            h_linear_vad_in_weighted_padded, lengths)
        h_utterance_padded = apply_to_padded_tensor(self.linear_utterance_out,
            h_linear_vad_in_weighted_padded, lengths)
        h_vad_padded = apply_to_padded_tensor(self.linear_vad_out,
            h_linear_vad_in_padded, lengths)
        h_all_padded = torch.cat([h_turn_padded, h_utterance_padded, h_vad_padded], dim=2)

        self._context = context
        self._lstm_out_context = lstm_out_context

        return h_all_padded


class TurnDetector0003(TurnDetector):
    def __init__(self, feature_extractor: TurnDetectorFeatureExtractor, *args,
                 **kwargs):
        super().__init__(feature_extractor, *args, **kwargs)
        self._model = TurnDetectorModel(feature_extractor.feature_dim)
        if self.device:
            self._model = self._model.to(self.device)

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
