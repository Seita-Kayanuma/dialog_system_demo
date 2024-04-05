# 0001 -> 0002
#   LSTMのパラメータを2倍（128->256)にした
#   LSTMの後に，中間層(256->128)を1層付け加えた
from .base import TurnDetector1d, TurnDetector1dTorchModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
import numpy as np


class TurnDetector1dTorchModel0002(TurnDetector1dTorchModel):
    def __init__(self, in_feature_dim):
        super().__init__()
        self._feature_dim = in_feature_dim
        
        self.linear_in1 = nn.Linear(self._feature_dim, 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_at = nn.Linear(128, 1)
        
        self._lstm1_context = None
        self._at_context = None
        self._yt_context = None

    def reset_context(self):
        self._lstm1_context = None
        self._at_context = None
        self._yt_context = None

    def detach_context(self):
        if self._lstm1_context is not None:
            self._lstm1_context = \
                tuple([c.clone().detach() for c in self._lstm1_context])
        if self._at_context is not None:
            self._at_context = self._at_context.clone().detach()
        if self._yt_context is not None:
            self._yt_context = self._yt_context.clone().detach()
            
    def forward(self, ut: PackedSequence,
                feat: PackedSequence,
                st: PackedSequence = None) -> PackedSequence:
        h_linear_in1 = torch.tanh(self.linear_in1(feat.data))
        h_linear_in1_packed = PackedSequence(h_linear_in1,
                                             feat.batch_sizes,
                                             feat.sorted_indices,
                                             feat.unsorted_indices)
        h_lstm1_packed, lstm1_context = self.lstm1(
            h_linear_in1_packed, self._lstm1_context)
        h_linear_1 = torch.tanh(self.linear_1(h_lstm1_packed.data))
        at_unlocked = torch.sigmoid(self.linear_at(h_linear_1))
        at_unlocked_packed = PackedSequence(at_unlocked, feat.batch_sizes,
                                            feat.sorted_indices,
                                            feat.unsorted_indices)
        # paddedのサイズはいずれも (L, B, *)
        # ロックされたa(t)の計算
        at_unlocked_padded, lengths = \
            pad_packed_sequence(at_unlocked_packed)
        batch_size = at_unlocked_padded.shape[1]
        ut_padded, _ = pad_packed_sequence(ut)

        # a(t), y(t)の計算
        if self._at_context is None:
            at_context = torch.zeros((1, batch_size, 1)).to(self.device)
        else:
            at_context = self._at_context
        
        if self._yt_context is None:
            yt_context = torch.zeros((1, batch_size, 1)).to(self.device)
        else:
            yt_context = self._yt_context

        at_padded = torch.zeros((0, batch_size, 1)).to(self.device)
        yt_padded = torch.zeros((0, batch_size, 1)).to(self.device)
        
        # stによるリセットの準備
        if st is not None:
            st_prev = torch.zeros((1, batch_size, 1)).to(self.device)
            st_padded, _ = pad_packed_sequence(st)
            
        for t in range(ut_padded.shape[0]):
            u = ut_padded[t:(t + 1), :, :]
            a = at_unlocked_padded[t:(t + 1), :, :]
            at = u * at_context + (1 - u) * a
            yt = at * ut_padded[t:(t + 1), :, :] + (1 - at) * yt_context

            if st is not None:
                st_current = st_padded[t:(t + 1), :, :]
                flag = (st_prev - st_current) > 0.9
                yt[flag] = 0.0
                st_prev = st_current

            at_padded = torch.cat([at_padded, at], dim=0)
            yt_padded = torch.cat([yt_padded, yt], dim=0)
            at_context = at
            yt_context = yt
            
        last_indices = (np.array(lengths) - 1).tolist()
        self._at_context = \
            at_padded[last_indices,
                      np.arange(batch_size), :].unsqueeze(0)
        self._yt_context = \
            yt_padded[last_indices, np.arange(batch_size), :].unsqueeze(0)
        self._lstm1_context = lstm1_context
        
        all_padded = torch.cat([yt_padded,
                                at_padded,
                                at_unlocked_padded], dim=2)
        all_packed = pack_padded_sequence(
            all_padded, lengths, enforce_sorted=False)
        return all_packed

    
class TurnDetector1d0002(TurnDetector1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = TurnDetector1dTorchModel0002(
            self.input_calculator.feature_dim)

    @property
    def torch_model(self) -> TurnDetector1dTorchModel:
        return self._model
