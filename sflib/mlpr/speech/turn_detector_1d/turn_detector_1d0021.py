# 0011 -> 0021
#  u(t)が，u0(t)とu1(t)の二次元．
#  u0(t)が従来のu(t)と対応．u0(t)が 0.5 を下回った時に y(t) = 0 にリセットする．
#  u1(t)を応答の入力とする．alphaのロックはu0(t)で行う．
# 0001 -> 0011
#  ut < 0.5 （喋っている場合）は yt = 0 にする
from .base import TurnDetector1d, TurnDetector1dTorchModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
import numpy as np


class TurnDetector1dTorchModel0021(TurnDetector1dTorchModel):
    def __init__(self, in_feature_dim):
        super().__init__()
        self._feature_dim = in_feature_dim
        
        self.linear_in1 = nn.Linear(self._feature_dim, 128)
        self.lstm1 = nn.LSTM(128, 128)
        self.linear_at = nn.Linear(128, 1)
        
        self._lstm1_context = None
        self._u0t_context = None
        self._at_context = None
        self._yt_context = None

    def reset_context(self):
        self._lstm1_context = None
        self._u0t_context = None
        self._at_context = None
        self._yt_context = None

    def detach_context(self):
        if self._lstm1_context is not None:
            self._lstm1_context = \
                tuple([c.clone().detach() for c in self._lstm1_context])
        if self._u0t_context is not None:
            self._u0t_context = self._u0t_context.clone().detach()
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
        at_unlocked = torch.sigmoid(self.linear_at(h_lstm1_packed.data))
        at_unlocked_packed = PackedSequence(at_unlocked, feat.batch_sizes,
                                            feat.sorted_indices,
                                            feat.unsorted_indices)
        # paddedのサイズはいずれも (L, B, *)
        # ロックされたa(t)の計算
        at_unlocked_padded, lengths = \
            pad_packed_sequence(at_unlocked_packed)
        batch_size = at_unlocked_padded.shape[1]
        ut_padded, _ = pad_packed_sequence(ut)
        u0t_padded = ut_padded[:, :, 0:1]
        u1t_padded = ut_padded[:, :, 1:2]

        # a(t), y(t)の計算
        if self._at_context is None:
            at_context = torch.zeros((1, batch_size, 1), device=self.device)
        else:
            at_context = self._at_context
        
        if self._yt_context is None:
            yt_context = torch.zeros((1, batch_size, 1), device=self.device)
        else:
            yt_context = self._yt_context

        # u0tによるリセットの準備
        if self._u0t_context is None:
            u0t1_padded = torch.cat(
                [u0t_padded[:1], u0t_padded[:-1]], dim=0)
        else:
            u0t1_padded = torch.cat(
                [self._u0t_context, u0t_padded[:-1]], dim=0)
        reset_flag_padded = (u0t1_padded >= 0.5) & (u0t_padded < 0.5)
        
        # stによるリセットの準備
        if st is not None:
            st_padded, _ = pad_packed_sequence(st)
            st1_padded = torch.cat([
                torch.zeros((1, batch_size, 1), device=self.device),
                st_padded[:-1, :, :]])
            reset_flag_padded = \
                reset_flag_padded | ((st1_padded - st_padded) > 0.9)

        # import ipdb; ipdb.set_trace()
        
        at_list = []
        yt_list = []
            
        for t in range(ut_padded.shape[0]):
            u0 = u0t_padded[t:(t + 1), :, :]
            u1 = u1t_padded[t:(t + 1), :, :]
            a = at_unlocked_padded[t:(t + 1), :, :]
            a_locked = u0 * at_context + (1 - u0) * a
            
            yt = u1 * a_locked + (1 - a_locked) * yt_context
            
            reset_flag = reset_flag_padded[t:(t + 1), :, :]
            yt[reset_flag] = 0.0

            at_list.append(a_locked)
            yt_list.append(yt)
            at_context = a_locked
            yt_context = yt
            
        # import ipdb; ipdb.set_trace()

        at_locked_padded = torch.cat(at_list, dim=0)
        yt_padded = torch.cat(yt_list, dim=0)
        
        last_indices = (np.array(lengths) - 1).tolist()
        self._u0t_context = u0t_padded[
            last_indices, np.arange(batch_size), :].unsqueeze(0)
        self._at_context = at_locked_padded[
            last_indices, np.arange(batch_size), :].unsqueeze(0)
        self._yt_context = yt_padded[
            last_indices, np.arange(batch_size), :].unsqueeze(0)
        self._lstm1_context = lstm1_context
                
        all_padded = torch.cat([yt_padded,
                                at_locked_padded,
                                at_unlocked_padded], dim=2)
        all_packed = pack_padded_sequence(
            all_padded, lengths, enforce_sorted=False)
        return all_packed

    
class TurnDetector1d0021(TurnDetector1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = TurnDetector1dTorchModel0021(
            self.input_calculator.feature_dim)

    @property
    def torch_model(self) -> TurnDetector1dTorchModel:
        return self._model
