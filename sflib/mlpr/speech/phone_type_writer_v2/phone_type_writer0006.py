from .base import PhoneTypeWriterTorchModel, PhoneTypeWriter, phone_list
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class PhoneTypeWriterModel(PhoneTypeWriterTorchModel):
    def __init__(self, in_feature_dim):
        super().__init__()
        self._feature_dim = in_feature_dim

        self.l1 = nn.Linear(self._feature_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.lstm = nn.LSTM(32, 32)
        self.l3 = nn.Linear(32, 32)
        self.l4 = nn.Linear(32, len(phone_list))

        self._context = None

    @property
    def hidden_feature_dim(self):
        return 32

    def reset_context(self):
        self._context = None

    def detach_context(self):
        if self._context is None:
            return
        self._context = tuple([c.clone().detach() for c in self._context])

    def calc_hidden_feature(self, feat: PackedSequence) -> PackedSequence:
        h1 = torch.relu(self.l1(feat.data))
        h2 = torch.relu(self.l2(h1))
        h2packed = PackedSequence(h2, feat.batch_sizes, feat.sorted_indices,
                                  feat.unsorted_indices)
        hlstm, context = self.lstm(h2packed, self._context)
        hlstm = torch.tanh(hlstm.data)
        h3 = torch.relu(self.l3(hlstm))
        h3_packed = PackedSequence(h3, feat.batch_sizes, feat.sorted_indices,
                                   feat.unsorted_indices)
        self._context = context
        return h3_packed

    def calc_output_from_hidden_feature(self, feat: PackedSequence
                                        ) -> PackedSequence:
        h4 = torch.relu(self.l4(feat.data))
        log_probs = F.log_softmax(h4, dim=1)
        log_probs_packed = PackedSequence(log_probs, feat.batch_sizes,
                                          feat.sorted_indices,
                                          feat.unsorted_indices)
        return log_probs_packed


class PhoneTypeWriter0006(PhoneTypeWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = PhoneTypeWriterModel(
            in_feature_dim=self.feature_extractor.feature_dim)

    @property
    def torch_model(self):
        return self._model
