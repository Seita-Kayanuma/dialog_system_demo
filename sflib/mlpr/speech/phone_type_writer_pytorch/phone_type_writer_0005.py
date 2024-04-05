from .base import PhoneTypeWriterTorchModel, PhoneTypeWriter, phone_list
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class PhoneTypeWriterModel(PhoneTypeWriterTorchModel):
    def __init__(self, in_feature_dim):
        super(PhoneTypeWriterModel, self).__init__()
        self._feature_dim = in_feature_dim

        self.l1 = nn.Linear(self._feature_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.lstm = nn.LSTM(128, 128)
        self.l3 = nn.Linear(128, len(phone_list))

        self._context = None

    @property
    def hidden_feature_dim(self):
        return 128

    def reset_context(self):
        self._context = None

    def detach_context(self):
        if self._context is None:
            return
        self._context = tuple([c.clone().detach() for c in self._context])

    def calc_hidden_feature(self, feat: PackedSequence) -> PackedSequence:
        h1 = torch.tanh(self.l1(feat.data))
        h2 = torch.tanh(self.l2(h1))
        h2packed = PackedSequence(h2, feat.batch_sizes, feat.sorted_indices,
                                  feat.unsorted_indices)
        hlstm, context = self.lstm(h2packed, self._context)
        hlstm = torch.tanh(hlstm.data)
        hlstm_packed = PackedSequence(hlstm, feat.batch_sizes,
                                      feat.sorted_indices,
                                      feat.unsorted_indices)
        self._context = context
        return hlstm_packed

    def predict_log_probs_with_hidden_feature(self, feat: PackedSequence
                                              ) -> PackedSequence:
        h3 = self.l3(feat.data)
        log_probs = F.log_softmax(h3, dim=1)
        log_probs_packed = PackedSequence(log_probs,
                                          feat.batch_sizes,
                                          feat.sorted_indices,
                                          feat.unsorted_indices)
        return log_probs_packed


class PhoneTypeWriter0005PyTorch(PhoneTypeWriter):
    def __init__(self, feature_extractor, *args, **kwargs):
        super().__init__(feature_extractor, *args, **kwargs)
        self._model = PhoneTypeWriterModel(
            in_feature_dim=self.feature_extractor.feature_dim)
        if self.device is not None:
            self._model.to(self.device)

    @property
    def torch_model(self):
        return self._model
