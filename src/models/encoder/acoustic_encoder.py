import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
    def forward(self, x):
        y = x.reshape(x.shape[0], -1)
        y = torch.pow(y, 2.0)
        y = torch.sum(y, (1, ), keepdim=True)
        y = torch.sqrt(y)
        y = y / np.prod(x.shape[1:])
        y = y + 1e-5
        return y


class Normalize(nn.Module):
    """
    与えられた係数で正規化する
    """
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x, coef):
        coef = coef.reshape(-1, 1, 1, 1)
        return x / coef


class AcousticEncoderSpec(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, encoding_dim):
        super().__init__()
        
        self.device = device
        self.l2norm = L2Norm()
        self.normalize = Normalize()
        self.c1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.c4 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn4 = nn.BatchNorm2d(32)
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, encoding_dim)        
        self.reset_state()


    def forward(self, inputs, input_lengths):
        """ Fusion multi-modal inputs
        Args:
            inputs: acoustic feature (B, L, input_dim)
            
        Returns:
            logits: acoustic representation (B, L, encoding_dim)
        """
        b, n, h, w = inputs.shape        
        t = max(input_lengths)
        inputs = inputs.reshape(b*n, 1, h, w)
        l2 = self.l2norm(inputs)
        inputs = self.normalize(inputs, l2)
        inputs = F.relu(self.bn1(self.c1(inputs)))
        inputs = F.relu(self.bn2(self.c2(inputs)))
        inputs = F.relu(self.bn3(self.c3(inputs)))
        inputs = F.relu(self.bn4(self.c4(inputs)))
        inputs = inputs.reshape(b, n, -1)
        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )

        # outputs : batch_size x maxlen x hidden_dim
        # rnn_h   : num_layers * num_directions, batch_size, hidden_dim
        # rnn_c   : num_layers * num_directions, batch_size, hidden_dim
        outputs, self.hidden_state = self.lstm(inputs, self.hidden_state)
        h, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )
        
        logits = self.fc(h)
        return logits
    
    
    def reset_state(self):
        self.hidden_state = None