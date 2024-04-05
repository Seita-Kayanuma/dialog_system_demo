import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


class VADSpec(nn.Module):

    def __init__(self, device, input_dim, hidden_dim): #, silence_encoding_type="concat"):
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
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
        
        self.reset_state()

    
    def forward(self, inputs, input_lengths):

        t = max(input_lengths)
        
        b, n, h, w = inputs.shape
        l2 = self.l2norm(inputs)
        inputs = self.normalize(inputs, l2)
        inputs = inputs.reshape(b*n, 1, h, w)
        inputs = self.bn1(F.relu(self.c1(inputs)))
        inputs = self.bn2(F.relu(self.c2(inputs)))
        inputs = self.bn3(F.relu(self.c3(inputs)))
        inputs = self.bn4(F.relu(self.c4(inputs))) 
        inputs = inputs.reshape(b, n, -1)

        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )

        
        outputs, self.hidden_state = self.lstm(inputs, self.hidden_state)

        outputs, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )

        
        logits = self.fc2(F.relu(self.fc1(outputs)))
        logits = logits.reshape(b, -1)
        
        return logits
    
    
    def get_features(self, inputs, input_lengths):
        
        with torch.no_grad():
        
            t = max(input_lengths)
            
            b, n, h, w = inputs.shape
            inputs = inputs.reshape(b*n, 1, h, w)
            inputs = self.bn1(F.relu(self.c1(inputs)))
            inputs = self.bn2(F.relu(self.c2(inputs)))
            inputs = self.bn3(F.relu(self.c3(inputs)))
            inputs = self.bn4(F.relu(self.c4(inputs))) 
            inputs = inputs.reshape(b, n, -1)
            
            inputs = rnn_utils.pack_padded_sequence(
                inputs, 
                input_lengths, 
                batch_first=True,
                enforce_sorted=False,
            )
            
            outputs, self.hidden_state = self.lstm(inputs, self.hidden_state)
            outputs, _ = rnn_utils.pad_packed_sequence(
                outputs, 
                batch_first=True,
                padding_value=0.,
                total_length=t,
            )
            
            feature_vad = self.fc1(outputs)
            feature_vad = feature_vad.reshape(b, t, -1)
        
        return feature_vad
    
    
    def reset_state(self):
        self.hidden_state = None

    
    def recog(self, inputs, input_lengths):
        outs = []
        with torch.no_grad():
            for i in range(len(input_lengths)):
                output = self.forward(inputs[i][:input_lengths[i]])
                outs.append(torch.sigmoid(output))            
        return outs


    def get_loss(self, probs, targets):
        return self.criterion(probs, targets.float())
    
    
    def get_evaluation(self, outputs, targets):
        pred = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        targets = targets.cpu().numpy()
        acc = accuracy_score(targets, pred)
        precision = precision_score(targets, pred)
        recall = recall_score(targets, pred)
        f1 = f1_score(targets, pred)
        return acc, precision, recall, f1
