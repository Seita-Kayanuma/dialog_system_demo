import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.autograd.set_detect_anomaly(True)


class VADCNNAE(nn.Module):

    def __init__(self, device, input_dim, hidden_dim): #, silence_encoding_type="concat"):
        super().__init__()
        
        self.device = device
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        self.fc = nn.Linear(hidden_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
        
        self.reset_state()

    def forward(self, inputs, input_lengths):

        t = max(input_lengths)
        batch = inputs.size(0)
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
        
        logits = self.fc(outputs)
        logits = logits.view(batch, -1)
        return logits
    
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
