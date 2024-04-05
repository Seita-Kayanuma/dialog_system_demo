import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.autograd.set_detect_anomaly(True)


class AcousticEncoderCNNAE(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, encoding_dim):
        super().__init__()
        
        self.device = device
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
        b, n, h = inputs.shape
        t = max(input_lengths)
       
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



class AcousticEncoderCNNAE2(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, encoding_dim):
        super().__init__()
        
        self.device = device
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        self.fc = nn.Linear(hidden_dim, encoding_dim)
        self.fc2 = nn.Linear(encoding_dim, 1)
        self.relu = nn.ReLU()
        self.criterion = nn.MSELoss(reduction='sum').to(device)
        self.reset_state()

    
    def forward(self, batch, split='train'):
        targets = batch[7].float().to(self.device)
        input_lengths = batch[9]
        inputs = batch[16].float().to(self.device)
        self.reset_state()
        
        batch_size, _, _ = inputs.shape
        t = max(input_lengths)

        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )
        outputs, self.hidden_state = self.lstm(inputs, self.hidden_state)
        h, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )        
        outputs = self.fc2(self.relu(self.fc(h)))
        outputs = outputs.view(batch_size, -1)
        
        batch_loss, batch_acc, batch_precision, batch_recall, batch_f1 = 0, 0, 0, 0, 0
        for i in range(batch_size):
            batch_loss = batch_loss + self.criterion(outputs[i][:input_lengths[i]], targets[i][:input_lengths[i]])
            acc, precision, recall, f1 = self.get_evaluation(outputs[i][:input_lengths[i]], targets[i][:input_lengths[i]])
            batch_acc = batch_acc + acc
            batch_precision = batch_precision + precision
            batch_recall = batch_recall + recall
            batch_f1 = batch_f1 + f1
        batch_loss = batch_loss / float(batch_size)
        batch_acc = batch_acc / float(batch_size)
        batch_precision = batch_precision / float(batch_size)
        batch_recall = batch_recall / float(batch_size)
        batch_f1 = batch_f1 / float(batch_size)
        
        outputs = {
            f'{split}_loss': batch_loss,
            f'{split}_acc': batch_acc,
            f'{split}_precision': batch_precision,
            f'{split}_recall': batch_recall,
            f'{split}_f1': batch_f1,
        }
        
        return outputs
    
    
    def get_features(self, inputs, input_lengths):
        with torch.no_grad():
            t = max(input_lengths)
            inputs = rnn_utils.pack_padded_sequence(
                inputs, 
                input_lengths, 
                batch_first=True,
                enforce_sorted=False,
            )
            outputs, self.hidden_state = self.lstm(inputs, self.hidden_state)
            h, _ = rnn_utils.pad_packed_sequence(
                outputs, 
                batch_first=True,
                padding_value=0.,
                total_length=t,
            )
            h = self.fc(h)
        return h

    
    def get_evaluation(self, outputs, targets):
        pred = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        targets = targets.cpu().numpy()
        acc = accuracy_score(targets, pred)
        precision = precision_score(targets, pred, zero_division=0.0)
        recall = recall_score(targets, pred)
        f1 = f1_score(targets, pred, zero_division=0.0)
        return acc, precision, recall, f1
    
    
    def reset_state(self):
        self.hidden_state = None
