import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from src.utils.utils import L2Norm, Normalize


torch.autograd.set_detect_anomaly(True)


class F0CNNAE(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        
        self.device = device
        self.lstm = torch.nn.LSTM(
            input_size=config.model_params.f0_input_dim, 
            hidden_size=config.model_params.f0_hidden_dim, 
            batch_first=True
        )
        self.fc1 = nn.Linear(config.model_params.f0_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.criterion = nn.MSELoss(reduction='sum').to(device)
        self.reset_state()

    
    def forward(self, batch, split='train'):
        specs = batch[0].to(self.device)
        feats = batch[1].to(self.device)
        f0_labels = batch[2].to(self.device)
        input_lengths = batch[3]
        batch_size = int(len(feats))
        
        b, t, h = feats.shape
        t = max(input_lengths)
        inputs = rnn_utils.pack_padded_sequence(
            feats, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )
        h, self.hidden_state = self.lstm(inputs, self.hidden_state)
        h, _ = rnn_utils.pad_packed_sequence(
            h, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )
        outputs_f0 = self.fc2(self.relu(self.fc1(h)))
        outputs_f0 = outputs_f0.reshape(b, -1)
        
        f0_loss = 0
        for i in range(batch_size):
            output_f0 = outputs_f0[i]
            loss_mask = (f0_labels[i] > 0).int()
            f0_loss = f0_loss + self.criterion(output_f0[:input_lengths[i]]*loss_mask[:input_lengths[i]], f0_labels[i][:input_lengths[i]]) / torch.sum(loss_mask[:input_lengths[i]])
        f0_loss = f0_loss / float(batch_size)
        
        outputs = {
            f'{split}_f0_loss': f0_loss,
        }
        
        self.reset_state()
        
        return outputs
    
    
    def inference(self, feats, input_lengths):
        with torch.no_grad():
            b, n, h = feats.shape
            t = max(input_lengths)
            inputs = rnn_utils.pack_padded_sequence(
                feats, 
                input_lengths, 
                batch_first=True,
                enforce_sorted=False,
            )
            h, self.hidden_state = self.lstm(inputs, self.hidden_state)
            h, _ = rnn_utils.pad_packed_sequence(
                h, 
                batch_first=True,
                padding_value=0.,
                total_length=t,
            )
            h = self.fc1(h)
            f0 = self.fc2(self.relu(h))
            h = h.view(b, n, -1)
            f0 = f0.view(b, n, -1)
        return h, f0
    
    
    def reset_state(self):
        self.hidden_state = None



class F0Spec(nn.Module):

    def __init__(self, config, device):
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
        self.lstm = torch.nn.LSTM(input_size=config.model_params.f0_input_dim, hidden_size=config.model_params.f0_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(config.model_params.f0_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.criterion = nn.MSELoss(reduction='sum').to(device)
        self.reset_state()

    
    def forward(self, batch, split='train'):
        specs = batch[0].to(self.device)
        feats = batch[1].to(self.device)
        f0_labels = batch[2].to(self.device)
        input_lengths = batch[3]
        batch_size = int(len(specs))
        t = max(input_lengths)
        b, n, h, w = specs.shape
        inputs = specs.reshape(b*n, 1, h, w)
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
        h, self.hidden_state = self.lstm(inputs, self.hidden_state)
        h, _ = rnn_utils.pad_packed_sequence(
            h, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )
        outputs_f0 = self.fc2(self.relu(self.fc1(h)))
        outputs_f0 = outputs_f0.reshape(b, -1)
        
        f0_loss = 0
        for i in range(batch_size):
            output_f0 = outputs_f0[i]
            loss_mask = (f0_labels[i] > 0).int()
            f0_loss = f0_loss + self.criterion(output_f0[:input_lengths[i]]*loss_mask[:input_lengths[i]], f0_labels[i][:input_lengths[i]]) / torch.sum(loss_mask[:input_lengths[i]])
        f0_loss = f0_loss / float(batch_size)
        
        outputs = {
            f'{split}_f0_loss': f0_loss,
        }
        
        self.reset_state()
        
        return outputs
    
    
    def inference(self, specs, input_lengths):
        with torch.no_grad():
            batch_size = int(len(specs))
            t = max(input_lengths)
            b, n, h, w = specs.shape
            inputs = specs.reshape(b*n, 1, h, w)
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
            h, self.hidden_state = self.lstm(inputs, self.hidden_state)
            h, _ = rnn_utils.pad_packed_sequence(
                h, 
                batch_first=True,
                padding_value=0.,
                total_length=t,
            )
            h = self.fc1(h)
            f0 = self.fc2(self.relu(h))
            h = h.view(b, n, -1)
            f0 = f0.view(b, n, -1)
        return h, f0
    
    
    def reset_state(self):
        self.hidden_state = None