import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.f0.f0 import F0Spec, F0CNNAE

torch.autograd.set_detect_anomaly(True)


class F0_Predictor(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.create_models()

    
    def create_models(self):
        """
        f0 = F0Spec(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.hidden_dim,
        )
        """
        f0 = F0CNNAE(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.hidden_dim,
        )
        self.f0 = f0

    
    def configure_optimizer_parameters(self):

        parameters = chain(
            self.f0.parameters(),
        )
        return parameters

    
    def forward(self, batch, split='train'):
        specs = batch[0].to(self.device)
        feats = batch[1].to(self.device)
        f0_labels = batch[2].to(self.device)
        input_lengths = batch[3]
        batch_size = int(len(specs))
        
        f0_loss = 0
        # outputs_f0 = self.f0(specs, input_lengths) # Spec
        outputs_f0 = self.f0(feats, input_lengths) # CNN-AE
        for i in range(batch_size):
            output_f0 = outputs_f0[i]
            loss_mask = (f0_labels[i] > 0).int()
            f0_loss = f0_loss + self.f0.get_loss(output_f0[:input_lengths[i]]*loss_mask[:input_lengths[i]], f0_labels[i][:input_lengths[i]]) / torch.sum(loss_mask[:input_lengths[i]])
        f0_loss = f0_loss / float(batch_size)

        outputs = {
            f'{split}_f0_loss': f0_loss,
        }
        
        self.f0.reset_state()

        return outputs