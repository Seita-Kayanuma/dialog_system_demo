import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.vad_and_f0.vad_and_f0 import VAD_AND_F0

torch.autograd.set_detect_anomaly(True)


class VAD_AND_F0_Predictor(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.create_models()

    
    def create_models(self):
        vad_and_f0 = VAD_AND_F0(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.hidden_dim,
        )
        self.vad_and_f0 = vad_and_f0

    
    def configure_optimizer_parameters(self):

        parameters = chain(
            self.vad_and_f0.parameters(),
        )
        return parameters

    
    def forward(self, batch, split='train'):
        chs = batch[0]
        vad_labels = batch[1].to(self.device)
        specs = batch[5].to(self.device)
        input_lengths = batch[6]
        f0_labels = batch[9].to(self.device)
        batch_size = int(len(chs))
        self.vad_and_f0.reset_state()
        
        vad_loss, vad_acc, f0_loss = 0, 0, 0
        outputs_vad, outputs_f0 = self.vad_and_f0(specs, input_lengths)
        for i in range(batch_size):
            output_vad = outputs_vad[i]
            output_f0 = outputs_f0[i]
            loss_mask = (f0_labels[i] > 0).int()
            vad_loss = vad_loss + self.vad_and_f0.get_loss_vad(output_vad[:input_lengths[i]], vad_labels[i][:input_lengths[i]])
            vad_acc = vad_acc + self.vad_and_f0.get_acc(output_vad[:input_lengths[i]], vad_labels[i][:input_lengths[i]])
            f0_loss = f0_loss + self.vad_and_f0.get_loss_f0(output_f0[:input_lengths[i]]*loss_mask[:input_lengths[i]], f0_labels[i][:input_lengths[i]]) / torch.sum(loss_mask[:input_lengths[i]])
        vad_loss = vad_loss / float(batch_size)
        vad_acc = vad_acc / float(batch_size)
        f0_loss = f0_loss / float(batch_size)

        outputs = {
            f'{split}_vad_loss': vad_loss,
            f'{split}_vad_acc': vad_acc,
            f'{split}_f0_loss': f0_loss,
        }

        return outputs