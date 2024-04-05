import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from src.models.vad.vad2 import VADCNNAE

torch.autograd.set_detect_anomaly(True)


class VoiceActivityDetactorCNNAE(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.create_models()

    
    def create_models(self):
        vad = VADCNNAE(
            self.device,
            self.config.model_params.vad_input_dim,
            self.config.model_params.vad_hidden_dim,
        )
        self.vad = vad

    
    def configure_optimizer_parameters(self):

        parameters = chain(
            self.vad.parameters(),
        )
        return parameters

    
    def forward(self, batch, split='train'):
        """     
        vad_labels = batch[1].to(self.device)
        specs = batch[5].to(self.device)
        input_lengths = batch[6]
        """
        feats = batch[1].to(self.device)
        vad_labels = batch[2].to(self.device)
        input_lengths = batch[3]
        batch_size = int(len(vad_labels))
        self.vad.reset_state()
        
        vad_loss, vad_acc, vad_precision, vad_recall, vad_f1 = 0, 0, 0, 0, 0
        outputs_vad = self.vad(feats, input_lengths)
        for i in range(batch_size):
            output_vad = outputs_vad[i]
            vad_loss = vad_loss + self.vad.get_loss(output_vad[:input_lengths[i]], vad_labels[i][:input_lengths[i]])
            acc, precision, recall, f1 = self.vad.get_evaluation(output_vad[:input_lengths[i]], vad_labels[i][:input_lengths[i]])
            vad_acc = vad_acc + acc
            vad_precision = vad_precision + precision
            vad_recall = vad_recall + recall
            vad_f1 = vad_f1 + f1
        vad_loss = vad_loss / float(batch_size)
        vad_acc = vad_acc / float(batch_size)
        vad_precision = vad_precision / float(batch_size)
        vad_recall = vad_recall / float(batch_size)
        vad_f1 = vad_f1 / float(batch_size)

        outputs = {
            f'{split}_loss': vad_loss,
            f'{split}_acc': vad_acc,
            f'{split}_precision': vad_precision,
            f'{split}_recall': vad_recall,
            f'{split}_f1': vad_f1,
        }
        return outputs
    
    def inference(self, batch, split='train'):
        feats = batch[0].to(self.device)
        input_lengths = batch[1]
        batch_size = int(len(feats))
        self.vad.reset_state()
        outputs = torch.sigmoid(self.vad(feats, input_lengths))
        return outputs
    
    def streaming_inference(self, feats, input_lengths):
        with torch.no_grad():
            vad_preds = torch.sigmoid(self.vad(feats, input_lengths))
        return vad_preds