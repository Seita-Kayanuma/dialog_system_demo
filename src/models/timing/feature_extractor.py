import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

#from src.models.encoder.transformer_encoder import TransformerEncoder
from src.models.encoder.transformer_encoder_mytokenizer import TransformerEncoder
from src.models.encoder.acoustic_encoder import AcousticEncoderSpec
from src.models.encoder.acoustic_encoder2 import AcousticEncoderCNNAE, AcousticEncoderCNNAE2
from src.models.encoder.timing_encoder import TimingEncoder
from src.models.encoder.timing_encoder2 import TimingEncoder2
from src.models.vad.vad import VADSpec
from src.models.vad.vad2 import VADCNNAE
from src.models.f0.model import F0CNNAE, F0Spec
from src.models.vad_and_f0.vad_and_f0 import VAD_AND_F0

torch.autograd.set_detect_anomaly(True)


class FeatureExtractor(nn.Module):

    def __init__(self, config, device, is_use_silence=True, is_use_n_word=False):
        super().__init__()
        
        self.device = device
        self.config = config
        self.is_use_silence = is_use_silence
        self.is_use_n_word = is_use_n_word
        
        self.create_models(config)

    
    def create_models(self, config):
        
        """
        ae = AcousticEncoderSpec(
            self.device,
            self.config.model_params.acoustic_input_dim,
            self.config.model_params.acoustic_hidden_dim,
            self.config.model_params.acoustic_encoding_dim,
        )
        """
        ae = AcousticEncoderCNNAE(
            self.device,
            self.config.model_params.acoustic_input_dim,
            self.config.model_params.acoustic_hidden_dim,
            self.config.model_params.acoustic_encoding_dim,
        )
        self.acoustic_encoder = ae
        te = TimingEncoder(
            self.config,
            self.device,
            self.config.model_params.timing_input_dim,
            self.config.model_params.timing_encoding_dim,
            is_use_silence=self.is_use_silence,
            is_use_n_word=self.is_use_n_word,
        )
        self.timing_encoder = te
        se = TransformerEncoder(
            self.config,
            self.device,
        )
        self.semantic_encoder = se
        """
        f0e = F0CNNAE(
            self.config,
            self.device
        )
        """
        """
        f0e = F0Spec(
            self.config,
            self.device
        )
        """
        # self.f0_encoder = f0e
        # self.f0_encoder.load_state_dict(torch.load(config.model_params.f0_model_path))

        
    def get_params(self):

        parameters = chain(
            self.acoustic_encoder.parameters(),
            self.timing_encoder.linear.parameters(),
            # self.timing_encoder2.linear.parameters(),
            self.semantic_encoder.parameters(),
        )
        return parameters

    
    def forward(self, specs, feats, idxs, input_lengths, texts, indices, split):
        
        # r_a = self.acoustic_encoder(specs, input_lengths) # Spec
        r_a = self.acoustic_encoder(feats, input_lengths) # CNNAE
        # r_a = self.acoustic_encoder.get_features(feats, input_lengths) # CNNAE
        # r_t = self.timing_encoder(specs, idxs, input_lengths, indices, split) # Spec
        r_t = self.timing_encoder(feats, idxs, input_lengths, indices, split) # CNNAE
        # r_t, _ = self.timing_encoder2(specs, idxs, input_lengths, indices, split)
        # r_t, r_f0 = self.timing_encoder2(specs, idxs, input_lengths, indices, split)
        r_s = self.semantic_encoder(idxs, input_lengths)
        # r_vad = self.vad.get_features(specs, input_lengths)
        # r_f0, f0_value = self.f0_encoder.inference(feats, input_lengths)
        # r_f0, f0_value = self.f0_encoder.inference(specs, input_lengths)
        
        # embs = r_a
        # embs = torch.cat([r_s, r_t], dim=-1)
        # embs = torch.cat([r_a, r_f0, r_vad, r_s], dim=-1)
        embs = torch.cat([r_s, r_a, r_t], dim=-1)
        # embs = torch.cat([r_s, r_t, r_f0], dim=-1)
        # embs = torch.cat([r_s, r_t, feats], dim=-1)
        # embs = torch.cat([r_s, r_a, r_t, r_vad_and_f0], dim=-1)
     
        return embs
    
    
    def streaming_inference(self, specs, feats, idxs, input_lengths, texts, indices, split, debug=False):
        # r_a = self.acoustic_encoder(specs, input_lengths) # Spec
        r_a = self.acoustic_encoder(feats, input_lengths) # CNNAE
        # r_a = self.acoustic_encoder.get_features(feats, input_lengths) # CNNAE
        # r_t = self.timing_encoder.streaming_inference(specs, idxs, input_lengths, indices, split, debug) # Spec
        r_t = self.timing_encoder.streaming_inference(feats, idxs, input_lengths, indices, split, debug) # CNNAE
        # r_t = self.timing_encoder(feats, idxs, input_lengths, indices, split, debug)
        # r_s = self.semantic_encoder(texts)
        r_s = self.semantic_encoder(idxs, input_lengths)
        # r_f0, f0_value = self.f0_encoder.inference(feats, input_lengths)
        # r_f0, f0_value = self.f0_encoder.inference(specs, input_lengths)
        
        if debug: r_t, silence, vad_preds = r_t
        # embs = torch.cat([r_a, r_t], dim=-1)
        embs = torch.cat([r_s, r_a, r_t], dim=-1)
        # embs = torch.cat([r_s, r_t, r_f0], dim=-1)
        # embs = torch.cat([r_s, r_t, feats], dim=-1)
        if debug: return embs, silence, vad_preds
        return embs
    
    
    def nonstreaming_inference(self, specs, feats, idxs, input_lengths, texts, indices, split, debug=False):
        # r_a = self.acoustic_encoder(specs, input_lengths) # Spec
        r_a = self.acoustic_encoder(feats, input_lengths) # CNNAE
        # r_t = self.timing_encoder.nonstreaming_inference(specs, idxs, input_lengths, indices, split, debug) # Spec
        r_t = self.timing_encoder(feats, idxs, input_lengths, indices, split, debug) # CNNAE
        # r_s = self.semantic_encoder(texts)
        r_s = self.semantic_encoder(idxs, input_lengths)
        # r_f0, f0_value = self.f0_encoder.inference(feats, input_lengths)
        if debug: r_t, silence, vad_preds = r_t
        # embs = torch.cat([r_s, r_t], dim=-1)
        # embs = torch.cat([r_s, r_a, r_t], dim=-1)
        # embs = torch.cat([r_s, r_t, r_f0], dim=-1)
        embs = torch.cat([r_s, r_t, f0_value], dim=-1)
        if debug: return embs, silence, vad_preds
        return embs
    
    
    def reset_state(self):
        self.acoustic_encoder.reset_state()
        self.timing_encoder.reset_state()
        # self.timing_encoder2.reset_state()
        # self.f0_encoder.reset_state()
