import torch
import datetime
import threading
import numpy as np
import sflib.sound.sigproc.spec_image as spec_image
from audio import AudioData
from utils import load_config
from src.models.encoder.cnn_ae import CNNAutoEncoder
from src.models.timing.model_baseline import BaselineSystem


# fujie_asr
TOKENPATH = 'data/tokens/char/tokens_fujie2.txt'
MODELPATH = 'exp/annotated/data_-500_2000/timing/baseline_cnnae/fjext2_cleansnr10snr20/cv0/best_val_loss_model.pth'
CONFIGPATH = 'configs/timing/annotated_timing_baseline_fjext.json'
# sakuma_asr
# TOKENPATH = 'data/tokens/char/tokens2.txt'
# MODELPATH = 'exp/annotated/data_-500_2000/timing/baseline_cnnae/jeida_old/cleansnr5snr10snr20/cv0/best_val_loss_model.pth'
# CONFIGPATH = 'configs/timing/annotated_timing_baseline_mla_s1234.json'


class TimingWorker:
    def __init__(self,
                 device,
                 frame_num = 5,
                 multi_process_num = 2
        ):
        self.frame_num = frame_num
        self.multi_process_num = multi_process_num
        self.device = device
        self.cond = threading.Condition()
        # self.device = torch.device("mps")
        self.create_token_list(TOKENPATH)
        self.create_models()
        self.timing_flg = False     # タイミング推定結果
        self.after_reset = True     # reset後か
        self.buffer_for_timing = []
        self.current_text = ''
        self.current_idxs = [0]
        self.reset()
    
    def create_models(self):
        self.generator = spec_image.SpectrogramImageGenerator(
            framesize=800,
            frameshift=160,
            fftsize=1024,
            image_width=10,
            image_height=None,
            image_shift=5
        )
        self.cnnae = CNNAutoEncoder(device="cpu")
        config = load_config(CONFIGPATH)
        self.timing_model = BaselineSystem(config, self.device)
        self.timing_model.load_state_dict(torch.load(MODELPATH, map_location=self.device))
        self.timing_model.to(self.device)
        self.timing_model.eval()
    
    def create_token_list(self, path):
        with open(path) as f:
            lines = f.readlines()
        self.tokens = [line.split()[0] for line in lines]
    
    def token2idx(self, token, unk=1):
        if token != token or token == '':
            return [0]
        token = token.replace('<eou>', '')
        token = token.split(' ')
        idxs = [self.tokens.index(t) if t in self.tokens else unk for t in token]
        return idxs
    
    def idx2token(self, idxs): 
        token = [self.tokens[idx] for idx in idxs]
        return token
        
    def inference(self, audio: float):
        if self.after_reset:
            pad = np.zeros(1600, np.int16)
            wav_chunk = np.concatenate([pad, audio])
            self.after_reset = False
        else:
            wav_chunk = audio
        spec = self.generator.input_wave(wav_chunk)
        feat, _ = self.cnnae(wav_chunk, streaming=True, single=False)
        spec = torch.tensor(np.array(spec)).view(1, -1, 512, 10).float()
        feat = torch.tensor(feat.reshape(1, -1, 128))
        input_length = self.multi_process_num
        indices = []
        for _ in range(self.multi_process_num):
            self.indices += 1
            indices.append(self.indices)
        batch = [spec, feat, [input_length], [[self.current_text]], [[self.current_idxs]], indices, 'test']
        out, _, _ = self.timing_model.streaming_inference(batch, debug=True)
        confidence = torch.sigmoid(out)
        self.timing_flg = (confidence > 0.5).detach().cpu().numpy()[0][-1]
        print(datetime.datetime.now(), '[TimingWorker inference]', self.timing_flg, f'text: {self.current_text}')
        # print(datetime.datetime.now(), '[TimingWorker inference]', len(self.buffer_for_timing), confidence, self.timing_flg, f'text: {self.current_text}')
    
    def process(self, audio: AudioData):
        self.buffer_for_timing.append(audio.data_np)
        if len(self.buffer_for_timing) >= self.frame_num * self.multi_process_num:
            audio_int16 = np.concatenate(self.buffer_for_timing[:self.frame_num * self.multi_process_num]).astype(np.int16)
            self.inference(audio_int16)
            self.buffer_for_timing = self.buffer_for_timing[self.frame_num * self.multi_process_num:]
    
    def get_data(self, audio: AudioData):
        # print('[TimingWorker get_data]', datetime.datetime.now(), len(self.buffer_for_timing))
        with self.cond:
            self.buffer_for_timing.append(audio.data_np)
            self.cond.notify_all()
    
    def run(self):
        while True:
            # print('[TimingWorker run]', datetime.datetime.now(), len(self.buffer_for_timing))
            with self.cond:
                while len(self.buffer_for_timing) < self.frame_num * self.multi_process_num:
                    self.cond.wait()
                audio_int16 = np.concatenate(self.buffer_for_timing[:self.frame_num * self.multi_process_num]).astype(np.int16)
            self.inference(audio_int16)
            self.buffer_for_timing = self.buffer_for_timing[self.frame_num * self.multi_process_num:]
        
    def reset(self):
        print(datetime.datetime.now(), '[TimingWorker reset]')
        self.buffer_for_timing.clear()
        self.generator.reset()
        self.timing_model.reset_state()
        self.timing_flg = False
        self.after_reset = True
        self.indices = 0