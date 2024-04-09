import os
import torch
import random
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import sflib.sound.sigproc.spec_image as spec_image
from src.utils.utils import load_config
from matplotlib.animation import FuncAnimation
from src.models.encoder.cnn_ae import CNNAutoEncoder
from src.models.vad.model_vad2 import VoiceActivityDetactorCNNAE


# COMMAND
# /Users/user/desktop/授業/lab/code/ResponseTimingEstimator_demo
# python demo/silent/demo_system.py


# PATH
## config
CONFIG_PATH = 'configs/timing/annotated_timing_baseline_mla_s1234.json'
## model
MODEL_PATH = 'exp/annotated/data_-500_2000/vad/cnnae/best_val_loss_model.pth'
## audio
AUDIO_PATH = 'demo/tts/audio/sample1'


## DEVICE [INPUT, OUTPUT]
sd.default.device = [1, 2]
print(sd.query_devices())


# METHOD
## seed initialize
def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

## play audio
def play_audio(wav_path):
    sig, sr = sf.read(wav_path, always_2d=True)
    sd.play(sig, sr)


## recoding
def callback(indata, frames, time, status):
    print(1)
    global i, audio_num, wav_chunk, chunk_size, vad_list, vad_binary_list, active_flg
    wav_chunk = np.frombuffer(indata[::, 0], dtype=np.int16)
    with torch.no_grad():
        # spectrogram
        if i == 0:
            pad = np.zeros(3000, np.int16)
            wav_chunk = np.concatenate([pad, wav_chunk])
        # CNNAE
        feat, _ = cnnae(wav_chunk, streaming=True, single=single)
        feat = torch.tensor(feat.reshape(1, -1, 128))
        input_lengths = []
        if single: 
            input_lengths.append(1)
        else: 
            input_lengths.append(n_size)
        vad_out = model.streaming_inference(feat, input_lengths)
        vad_binary_out = (vad_out > vad_thres).int()
    i += 1
    vad_list.extend(*vad_out.tolist())
    vad_binary_list.extend(*vad_binary_out.tolist())
    print(vad_list[-1])
    
    if len(vad_binary_list) >= lim and sum(vad_binary_list[-lim:]) == 0:
        print(2)
        # 発話
        play_audio(os.path.join(AUDIO_PATH, audio_list[audio_num]))
        sd.wait()
        audio_num += 1
        print(audio_num, len(audio_list))
        if audio_num >= len(audio_list):
            sd.stop()
            active = False
        # 初期化
        i = 0
        vad_list = []
        vad_binary_list = []
        generator.reset()
        model.vad.reset_state()



# setting
## seed
seed_everything(42)
## aduio
audio_list = [audio for audio in os.listdir(AUDIO_PATH) if 'wav' in audio]
audio_list.sort()
## spectrogram
generator = spec_image.SpectrogramImageGenerator(
    framesize=800, 
    frameshift=160, 
    fftsize=1024, 
    image_width=10, 
    image_height=None, 
    image_shift=5
)
generator.reset()
## cnnae
cnnae = CNNAutoEncoder(device='cpu')
## timing model
config = load_config(CONFIG_PATH)
device = torch.device('cpu')
model = VoiceActivityDetactorCNNAE(config, device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
model.vad.reset_state()

# values
# chunk_size = 800*2            # 100[ms]ごと
chunk_size = 800                # 50[ms]ごと
time_size = chunk_size // 16
n_size = time_size // 50
single = False
vad_list = [0]
vad_binary_list = [0]
dtype = 'int16'
wav_chunk = np.array([], dtype=dtype)
max_x = 50000
t = [i * time_size // n_size for i in range(max_x // time_size + 1)]
i = 0
audio_num = 0

# silence limit
vad_thres = 0.5
lim = 14            # 700[ms]

# wav 
CHANNELS = 1                    # モノラル
RATE = 16000                    # サンプルレート（Hz）
DTYPE = 'int16'
stream = sd.InputStream(
        channels=CHANNELS,
        dtype=dtype,
        callback=callback,
        samplerate=RATE,
        blocksize=chunk_size,
)
stream.start()
active_flg = True
while stream.active and active_flg:
    pass
stream.stop()
stream.close()