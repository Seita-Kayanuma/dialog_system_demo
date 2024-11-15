import os
import torch
import random
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import sflib.sound.sigproc.spec_image as spec_image
from src.utils.utils import load_config
from matplotlib.animation import FuncAnimation
from src.models.encoder.cnn_ae import CNNAutoEncoder
from src.models.vad.model_vad2 import VoiceActivityDetactorCNNAE


# COMMAND
# /Users/user/desktop/授業/lab/code/ResponseTimingEstimator_demo
# python demo_/silent/old/demo.py


# PATH
## config
CONFIG_PATH = 'configs/timing/annotated_timing_baseline_mla_s1234.json'
## model
MODEL_PATH = 'exp/annotated/data_-500_2000/vad/cnnae/best_val_loss_model.pth'
## out
OUTDIR = 'demo/streaming2/video'


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

## recoding
def callback(indata, frames, time, status):
    global wav_chunk
    wav_chunk = np.frombuffer(indata[::, 0], dtype=np.int16)

## inference and plot
def update_plot(frame):
    global i, t, wav_chunk, chunk_size, vad_list, pred_list, utter_flg
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
    if (len(vad_binary_list) >= lim and sum(vad_binary_list[-lim:]) == 0) or utter_flg:
        pred_list.extend([1 for _ in range(n_size)])
        utter_flg = True
    else:
        pred_list.extend([0 for _ in range(n_size)])
    line_vad.set_data(t[:len(vad_list)], vad_list)
    line_pred.set_data(t[:len(pred_list)], pred_list)
    return line_vad, line_pred



# setting
## seed
seed_everything(42)
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
pred_list = [0]
vad_list = [0]
vad_binary_list = [0]
dtype = 'int16'
wav_chunk = np.array([], dtype=dtype)
max_x = 50000
t = [i * time_size // n_size for i in range(max_x // time_size + 1)]
i = 0

# silence limit
vad_thres = 0.5
lim = 100
utter_flg = False

# wav 
CHANNELS = 1                    # モノラル
RATE = 16000                    # サンプルレート（Hz）
DTYPE = 'int16'
stream = sd.InputStream(
        channels=CHANNELS,
        dtype=dtype,
        callback=callback,
        samplerate=RATE,
        blocksize=chunk_size
)

# matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
line_vad, = ax.plot([], [], color='#ff7f00')
line_pred, = ax.plot([], [], color='b')
ax.set_xlim(0, max_x)
ax.set_ylim(0, 1)
ax.axhline(0.5, color='k', linestyle='dashed', lw=1)

# animation
ani = FuncAnimation(fig, update_plot, interval=1, blit=True)
with stream:
    plt.show()