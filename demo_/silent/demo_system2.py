import os
import torch
import queue
import random
import threading
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import sflib.sound.sigproc.spec_image as spec_image
from src.utils.utils import load_config
from src.models.encoder.cnn_ae import CNNAutoEncoder
from src.models.vad.model_vad2 import VoiceActivityDetactorCNNAE
from fjext.kana_kan_asr.asr import ASR
# from espnet2.bin.asr_inference import Speech2Text


# COMMAND
# /Users/user/desktop/授業/lab/code/ResponseTimingEstimator_demo
# python demo_/silent/old/demo_system2.py


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
    sd.wait()


## recoding
def callback(indata, frames, time, status):
    global wav_chunk
    wav_chunk.put(np.frombuffer(indata[::, 0], dtype=np.int16) / (2 ** 15 - 1))


## model inference
def run():
    global wav_chunk, audio_num
    vad_init = True
    vad_list = []
    vad_binary_list = []
    while True:
        data = wav_chunk.get()
        if data is None: continue
        with torch.no_grad():
            if vad_init:
                pad = np.zeros(1440, np.int16)
                data = np.concatenate([pad, data])
                vad_init = False
            feat, _ = cnnae(data, streaming=True, single=single)
            t, _ = feat.shape 
            feat = torch.tensor(feat.reshape(1, -1, 128))
            input_lengths = [t]
            vad_out = model.streaming_inference(feat, input_lengths)
            vad_binary_out = (vad_out > vad_thres).int()
        vad_list.extend(*vad_out.tolist())
        vad_binary_list.extend(*vad_binary_out.tolist())
        print(vad_list[-1])
        
        if len(vad_binary_list) >= lim and sum(vad_binary_list[-lim:]) == 0:
            print(2)
            # 発話
            play_audio(os.path.join(AUDIO_PATH, audio_list[audio_num]))
            audio_num += 1
            print(audio_num, len(audio_list))
            if audio_num >= len(audio_list):
                sd.stop()
                active = False
            # 初期化
            vad_init = True
            vad_list = []
            vad_binary_list = []
            wav_chunk = queue.Queue()
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
# asr fujie_model
asr = ASR.from_pretrained(
    espnet2_asr_model_tag="fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105",
    espnet2_asr_args=dict(
        streaming=True,
        lm_weight=0.0,
        beam_size=20,
        beam_search_config=dict(search_type="maes")
    ),
    kana_kanji_model_tag="fujie/kana_kanji_20240307"
)
# speech2text = Speech2Text.from_pretrained("fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105")
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
dtype = 'int16'
wav_chunk = queue.Queue()
asr_buffer = queue.Queue()
max_x = 50000
t = [i * time_size // n_size for i in range(max_x // time_size + 1)]
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
model_thread = threading.Thread(target=run)
model_thread.daemon = True
model_thread.start()
active_flg = True
while stream.active and active_flg:
    pass
stream.stop()
stream.close()