import os
import torch
import random
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import sflib.sound.sigproc.spec_image as spec_image
from src.utils.utils import load_config
from src.models.encoder.cnn_ae import CNNAutoEncoder
from matplotlib.animation import FuncAnimation
from src.models.timing.model_baseline import BaselineSystem
from espnet2.bin.asr_parallel_transducer_inference import Speech2Text


# COMMAND
# /Users/user/desktop/授業/lab/code/ResponseTimingEstimator_demo
# python demo/realtime/demo.py


# PATH
## config
CONFIG_PATH = 'configs/timing/annotated_timing_baseline_mla_s1234.json'

## model
MODEL_PATH = 'exp/annotated/data_-500_2000/timing/baseline_cnnae/jeida_old/cleansnr5snr10snr20/cv0/best_val_loss_model.pth'

## out
OUTDIR = 'demo/streaming2/video'

## DEVICE [INPUT, OUTPUT]
sd.default.device = [2, 3]
print(sd.query_devices())


# METHOD
## seed initialize
def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

## asr decoding
def asr_streaming_decoding(data, is_final=False):
    speech = data.astype(np.float16)/32767.0
    hyps = speech2text.streaming_decode(speech=speech, is_final=is_final)
    if hyps[2] is None:
        results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[1]))
    else:
        results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[2]))
    if results is not None and len(results) > 0 and len(results[0]) > 0:
        text = results[0][0]
        token_int = results[0][2]
    else:
        text = ''
        token_int = [0]
    if token_int == []:
        token_int = [0]
    return text, token_int

## recoding
def callback(indata, frames, time, status):
    global wav_chunk
    wav_chunk = np.frombuffer(indata[::, 0], dtype=np.int16)

## inference and plot
def update_plot(frame):
    global i, t, wav_chunk, chunk_size, pre_text, text, pre_id, token_int, asr_buffer, vad_list, pred_list
    with torch.no_grad():
        # ASR
        asr_buffer = np.concatenate([asr_buffer, wav_chunk])
        if len(asr_buffer) >= asr_chunk_size:
            asr_chunk = asr_buffer[:asr_chunk_size]
            asr_buffer = asr_buffer[asr_chunk_size:]
            text, token_int = asr_streaming_decoding(asr_chunk)
        else:
            text = pre_text
            token_int = pre_id
        pre_text = text
        pre_id = token_int
        print(text)
        # spectrogram
        if i == 0:
            pad = np.zeros(3000, np.int16)
            # pad = np.zeros(800, np.int16)
            wav_chunk = np.concatenate([pad, wav_chunk])
        spec = generator.input_wave(wav_chunk)
        # CNNAE
        feat, _ = cnnae(wav_chunk, streaming=True, single=single)
        # Timing Estimator
        spec = torch.tensor(np.array(spec)).view(1, -1, 512, 10).float()
        feat = torch.tensor(feat.reshape(1, -1, 128))
        # if i == 0: input_lengths = [1]
        texts = [[text]]
        idxs = [[token_int]]
        input_lengths = []
        indices = []
        if single:
            input_lengths.append(1)
            indices.append(i)
        else:
            input_lengths.append(n_size)
        for j in range(n_size):
            indices.append(n_size*i+j)
        batch = [spec, feat, input_lengths, texts, idxs, indices, 'test']
        out, silence, vad_out = model.streaming_inference(batch, debug=True) 
        out = torch.sigmoid(out)
    i += 1
    pred_list.extend(*out.tolist())
    vad_list.extend(*vad_out.tolist())
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
## asr
speech2text = Speech2Text(
    asr_base_path='asr_espnet/egs2/atr/asr1',
    asr_train_config='asr_espnet/egs2/atr/asr1/exp/asr_train_asr_cbs_transducer_848_finetune_raw_jp_char_sp/config.yaml',
    asr_model_file='asr_espnet/egs2/atr/asr1/exp/asr_train_asr_cbs_transducer_848_finetune_raw_jp_char_sp/valid.loss_transducer.ave_10best.pth',
    token_type=None,
    bpemodel=None,
    beam_size=5,
    beam_search_config={"search_type": "maes"},
    lm_weight=0.0,
    nbest=1
)
speech2text.reset_inference_cache()
## cnnae
cnnae = CNNAutoEncoder(device='cpu')
## timing model
config = load_config(CONFIG_PATH)
device = torch.device('cpu')
model = BaselineSystem(config, device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
model.reset_state()

# values
# chunk_size = 800*2            # 100[ms]ごと
chunk_size = 800                # 50[ms]ごと
asr_chunk_size = 2048           # 128[ms]ごと
time_size = chunk_size // 16
n_size = time_size // 50
single = False
pred_list = [0]
vad_list = [0]
text = ''
pre_text = ''
pre_id = [0]
token_int = [0]
dtype = 'int16'
wav_chunk = np.array([], dtype=dtype)
asr_buffer = np.array([], dtype=dtype)
t = [i * time_size // n_size for i in range(500)]
i = 0

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
ax.set_xlim(0, 10000)
ax.set_ylim(0, 1)
ax.axhline(0.5, color='k', linestyle='dashed', lw=1)

# animation
ani = FuncAnimation(fig, update_plot, interval=1, blit=True)
with stream:
    plt.show()