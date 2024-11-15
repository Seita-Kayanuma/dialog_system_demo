import os
import sys
import wave
import time
import glob
import torch
import random
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import sflib.sound.sigproc.spec_image as spec_image
from tqdm import tqdm
from src.utils.utils import load_config
from moviepy.editor import VideoFileClip, AudioFileClip
from matplotlib.animation import FuncAnimation, ArtistAnimation
from src.models.vad.model_vad import VoiceActivityDetactor
from espnet2.bin.asr_parallel_transducer_inference import Speech2Text

# command
# python demo/demo.py

# PATH
## CONFIG
CONFIG_PATH = 'configs/vad/annotated_vad.json'

## MODEL
# MODEL_PATH = 'exp/annotated/data_-500_2000/vad/best_val_loss_model.pth'
MODEL_PATH = 'exp/annotated/data_-500_2000/vad/revised/best_val_loss_model.pth'

## WAV
WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/original.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/original2.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/yaguchi_iphone.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/yaguchi_mac.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/original_mac.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/yaguchinoise.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/dataset/20131115-2_01_010.wav'


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    
    # seed
    seed_everything(42)
    
    # setting
    ## spectrogram
    generator = spec_image.SpectrogramImageGenerator(framesize=800, frameshift=160, fftsize=1024, image_width=10, image_height=None, image_shift=5)
    
    # vad model
    config = load_config(CONFIG_PATH)
    device = torch.device('cpu')
    model = VoiceActivityDetactor(config, device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) #, strict=False)
    model.to(device)
    
    
    # wav
    wf = wave.open(WAV_PATH)
    wav = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
    
    
    # wav eou data
    wavdata_path = os.path.join('/'.join(WAV_PATH.split('/')[:-1]), 'data.txt')
    eou = -1
    with open(wavdata_path) as f:
        lines = f.readlines()
    for line in lines:
        name, name_eou = line.split(' ')
        if WAV_PATH.split('/')[-1] == name:
            eou = int(name_eou)
    
    
    ## init
    chunk_size = 800*2         # 100[ms]ごと
    asr_chunk_size = 2048      # 128[ms]ごと
    pred_list = []
    text_list = []
    silence_list = []
    vad_list = []
    text = ''
    pre_text = ''
    pre_id = [0]
    asr_buffer = np.array([], dtype='int16')
    
    ## model
    with torch.no_grad():
        
        for i in range(len(wav)//chunk_size):
            
            # wavの範囲
            wav_chunk = wav[chunk_size*i:chunk_size*(i+1)]
            
            # spectrogram
            if i == 0:
                pad = np.zeros(800, np.int16)
                wav_chunk = np.concatenate([pad, wav_chunk])
            result = generator.input_wave(wav_chunk)

            # Timing Estimator
            feat = torch.tensor(result).unsqueeze(0).float()
            input_lengths = [1]
            texts = [[text]]
            idxs = [[token_int]]
            indices = [None]

            batch = [feat, input_lengths, texts, idxs, indices, 'test']
            out, silence, vad_out = model.streaming_inference(batch, debug=True) 
            out = torch.sigmoid(out).item()

            pred_list.append(out)
            text_list.append(text)
            silence_list.append(silence)
            vad_list.append(vad_out.item())
    
        
    # make movie
    pred_list.insert(0, 0)
    t1 = [i * 100 for i in range(len(pred_list))]
    t2 = [(i / chunk_size) * 100 for i in range(len(wav))]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1
    ax1.set_xlim(0, max(t1))
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color='k', linestyle='dashed', lw=1)
    # ax2
    ax2.set_xlim(0, max(t1))
    
    # frame
    graphs = []
    timinglines = []
    timing_value = None
    # t = 0
    # graph = ax1.plot(0, 0, color='b')
    # graphs.append(graph)
    # t > 0
    for i in range(len(t1)):
        x1 = t1[:i+1]
        y1 = pred_list[:i+1]
        graph = ax1.plot(x1, y1, color='b')
        if y1[-1] > 0.5 and timing_value == None:
            timing_value = x1[-1]
        if timing_value != None:
            graph += ax1.plot([timing_value, timing_value], [0, 1], color='r')
            if eou != -1:
                graph += [ax1.text(timing_value+30, 0.03, str(timing_value - eou), fontsize='large', color='k')]
        if i * 100 >= eou and eou != -1:
            graph += ax1.plot([eou, eou], [0, 1], color='y')
        x2 = t2[:chunk_size*i]
        y2 = wav[:chunk_size*i]
        graph += ax2.plot(x2, y2, color='c')
        graphs.append(graph)
    # animation
    ani = ArtistAnimation(fig, graphs, interval=100)
    # save 
    ## video
    model_name = MODEL_PATH.split('/')[-3]
    wav_name = WAV_PATH.split('/')[-1].replace('.wav', '')
    out_path = os.path.join(OUTDIR, model_name)
    os.makedirs(out_path, exist_ok=True)
    ani.save(os.path.join(out_path, f'{wav_name}.mp4'), writer='ffmpeg')
    ## last image
    out_image_path = os.path.join('/'.join(OUTDIR.split('/')[:-1]), 'image', model_name)
    os.makedirs(out_image_path, exist_ok=True)
    plt.savefig(os.path.join(out_image_path, wav_name.replace('wav', 'jpg')))
    plt.show()
    
    
    # add sound
    video_clip = VideoFileClip(os.path.join(out_path, f'{wav_name}.mp4'))
    audio_clip = AudioFileClip(WAV_PATH)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(os.path.join(out_path, f'{wav_name}_full.mp4'), codec="libx264", audio_codec="aac", temp_audiofile='temp-audio.m4a', remove_temp=True)


if __name__ == '__main__':
    main()