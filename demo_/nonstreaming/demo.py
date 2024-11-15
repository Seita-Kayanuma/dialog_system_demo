import os
import sys
import wave
import time
import glob
import torch
import random
import pyaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sflib.sound.sigproc.spec_image as spec_image
from tqdm import tqdm
from src.utils.utils import load_config
from src.models.encoder.cnn_ae import CNNAutoEncoder
from moviepy.editor import VideoFileClip, AudioFileClip
from matplotlib.animation import FuncAnimation, ArtistAnimation
from src.models.timing.model_baseline import BaselineSystem

# command
# python demo_/nonstreaming/demo.py

# PATH
## CONFIG
CONFIG_PATH = 'configs/timing/annotated_timing_baseline_mla_s1234.json'

## MODEL
MODEL_PATH = 'exp/annotated/data_-500_2000/timing/spec_cnnae/aespec_vadcnnae/cv1/best_val_loss_model.pth'
# MODEL_PATH = 'exp/annotated/data_-500_2000/timing/yaguchinoise1e-4/cv1/best_val_loss_model.pth'

## OUTDIR
OUTDIR = 'demo/nonstreaming/video'

## WAV
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/original_silence.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/original.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/yaguchi_iphone.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/yaguchi_mac.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/original_mac.wav'
# WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/wav/20131101-1_01/20131101-1_01_001_ch0/yaguchinoise.wav'
WAV_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/ATR_Annotated/data_-500_2000/wav/20131125-6_01/20131125-6_01_001.wav'


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def token2idx(token, token_list, unk=1): 
    if token != token or token == '': return [0]
    token = token.replace('<eou>', '')
    idxs = [token_list.index(t) if t in token_list else unk for t in token]
    return idxs


def convert_frate(text, fr1=50, fr2=128):
        p50 = 0
        p128 = 0
        text50 = []
        text128=['']+text
        for i in range(len(text)*fr2//fr1):
            t = fr1*(i+1)
            p128 = t // fr2
            if len(text)-1<p128:
                text50.append(text[-1])
            else:
                text50.append(text[p128])
        return text50


def main():
    
    # seed
    seed_everything(42)
    

    # model
    config = load_config(CONFIG_PATH)
    device = torch.device('cpu')
    model = BaselineSystem(config, device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) #, strict=False)
    model.to(device)
    

    # wav
    wf = wave.open(WAV_PATH)
    wav = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
    
    
    # time range
    # chunk_size = 800*2         # 100[ms]ごと
    chunk_size = 800           # 50[ms]ごと
    asr_chunk_size = 2048      # 128[ms]ごと
    time_size = chunk_size // 16
    n_size = time_size // 50
    
    
    # data
    path = WAV_PATH.split('/')
    base_path = '/'.join(path[:-3])
    file_name = path[-2]
    turn_name = path[-1].replace('.wav', '')
    ## spectrogram(dataset)
    specs = np.load(os.path.join(base_path, 'spectrogram', file_name, turn_name+'_spectrogram.npy'))
    specs = torch.tensor(specs).view(1, -1, 512, 10).float()
    ## spectrogram(全て生成)
    """
    pad = np.zeros(int(16000*0.05), np.int16)
    x = np.concatenate([pad, wav, pad])
    generator = spec_image.SpectrogramImageGenerator()
    specs = generator.input_wave(x)
    specs = torch.tensor(np.vstack(specs)).view(1, -1, 512, 10).float()
    """
    ## spectrogram(streaming)
    """
    generator = spec_image.SpectrogramImageGenerator()
    specs = []
    for i in range(len(wav)//chunk_size):
        wav_mini_chunk = wav[chunk_size*i:chunk_size*(i+1)]
        if i == 0:
            pad = np.zeros(1440, np.int16)
            wav_mini_chunk = np.concatenate([pad, wav_mini_chunk])
        spec = generator.input_wave(wav_mini_chunk)
        specs.append(spec)
    specs = torch.tensor(np.vstack(specs)).view(1, -1, 512, 10).float()
    """
    ## feat
    feats = np.load(os.path.join(base_path, 'cnn_ae', file_name, turn_name+'_spec.npy'))
    feats = torch.tensor(feats).view(1, -1, 128)
    ## text
    df_text = pd.read_csv(os.path.join(base_path, 'texts/cbs-t_mla_848', file_name, turn_name+'.csv'))
    df_text[pd.isna(df_text['asr_recog'])] = ''
    texts = df_text['asr_recog'].tolist()
    texts = [txt.replace('。', '')+'' for txt in texts]
    texts = convert_frate(texts)
    n = 100 // 50 # self.asr_delay//self.frame_length
    if n >= 0:
        texts = ['']*n+texts
    else:
        texts = texts[abs(n):]+[texts[-1]]*abs(n)
    ## idx
    token_path = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/tokens/char/tokens2.txt'
    with open(token_path) as f:
            lines = f.readlines()
    token_list = [line.split()[0] for line in lines]
    idxs = [token2idx(token=t, token_list=token_list) for t in texts]
    ## input_lengths
    input_lengths = torch.tensor([specs.shape[1]])
    ## indice
    indices = [None]
    # indices = [i for i in range(specs.shape[1])]
    ## batch
    batch = [specs, feats, input_lengths, [texts], [idxs], indices, 'test']
    
    
    # eou
    df_csv = pd.read_csv(os.path.join(base_path, 'csv', file_name+'.csv'))
    df_wav = pd.read_csv(os.path.join(base_path, 'wav_start_end', file_name+'.csv'))
    number = int(turn_name.split('_')[-1]) - 1
    eou = df_csv.iloc[number]['end'] - df_wav.iloc[number]['wav_start']
    
    
    # timing estimation
    ## reset model
    model.eval()
    model.reset_state()

    
    ## model
    with torch.no_grad():
        # nonstreaming
        out, silence, vad_list = model.nonstreaming_inference(batch, debug=True)
        vad_list = vad_list[0].tolist() 
        pred_list = torch.sigmoid(out[0]).tolist()
        # streaming
        """
        vad_list = []
        pred_list = []
        i = 0
        for spec, feat, text, idx in zip(specs[0], feats[0], texts, idxs):
            out, silence, vad_out = model.streaming_inference([spec.reshape(1, -1, 512, 10), feat.reshape(1, -1, 128), [1], [[text]], [[idx]], [i], 'test'], debug=True)
            out = torch.sigmoid(out)
            vad_list.extend(*vad_out.tolist())
            pred_list.extend(*out.tolist())
            i += 1
        """

    # make movie
    pred_list.insert(0, 0)
    vad_list.insert(0, 0)
    t1 = [i * time_size // n_size for i in range(len(pred_list))]
    t2 = [(i / chunk_size) * time_size for i in range(len(wav))]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1
    ax1.set_xlim(0, max(t1))
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color='k', linestyle='dashed', lw=1)
    # ax2
    ax2.set_xlim(0, max(t1))
    
    # frame
    graphs = []
    timing_value = None
    # t = 0
    # graph = ax1.plot(0, 0, color='b')
    # graphs.append(graph)
    # t > 0
    for i in range(0, len(t1)):
        x1 = t1[:i+1]
        y1 = pred_list[:i+1]
        y_vad = vad_list[:i+1]
        graph = ax1.plot(x1, y1, color='b')
        graph += ax1.plot(x1, y_vad, color='#ff7f00')
        if y1[-1] > 0.5 and timing_value == None:
            timing_value = x1[-1]
        if timing_value != None:
            graph += ax1.plot([timing_value, timing_value], [0, 1], color='r')
            if eou != -1:
                graph += [ax1.text(timing_value+30, 0.03, str(timing_value - eou), fontsize='large', color='k')]
        if i * time_size >= eou and eou != -1:
            graph += ax1.plot([eou, eou], [0, 1], color='y')
        x2 = t2[:(chunk_size//n_size)*i]
        y2 = wav[:(chunk_size//n_size)*i]
        graph += ax2.plot(x2, y2, color='c')
        graphs.append(graph)
    # animation
    ani = ArtistAnimation(fig, graphs, interval=time_size//n_size)
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