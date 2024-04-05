import os
import wave
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sflib.sound.sigproc.spec_image as spec_image
from src.utils.utils import load_config
from src.models.encoder.cnn_ae import CNNAutoEncoder
from moviepy.editor import VideoFileClip, AudioFileClip
from matplotlib.animation import ArtistAnimation
from src.models.timing.model_baseline import BaselineSystem
from espnet2.bin.asr_parallel_transducer_inference import Speech2Text


# command
# /Users/user/desktop/授業/lab/code/ResponseTimingEstimator_demo
# python demo/streaming/demo.py


# PATH
## CONFIG
CONFIG_PATH = 'configs/timing/annotated_timing_baseline_mla_s1234.json'

## MODEL
# MODEL_PATH = 'exp/annotated/data_-500_2000/timing/spec_cnnae/aef0cnnaeatr128_vadcnnae/cv1/best_val_loss_model.pth'
# MODEL_PATH = 'exp/annotated/data_-500_2000/timing/baseline_cnnae/clean/cv1/best_val_loss_model.pth'
MODEL_PATH = 'exp/annotated/data_-500_2000/timing/baseline_cnnae/jeida_old/cleansnr5snr10snr20/cv0/best_val_loss_model.pth'

## OUTDIR
OUTDIR = 'demo/streaming/video'

## WAV
# WAV_PATH = 'wav/20131101-1_01/20131101-1_01_001_ch0/original_silence.wav'
# WAV_PATH = 'wav/20131101-1_01/20131101-1_01_001_ch0/original.wav'
# WAV_PATH = 'wav/20131101-1_01/20131101-1_01_001_ch0/yaguchi_mac.wav'
# WAV_PATH = 'wav/20131101-1_01/20131101-1_01_001_ch0/original_mac.wav'
# WAV_PATH = 'wav/20131101-1_01/20131101-1_01_001_ch0/yaguchinoise.wav'
# WAV_PATH = 'wav/20131125-6_01/20131125-6_01_001.wav'
WAV_PATH = 'wav/yaguchi_iphone/yaguchi_iphone3.wav'


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
    generator = spec_image.SpectrogramImageGenerator(
        framesize=800, 
        frameshift=160, 
        fftsize=1024, 
        image_width=10, 
        image_height=None, 
        image_shift=5
    )
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
        nbest=1,
        #device = "cuda:0", # "cpu",
        # device = "cpu",
    )
    # asr decoding
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
    # cnnae
    cnnae = CNNAutoEncoder(device='cpu')
    
    
    # model
    config = load_config(CONFIG_PATH)
    device = torch.device('cpu')
    model = BaselineSystem(config, device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    

    # wav
    wf = wave.open(WAV_PATH)
    wav = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
    
    
    # wav eou data
    eou = -1
    wavdata_path = os.path.join('/'.join(WAV_PATH.split('/')[:-1]), 'data.txt')
    path = WAV_PATH.split('/')
    base_path = '/'.join(path[:-3])
    file_name = path[-2]
    turn_name = path[-1].replace('.wav', '')
    if os.path.isfile(wavdata_path):
        with open(wavdata_path) as f:
            lines = f.readlines()
        for line in lines:
            name, name_eou = line.split(' ')
            if WAV_PATH.split('/')[-1] == name:
                eou = int(name_eou)
    elif os.path.isfile(os.path.join(base_path, 'csv', file_name+'.csv')):
        df_csv = pd.read_csv(os.path.join(base_path, 'csv', file_name+'.csv'))
        df_wav = pd.read_csv(os.path.join(base_path, 'wav_start_end', file_name+'.csv'))
        number = int(turn_name.split('_')[-1]) - 1
        eou = df_csv.iloc[number]['end'] - df_wav.iloc[number]['wav_start']
    else:
        exit()
    
    
    # timing estimation
    ## reset model
    speech2text.reset_inference_cache()
    model.eval()
    model.reset_state()
    generator.reset()
    
    ## init
    chunk_size = 800*2         # 100[ms]ごと
    # chunk_size = 800           # 50[ms]ごと
    asr_chunk_size = 2048      # 128[ms]ごと
    time_size = chunk_size // 16
    n_size = time_size // 50
    single = False
    pred_list = []
    text_list = []
    silence_list = []
    vad_list = []
    text = ''
    pre_text = ''
    pre_id = [0]
    asr_buffer = np.array([], dtype='int16')
    
    spec_list = []
    
    """
    pad = np.zeros(1440, np.int16)
    wav_chunk = np.concatenate([pad, wav])
    spec_list2 = generator.input_wave(wav_chunk)
    generator.reset()
    """
    
    ## model
    with torch.no_grad():
        
        for i in range(len(wav)//chunk_size):
            
            # wavの範囲
            wav_chunk = wav[chunk_size*i:chunk_size*(i+1)]

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
            
            # spectrogram
            if i == 0:
                pad = np.zeros(1440, np.int16)
                # pad = np.zeros(800, np.int16)
                wav_chunk = np.concatenate([pad, wav_chunk])
            spec = generator.input_wave(wav_chunk)
            spec_list.extend(spec)
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
            pred_list.extend(*out.tolist())
            text_list.append(text)
            silence_list.extend(*silence.tolist())
            vad_list.extend(*vad_out.tolist())
    
    """
    for spec, spec2 in zip(spec_list, spec_list2):
        print(spec[0])
        print()
        print(spec2[0])
        print('-')
    """
        
    # make movie
    pred_list.insert(0, 0)
    vad_list.insert(0, 0)
    if single:
        t1 = [i * time_size for i in range(len(pred_list))]
    else:
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
        if i * time_size // n_size >= eou and eou != -1:
            graph += ax1.plot([eou, eou], [0, 1], color='y')
        if single:
            x2 = t2[:chunk_size*i]
            y2 = wav[:chunk_size*i]
        else:
            x2 = t2[:(chunk_size//n_size)*i]
            y2 = wav[:(chunk_size//n_size)*i]
        graph += ax2.plot(x2, y2, color='c')
        graphs.append(graph)
    # animation
    if single:
        ani = ArtistAnimation(fig, graphs, interval=time_size)
    else:
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