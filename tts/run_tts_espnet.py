import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none


# python tts/run_tts_espnet.py
CSVPATH = 'demo/scinario/randomQA/kana-kanji.csv'
OUTPATH = 'demo/audio/tts/randomQA/jsut_16k'


def main():
    # source: https://zenn.dev/syoyo/articles/8b533927189bde
    lang = 'Japanese'
    vocoder_tag = 'none'
    # JSUT
    tag = 'kan-bayashi/jsut_full_band_vits_prosody'
    # つくよみ
    # tag = 'kan-bayashi/tsukuyomi_full_band_vits_prosody'
    
    # sr
    new_sr = 16000
    org_sr = 44100
    
    # output
    os.makedirs(OUTPATH, exist_ok=True)

    # Use device="cuda" if you have GPU
    text2speech = Text2Speech.from_pretrained(
        model_tag=str_or_none(tag),
        vocoder_tag=str_or_none(vocoder_tag),
        device="cpu",
        # Only for Tacotron 2 & Transformer
        threshold=0.5,
        # Only for Tacotron 2
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2 & VITS
        speed_control_alpha=1.0,
        # Only for VITS
        noise_scale=0.333,
        noise_scale_dur=0.333,
    )
    
    df = pd.read_csv(CSVPATH)
    texts = list(df['text'][df['system']==1])
    for i in range(len(texts)):
        audio_path = os.path.join(OUTPATH, f'{i}.wav')
        print(f'STEP【{i+1}/{len(texts)}】')
        with torch.no_grad():
            wav = text2speech(texts[i])["wav"]
        wavdata = wav.view(-1).cpu().numpy()
        num_samples_new = int(len(wavdata) * new_sr / org_sr)
        new_wavdata = resample(wavdata, num_samples_new)
        new_wavdata = new_wavdata / np.max(np.abs(new_wavdata))
        new_wavdata = (new_wavdata * 32767).astype(np.int16)
        wavfile.write(audio_path, new_sr, new_wavdata)
    

if __name__ == '__main__':
    main()