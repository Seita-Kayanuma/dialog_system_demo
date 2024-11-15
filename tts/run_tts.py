import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from ttslearn.tacotron import Tacotron2TTS


# Tacotron2
# python demo_/tts/run_tts.py
WAVPATH = 'demo_/tts/audio/sample'
CSVPATH = 'demo_/tts/audio/sample3'
OUTPATH = 'demo_/tts/audio/sample3'


# list_to_audio
def list_to_audio():
    engine = Tacotron2TTS()
    df_path = os.path.join(CSVPATH, 'data.csv')
    df = pd.read_csv(df_path)
    texts = list(df['text'][df['system']==0])
    for i in range(len(texts)):
        audio_path = os.path.join(OUTPATH, f'{i}.wav')
        print(f'STEP【{i+1}/{len(texts)}】')
        x, sr = engine.tts(texts[i])
        wavfile.write(audio_path, sr, x.astype(np.int16))


def text_to_audio():
    text = 'はい'
    engine = Tacotron2TTS()
    audio_path = os.path.join(WAVPATH, 'hai.wav')
    x, sr = engine.tts(text)
    wavfile.write(audio_path, sr, x.astype(np.int16))


if __name__ == '__main__':
    list_to_audio()
    # text_to_audio()