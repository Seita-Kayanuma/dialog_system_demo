import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from ttslearn.tacotron import Tacotron2TTS


# python demo/tts/run_tts.py
CSVPATH = 'demo/tts/audio/sample2'


# Tacotron2
def taco():
    engine = Tacotron2TTS()
    df_path = os.path.join(CSVPATH, 'data.csv')
    df = pd.read_csv(df_path)
    texts = list(df['text'][df['system']==0])
    for i in range(len(texts)):
        audio_path = os.path.join(CSVPATH, f'{i}.wav')
        print(f'STEP【{i+1}/{len(texts)}】')
        x, sr = engine.tts(texts[i])
        wavfile.write(audio_path, sr, x.astype(np.int16))


if __name__ == '__main__':
    taco()