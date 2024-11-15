import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import resample
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none

# python demo/preprocess/tts_answer.py

# python demo/preprocess/run_tts_espnet.pyをもとに作成。Q&Aの答えを合成するために作成
# condaの(espnet)環境で実行

# シナリオのパス
CSVPATH = '/Users/seita/work/RTE/Surgery_explanation/chatGPT/data/Q&A/sample.csv'
OUTPATH = 'demo/audio/tts/operation/Q&A/sample'

# 音声合成　事前学習モデル
TAG = 'kan-bayashi/jsut_full_band_vits_prosody'

def main():
    new_sr = 16000
    org_sr = 44100
    
    # 無音の長さ（秒）
    silence_duration = 0.5  # 0.5秒の無音
    silence_array = np.zeros(int(new_sr * silence_duration), dtype=np.int16)
    
    # text2speech モデルの読み込み
    text2speech = Text2Speech.from_pretrained(
        model_tag=str_or_none(TAG),
        vocoder_tag=str_or_none('none'),
        device="cpu",
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
        speed_control_alpha=1.0,
        noise_scale=0.333,
        noise_scale_dur=0.333,
    )
    
    # シナリオの読み込み
    df = pd.read_csv(CSVPATH)
    texts = list(df['回答'])
    
    os.makedirs(OUTPATH, exist_ok=True)
    
    # 行ごとの音声ファイルの結合用リスト
    combined_audio = []

    # 音声生成と保存
    for i in tqdm(range(len(texts))):
        single_texts = [st for st in texts[i].split('。') if st != '']
        audio_list = []
        
        for j, sentence in enumerate(single_texts):
            with torch.no_grad():
                wav = text2speech(sentence)["wav"]
            wavdata = wav.view(-1).cpu().numpy()
            
            # サンプリング周波数変換
            num_samples_new = int(len(wavdata) * new_sr / org_sr)
            new_wavdata = resample(wavdata, num_samples_new)
            new_wavdata = new_wavdata / np.max(np.abs(new_wavdata))
            new_wavdata = (new_wavdata * 32767).astype(np.int16)
            
            audio_list.append(new_wavdata)
            # 無音を追加
            audio_list.append(silence_array)
        
        # 結合してリストに追加
        combined_audio.extend(audio_list)

        # 1行分の音声をファイルとして保存
        combined_audio_array = np.concatenate(combined_audio)
        file_name = f'{str(i).zfill(4)}.wav'
        audio_path = os.path.join(OUTPATH, file_name)
        wavfile.write(audio_path, new_sr, combined_audio_array)
        combined_audio.clear()  # 次の行のためにリセット

if __name__ == '__main__':
    main()

