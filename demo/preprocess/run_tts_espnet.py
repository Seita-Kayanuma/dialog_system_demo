import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import resample
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none


# python demo/preprocess/run_tts_espnet.py
# condaの(espnet)環境で実行

# シナリオのパス
CSVPATH = 'demo/scinario/ope/data_20241104_090511_summary_20241104_085657.txt.csv'
# CSVPATH = 'demo/scinario/operation/demo_kana-kanji.csv'
# CSVPATH = 'demo/scinario/randomQA/choice_sample.csv'
# 音声の保存パス
# OUTPATH = 'demo/audio/tts/operation/demo_jsut_16k_0000'
# OUTPATH = 'demo/audio/tts/choice_sample/jsut_16k'

# CSVPATHから最後のファイル名を取得
csv_filename = os.path.basename(CSVPATH)
# 拡張子を除いたファイル名（任意で必要に応じて）
csv_basename = os.path.splitext(csv_filename)[0]
# 音声の保存パス
OUTPATH = os.path.join('demo/preprocess/data/scinario', csv_basename)

# 音声合成　事前学習モデル
TAG = 'kan-bayashi/jsut_full_band_vits_prosody' # JSUT
# TAG = 'kan-bayashi/tsukuyomi_full_band_vits_prosody' # つくよみ


def main():
    
    # sr
    new_sr = 16000
    org_sr = 44100
    # text2speech
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
    texts = list(df['text'][df['system']==1])
    single_texts = []
    for text in texts:
        single_text = text.split('。')
        single_text = [st for st in single_text if st != '']
        single_texts.extend(single_text)
    
    # ディレクトリ作成
    os.makedirs(OUTPATH, exist_ok=True)
    
    # 音声生成/サンプリング周波数変換/保存
    for i in tqdm(range(len(single_texts))):
        # 4桁のゼロ埋めにしてファイル名を生成
        file_name = f'{str(i).zfill(4)}.wav'
        audio_path = os.path.join(OUTPATH, file_name)
        # 音声生成
        with torch.no_grad():
            wav = text2speech(single_texts[i])["wav"]
        wavdata = wav.view(-1).cpu().numpy()
        # サンプリング周波数変換
        num_samples_new = int(len(wavdata) * new_sr / org_sr)
        new_wavdata = resample(wavdata, num_samples_new)
        new_wavdata = new_wavdata / np.max(np.abs(new_wavdata))
        new_wavdata = (new_wavdata * 32767).astype(np.int16)
        # 音声保存
        wavfile.write(audio_path, new_sr, new_wavdata)
    

if __name__ == '__main__':
    main()