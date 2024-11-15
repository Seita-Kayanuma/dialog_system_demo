import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import resample
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import sys

# python demo/preprocess/modules/tts.py "output_20241109_174659.csv"

# demo/preprocess/run_tts_espnet.pyに修正を加えたもの

def main(filename):
    CSVBASE = 'demo/preprocess/data/scinario_kanji'
    CSVPATH = os.path.join(CSVBASE, filename)  # 漢字のシナリオ
    OUTPATH_BASE = 'demo/preprocess/data/scinario'

    # 拡張子を除いたファイル名（任意で必要に応じて）
    csv_basename = os.path.splitext(filename)[0]
    OUTPATH = os.path.join(OUTPATH_BASE, csv_basename)

    # サンプリング周波数
    new_sr = 16000
    org_sr = 44100

    # 音声合成モデルのロード
    text2speech = Text2Speech.from_pretrained(
        model_tag=str_or_none('kan-bayashi/jsut_full_band_vits_prosody'),
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
    texts = list(df['text'][df['system'] == 1])
    single_texts = []
    for text in texts:
        single_text = text.split('。')
        single_text = [st for st in single_text if st != '']
        single_texts.extend(single_text)

    # 出力ディレクトリ作成
    os.makedirs(OUTPATH, exist_ok=True)

    # 音声生成/サンプリング周波数変換/保存
    for i in tqdm(range(len(single_texts))):
        file_name = f'{str(i).zfill(4)}.wav'
        audio_path = os.path.join(OUTPATH, file_name)
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
    if len(sys.argv) != 2:
        print("Usage: python run_tts_espnet.py <FILENAME>")
        sys.exit(1)

    FILENAME = sys.argv[1]
    main(FILENAME)
