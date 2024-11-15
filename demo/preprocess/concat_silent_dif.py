import os
import pandas as pd
from pydub import AudioSegment


# python demo/preprocess/concat_silent_dif.py
CSVPATH = 'demo/scinario/concat_duration.csv'
OUTPATH = f'demo/audio/silent'
NAME = 'concat_duration_kayanuma'


def main():
    
    df = pd.read_csv(CSVPATH)
    allwav = AudioSegment.silent(duration=0)
    
    for i, (wavpath, duration) in enumerate(zip(df['wav'], df['duration'])):
        wav = AudioSegment.from_file(wavpath)
        if i == 0:
            silent_duration = 0
        else:
            silent_duration = duration
        silent_segment = AudioSegment.silent(duration=silent_duration)
        allwav = allwav + silent_segment + wav

    # 連結した音声ファイルを保存
    os.makedirs(OUTPATH, exist_ok=True)
    allwav.export(os.path.join(OUTPATH, f'{NAME}.wav'), format="wav")
    

if __name__ == '__main__':
    main()