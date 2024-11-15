import os
import pandas as pd
from pydub import AudioSegment


# python demo/preprocess/concat_silent_same.py
# 無音区間の長さ（秒）
USER_DURATION = 0.3
SYSTEM_DURATION = 0.3
CSVPATH = 'demo/scinario/concat_short.csv'
OUTPATH = f'demo/audio/silent/short'


def main():
    
    df = pd.read_csv(CSVPATH)
    allwav = AudioSegment.silent(duration=0)
    
    for i, wavpath in enumerate(df['wav']):
        wav = AudioSegment.from_file(wavpath)
        if i == 0:
            silent_duration = 0
        elif i % 2 == 0:
            silent_duration = USER_DURATION
        else:
            silent_duration = SYSTEM_DURATION
        silent_segment = AudioSegment.silent(duration=silent_duration*1000)
        allwav = allwav + silent_segment + wav

    # 連結した音声ファイルを保存
    os.makedirs(OUTPATH, exist_ok=True)
    allwav.export(os.path.join(OUTPATH, f'{int(SYSTEM_DURATION*1000)}.wav'), format="wav")
    

if __name__ == '__main__':
    main()