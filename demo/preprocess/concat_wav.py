from pydub import AudioSegment


# python demo/preprocess/concat_wav.py
WAV1PATH = 'demo/tts/audio/jsut/567.wav'
WAV2PATH = 'demo/tts/audio/jsut/8.wav'
OUTPATH = 'demo/tts/audio/jsut/5678.wav'
# 無音区間の長さ（秒）
DURATION = 1


def main():
    # 音声ファイルの読み込み
    wav1 = AudioSegment.from_file(WAV1PATH)
    wav2 = AudioSegment.from_file(WAV2PATH)
    
    # 無音区間の作成
    silent_segment = AudioSegment.silent(duration=DURATION*1000)

    # 音声ファイルを無音区間で連結
    combined = wav1 + silent_segment + wav2

    # 連結した音声ファイルを保存
    combined.export(OUTPATH, format="wav")
    

if __name__ == '__main__':
    main()