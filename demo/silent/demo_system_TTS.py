import os
import time
import wave
import torch
import pyaudio
import datetime
import threading
import numpy as np
import pandas as pd
from enum import Enum
from queue import Queue
from dataclasses import dataclass
import time

from asr import ASR
from utils import split_pron_to_mora
from vad_silero import VAD, VADState
from audio import AbsAudio, AudioData
import openai
import threading

import asyncio
import websockets
import json
import base64
from pydub import AudioSegment
from pydub.playback import play

# OpenAI API キーの設定（環境変数から取得）
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# PyTorchのスレッド数を制限
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# python demo/silent/demo_system_TTS.py 


class MainASRState(Enum):
    Idle = 0
    Started = 1

class AudioPlayerState(Enum):
    Backchannel = 0
    Utterance = 1

@dataclass
class VADData:
    audio_data: AudioData
    vad_state: VADState


class ASRWorker:
    def __init__(self,
                 asr: ASR,
                 audio: AbsAudio,
        ):
        self.audio = audio
        self.asr = asr
        self.asr_state = MainASRState.Idle
        self.asr_start_time = None
        self.queue = Queue()
        # yaguchi add
        self.history_text = ''
        self.history_len = 0
        self.all_text = ''
        self.all_len = 0

    def _get_current_time(self):
        return self.audio.current_time()

    def setActive(self, active: bool):
        self.is_active = active

    def put(self, data: VADData):
        self.queue.put(data)

    def run(self):
        while True:
            data = self.queue.get()
            audio_data = data.audio_data
            vad_state = data.vad_state

            if vad_state == VADState.Started:
                self.asr_state = MainASRState.Started

            if self.asr_state == MainASRState.Started:
                is_final = vad_state == VADState.Ended
                kana = self.asr.recognize(audio_data.data_np, is_final=is_final)

                if kana is not None:
                    mora, mora_len = split_pron_to_mora(kana)
                    self.all_text = self.history_text + f' {mora}'
                    self.all_len = self.history_len + mora_len
                    print(datetime.datetime.now(), '[ASRWorker run]', f'text: {self.all_text}')
                
                    if is_final:
                        self.asr_state = MainASRState.Idle
                        self.history_text += f' {mora}'
                        self.history_len += mora_len
    
    def reset(self):
        self.history_text = ''
        self.history_len = 0
        self.all_text = ''
        self.all_len = 0


# yaguchi add
class AudioPlayer:
    def __init__(self,
                 scinario_path: str
        ):
        self.pyaudio = pyaudio.PyAudio()
        self.scinario_path = scinario_path
        self.audio_num = 0
        self.backchannel_path = "demo/silent/scinario/sample/hai.wav"
        path_list = [os.path.join(self.scinario_path, name) for name in os.listdir(self.scinario_path) if 'wav' in name]
        self.path_list = path_list
        self.audio_limit = len(path_list)
        self.flg = True
        
        self.audio_datas = []
        # pyaudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True
        )
        path_list.sort()
        path_list.append(self.backchannel_path)
        print(path_list)
        for path in path_list:
            wf = wave.open(path, 'rb')
            audio_data = wf.readframes(wf.getnframes())
            self.audio_datas.append(audio_data)
            wf.close()
    
    def play_pyaudio(self, audio_data: np.ndarray):
        self.stream.write(audio_data)

    def play(self, state: AudioPlayerState):
        # 相槌
        if state == AudioPlayerState.Backchannel:
            self.play_pyaudio(self.audio_datas[-1])
            # self.play_pyaudio(self.backchannel_path)
        # 発話
        else:
            self.play_pyaudio(self.audio_datas[self.audio_num])
            # self.play_pyaudio(self.path_list[self.audio_num])
            self.audio_num += 1
            # シナリオが最後まで進んだら終了
            if self.audio_num >= self.audio_limit:
                self.flg = False
            

class Main:
    def __init__(self,
                 audio: AbsAudio,
                 vad: VAD,
                 asr: ASR
        ):
        self.asr = asr
        self.audio = audio
        self.vad = vad
        self.cond = threading.Condition()
        self.stdscr = None
        self.asr_worker = None
        self.audioplayer = None
        self.user_utterance_flg = False
        self.system_utterance_flg = False
        self.silent_backchannel_limit = 40      # 400[ms]で相槌
        self.silent_utterance_limit = 100       # 1000[ms]で発話 #変更萱沼
        # self.scinario_path = "demo/silent/scinario/tsukuyomi"
        self.scinario_path = "demo/silent/scinario/ope" 
        df = pd.read_csv(os.path.join(self.scinario_path, 'data_.csv'))
        self.len_user_uttrances = [int(l*0.8) for l in list(df['len'][df['system']==0])]

    def _get_current_time(self):
        return self.audio.current_time()

    def _vad_callback(self, data: AudioData, state: VADState):
        assert self.asr_worker is not None
        self.asr_worker.put(VADData(data, state))
        self.vad_state = state
        # yaguchi add
        if state == VADState.Started: 
            self.user_utterance_flg = True
    
    # システム発話
    def audio_play(self, state: AudioPlayerState):
        assert self.audioplayer is not None
        print(datetime.datetime.now(), '[Main/audio_play]')
        self.vad.system_utterance = True
        # 相槌
        if state == AudioPlayerState.Backchannel:
            self.audioplayer.play(state)
        # 発話
        else:
            self.system_utterance_flg = True
            self.audioplayer.play(state)
        self.reset(state)
    
    # 相槌/発話後処理
    def reset(self, state: AudioPlayerState):
        self.user_utterance_flg = False
        self.vad.system_utterance = False
        if state == AudioPlayerState.Utterance:
            self.system_utterance_flg = False
            self.vad.silent_time = 0
            self.asr_worker.reset()
            
    
            
    def run(self):
        self.audioplayer = AudioPlayer(self.scinario_path)
        self.asr_state = MainASRState.Idle
        self.vad_state = VADState.Idle
        self.audio.add_callback(self.vad.process)
        self.vad.add_callback(self._vad_callback)
        self.asr_worker = ASRWorker(
            self.asr,
            self.audio
        )
        self.asr_worker_thread = threading.Thread(target=self.asr_worker.run)
        self.asr_worker_thread.daemon = True
        self.asr_worker_thread.start()
        self.audio.start()
        
        # 発話/相槌 処理
        first_loop = True
        while self.audioplayer.flg:
            # システム主導
            if first_loop:
                self.audio_play(AudioPlayerState.Utterance)
                first_loop = False
            
            if self.user_utterance_flg and not self.system_utterance_flg:
                # print(datetime.datetime.now(), '[Main/run]', self.vad.silent_time) #　ゼミでのデモ用にコメントアウト
                # ユーザの発話が想定以上の長さじゃない時
                if self.asr_worker.all_len < self.len_user_uttrances[self.audioplayer.audio_num]:
                    # if self.vad.silent_time >= self.silent_backchannel_limit:
                    #     self.audio_play(AudioPlayerState.Backchannel)
                    if self.vad.silent_time >= self.silent_utterance_limit*3:
                        self.audio_play(AudioPlayerState.Utterance)
                    pass
                # ユーザの発話が想定以上の長さになった時 ユーザが質問を行うとき
                else:
                    if self.vad.silent_time >= self.silent_utterance_limit:
                        # self.audio_play(AudioPlayerState.Utterance)
                        time1 = time.time()
                        kanji = self.asr.convert_kana_to_text(self.asr_worker.all_text) #遅かったら引数に指定することで、n best の条件を変えられる。defo5らしい
                        time2 = time.time()
                        time3 = time.time()
                        self.get_chatgpt_response(kanji) # ここでTTS
                        time4 = time.time()
                        for i in range(1):
                            print("モーラ 出力:", self.asr_worker.all_text)
                            print("かな漢字 出力:", kanji)
                            print("かな漢字変換にかかる時間", time2 - time1)
                            print("OpenAI TTSにかかる時間", time4 - time3)
                        self.asr_worker.all_len = 0 #質問フェーズなので、asr_worker.all_lenリセットをかけないと
            time.sleep(0.01)
    
    async def get_chatgpt_response_async(self, recognized_text):
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "OpenAI-Beta": "realtime=v1",
        }

        async with websockets.connect(url, extra_headers=headers) as ws:
            print("サーバーに接続しました。")

            # summary.txtを読み込む
            summary_path = 'demo/preprocess/data/summary/summary_20241103_233544.txt'

            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_text = f.read()
            except FileNotFoundError:
                print(f"エラー: ファイル {summary_path} が見つかりませんでした。")
                summary_text = ""
            except Exception as e:
                print(f"ファイルの読み込み中にエラーが発生しました: {e}")
                summary_text = ""

            # 指示を更新
            instructions_text = (
                "あなたは医療の専門家です。患者さんからの質問に対して、わかりやすく、簡潔に回答してください。"
                "以下は手術の概要です。必要に応じて回答に活用してください。\n\n"
                f"{summary_text}"
            )

            await ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "instructions": instructions_text
                }
            }))

            # 患者の質問を送信
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": recognized_text
                        }
                    ]
                }
            }
            await ws.send(json.dumps(event))

            # 音声データを保存するためのバッファ
            audio_chunks = []

            # 保存パスを設定
            os.makedirs('demo/audio/chatGPT_response', exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demo/audio/chatGPT_response/output_audio_{timestamp}.wav"

            async for message in ws:
                event = json.loads(message)
                if event['type'] == 'response.audio.delta':
                    audio_base64 = event['delta']
                    audio_data = base64.b64decode(audio_base64)
                    audio_chunks.append(audio_data)

                elif event['type'] == 'response.audio.done':
                    complete_audio = b"".join(audio_chunks)
                    audio_segment = AudioSegment(
                        complete_audio,
                        frame_rate=24000,
                        sample_width=2,
                        channels=1
                    )
                    play(audio_segment)
                    with wave.open(filename, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(24000)
                        wav_file.writeframes(complete_audio)
                    print(f"音声が '{filename}' として保存されました。")
                    break

                elif event['type'] == 'response.text.done':
                    print("テキストの受信が完了しました。")
                elif event['type'] == 'error':
                    print(f"エラー: {event['error']['message']}")
                    break

    def get_chatgpt_response(self, recognized_text):
        asyncio.run(self.get_chatgpt_response_async(recognized_text))
      
    

if __name__ == "__main__":
    from audio import PyaudioAudio

    asr = ASR.from_pretrained(
        espnet2_asr_model_tag="fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105",
        espnet2_asr_args=dict(
            streaming=True,
            lm_weight=0.0,
            beam_size=20,
            beam_search_config=dict(search_type="maes")
        ),
        kana_kanji_model_tag="fujie/kana_kanji_20240307",
    )

    audio = PyaudioAudio()
    vad = VAD(webrtcvad_mode=3, end_frame_num_thresh=30)

    main = Main(audio, vad, asr)
    main.run()