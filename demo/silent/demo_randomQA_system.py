import os
import time
import wave
import torch
import random
import pyaudio
import datetime
import threading
import numpy as np
import pandas as pd
from enum import Enum
from queue import Queue
from dataclasses import dataclass

from asr import ASR
from utils import split_pron_to_mora, get_similar_word
from vad_silero import VAD, VADState
from audio import AbsAudio, AudioData


# PyTorchのスレッド数を制限
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# python demo/silent/demo_randomQA_system.py


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
        self.user_choice = None

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
                    
                    self.user_choice = get_similar_word(self.all_text.replace(' ', ''), ['イチゴウカン', '二ゴウカン', 'サンゴウカン'])
                    
                    print(datetime.datetime.now(), '[ASRWorker run]', f'text: {self.all_text}', f'user_choice: {self.user_choice}')
                
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
        names = [name for name in os.listdir(self.scinario_path) if 'wav' in name]
        path_list = [os.path.join(self.scinario_path, name) for name in names]
        # QAの質問選択(ランダム)
        """
        scinario_nums = random.sample(range(len(names)-2), 4)
        scinario_nums.extend([len(names)-2, len(names)-1])
        path_list = [os.path.join(self.scinario_path, f'{scinario_num}.wav') for scinario_num in scinario_nums]
        """
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
        # path_list.sort()
        for path in path_list:
            wf = wave.open(path, 'rb')
            audio_data = wf.readframes(wf.getnframes())
            self.audio_datas.append(audio_data)
            wf.close()
    
    def play_pyaudio(self, audio_data: np.ndarray):
        self.stream.write(audio_data)

    def play(self, state: AudioPlayerState, user_choice: int):
        # 相槌
        if state == AudioPlayerState.Backchannel:
            self.play_pyaudio(self.audio_datas[-1])
            # self.play_pyaudio(self.backchannel_path)
        # 発話
        else:
            self.play_pyaudio(self.audio_datas[user_choice])
            # self.play_pyaudio(self.audio_datas[user_choice])
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
        self.silent_utterance_limit = 120        # 700[ms]で発話
        self.scinario_path = "demo/audio/tts/choice_sample/jsut_16k"

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
            self.audioplayer.play(state, self.asr_worker.user_choice)
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
        print(datetime.datetime.now(), '[Main/run]', 'start')
        
        # 発話/相槌 処理
        while self.audioplayer.flg:
            if self.user_utterance_flg and not self.system_utterance_flg:
                print(datetime.datetime.now(), '[Main/run]', self.vad.silent_time)
                if self.vad.silent_time >= self.silent_utterance_limit and self.asr_worker.all_len >= 3:
                    self.audio_play(AudioPlayerState.Utterance)
            time.sleep(0.01)


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