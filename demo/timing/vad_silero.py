from typing import Any, Dict, Union, Tuple
from enum import Enum

from audio import AbsAudio
import numpy as np
from collections import deque
import time

from audio import AudioData

import torch
torch.set_num_threads(1)


class VADState(Enum):
    Idle = 0,
    Started = 1,
    Ended = 2,
    Continue = 3,


class VAD:
    def __init__(self,
                 webrtcvad_mode=3, # 0 is the most aggressive (to detect speech), 3 is the least.
                 start_frame_num_thresh=5,
                 start_frame_rollback=10,
                 end_frame_num_thresh=30, # 修正
                 speech_reco_frame_chunk_num=10
    ):
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad') # , force_reload=True)
        self.model = model
        self.start_frame_num_thresh = start_frame_num_thresh
        self.start_frame_rollback = start_frame_rollback
        self.end_frame_num_thresh = end_frame_num_thresh
        self.speech_reco_frame_chunk_num = speech_reco_frame_chunk_num
        self.start_frame_num = 0

        self.buffer = []
        self.count = 0
        self.is_on = False

        self.callbacks = []

        self.buffer_for_vad = []
        self.prev_is_speech = False
        
        # yaguchi add
        self.silent_time = 0
        self.system_utterance = False

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def remove_callback(self, callback):
        self.callbacks.remove(callback)

    def process(self, audio: AudioData) -> Tuple[Union[AudioData, None], VADState]:
        self.buffer.append(audio)

        self.buffer_for_vad.append(audio.data_np)
        if len(self.buffer_for_vad) < 4:
            vad_result = self.prev_is_speech
        else:
            audio_float32 = np.concatenate(self.buffer_for_vad).astype(np.float32)
            
            # 萱沼追加　20240716 共有時
            expected_samples = 512
            if len(audio_float32) < expected_samples:
                audio_float32 = np.pad(audio_float32, (0, expected_samples - len(audio_float32)), 'constant')
            else:
                audio_float32 = audio_float32[:expected_samples]
            
            confidence = self.model(torch.tensor(audio_float32), 16000).item()
            vad_result = confidence > 0.5
            self.prev_is_speech = vad_result
            self.buffer_for_vad.clear()

        def _flush_buffer():
            assert len(self.buffer) > 0, "Buffer is empty"
            data_bytes = b''.join([a.data_bytes for a in self.buffer])
            data_np = np.concatenate([a.data_np for a in self.buffer])
            time = self.buffer[0].time
            self.buffer.clear()
            return AudioData(data_bytes, data_np, time)

        result = None
        state = VADState.Idle
        
        if vad_result: self.silent_time = 0
        else: self.silent_time += 1
        
        if self.is_on:
            if vad_result:
                self.count = 0
            else:
                self.count += 1
            # システム発話中は強制的にVADをオフにする
            if self.count >= self.end_frame_num_thresh or self.system_utterance:
                self.forced_termination = False
                self.is_on = False
                self.count = 0
                result = _flush_buffer()
                state = VADState.Ended
            elif len(self.buffer) >= self.speech_reco_frame_chunk_num:
                result = _flush_buffer()
                state = VADState.Continue
            else:
                result = None
                state = VADState.Continue
        else:
            # システム発話中以外でカウント
            if not self.system_utterance:
                if self.count == 0 and vad_result:
                    self.count += 1
                elif self.count > 0 and vad_result:
                    self.count += 1
                else:
                    self.count = 0
            if self.count >= self.start_frame_num_thresh:
                self.is_on = True
                self.count = 0
                result = _flush_buffer()
                state = VADState.Started
            else:
                self.buffer = self.buffer[-self.start_frame_rollback:]
                result = None
                state = VADState.Idle

        if result is not None:
            for callback in self.callbacks:
                callback(result, state)

        return result, state
        

if __name__ == "__main__":
    from .audio import PyaudioAudio

    vad = VAD()
    audio = PyaudioAudio()

    def vad_callback(data, state):
        print(state, data.data_np.shape, data.time)

    vad.add_callback(vad_callback)
    audio.add_callback(vad.process)

    audio.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        audio.stop()
    # while True:
    #     audio_data = audio.get()
    #     if audio_data is not None:
    #         result, state = vad.run(audio_data)
    #         print(state)
    #     else:
    #         break
    # audio.stop()