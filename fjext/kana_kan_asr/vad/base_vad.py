from typing import Any, Dict, Union, Tuple
from enum import Enum
from ..audio import AudioData

class VADState(Enum):
    Idle = 0,
    Started = 1,
    Ended = 2,
    Continue = 3,

class BaseVAD(object):
    def __init__(self,
                 start_frame_num_thresh=5,
                 start_frame_rollback=10,
                 end_frame_num_thresh=30,
                 speech_reco_frame_chunk_num=10):
        self.start_frame_num_thresh = start_frame_num_thresh
        self.start_frame_rollback = start_frame_rollback
        self.end_frame_num_thresh = end_frame_num_thresh
        self.speech_reco_frame_chunk_num = speech_reco_frame_chunk_num
        self.start_frame_num = 0

        self.buffer = []
        self.count = 0
        self.is_on = False

        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def remove_callback(self, callback):
        self.callbacks.remove(callback)

    def _run_vad(self, audio: AudioData) -> Tuple[book, float]:
        raise NotImplementedError

    def process(self, audio: AudioData) -> Tuple[Union[AudioData, None], VADState]:
        vad_result = self._run_vad(audio)
        self.buffer.append(audio)

        def _flush_buffer():
            assert len(self.buffer) > 0, "Buffer is empty"
            data_bytes = b''.join([a.data_bytes for a in self.buffer])
            data_np = np.concatenate([a.data_np for a in self.buffer])
            time = self.buffer[0].time
            self.buffer.clear()
            return AudioData(data_bytes, data_np, time)

        result = None
        state = VADState.Idle

        if self.is_on:
            if vad_result:
                self.count = 0
            else:
                self.count += 1
            if self.count >= self.end_frame_num_thresh:
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

        return None, VADState.Idle