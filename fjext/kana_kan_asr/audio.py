from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from dataclasses import dataclass
import time
import queue
import threading
import numpy as np
import pyaudio

@dataclass
class AudioData:
    data_bytes: bytes
    data_np: np.ndarray
    time: float


class AbsAudio(ABC):
    def __init__(self, enable_queue=False):
        self.enable_queue = enable_queue
        self.queue = None
        self.callbacks = []

    @abstractmethod
    def current_time(self) -> float:
        raise NotImplementedError

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def remove_callback(self, callback):
        self.callbacks.remove(callback)

    def start(self):
        if self.enable_queue:
            self.queue = queue.Queue()

    def stop(self):
        if self.enable_queue:
            self.queue = None

    def get(self) -> AudioData:
        """Get audio data.

        Returns:
            AudioData

        Raises:
            AssertionError: If queue is not enabled
        """
        assert self.enable_queue, "Queue is not enabled"
        return self.queue.get()

    def _put(self, data: AudioData):
        if self.enable_queue:
            self.queue.put(data)
        for callback in self.callbacks:
            callback(data)

class SoundfileAudio(AbsAudio):
    def __init__(self,
                 filename,
                 chunk_size=160,
                 simulate_realtime=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        import soundfile as sf
        self.audio, fs = sf.read(filename)
        assert fs == 16000, "Sampling rate must be 16kHz"

        self.chunk_size = chunk_size
        self.simulate_realtime = simulate_realtime

        self.thread = None

    def start(self):
        assert self.thread is None, "Audio is already started"
        super().start()
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        assert self.thread is None, "Audio is not started"
        self.thread.join()
        super().stop()

    def _run(self):
        start_time = time.time()

        pos = 0
        while pos < len(self.audio):
            current_time = time.time()
            # wait for the expected time to simulate real-time
            if self.simulate_realtime:
                expected_time = pos / 16000 + start_time
                if current_time < expected_time:
                    time.sleep(expected_time - current_time)
            # get the next chunk
            if pos + self.chunk_size < len(self.audio):
                chunk = self.audio[pos:pos+self.chunk_size]
            else:
                chunk = self.audio[pos:]
            pos += self.chunk_size
            chunk_bytes = np.int16(chunk * (2**15 - 1)).tobytes()

            result = AudioData(
                data_bytes=chunk_bytes,
                data_np=chunk,
                time=current_time,
            )
            self._put(result)

    def current_time(self) -> float:
        return time.time()

class PyaudioAudio(AbsAudio):
    def __init__(self,
                 input_device_name=None,
                 chunk_size=160,
                 **kwargs):
        super().__init__(**kwargs)

        self.chunk_size = chunk_size
        self.pyaudio = pyaudio.PyAudio()

        # 入力デバイスのインデクスを設定
        if input_device_name is not None:
            input_device_index = self._get_device_index(input_device_name)
            if input_device_index is None:
                raise ValueError(f"Input device '{input_device_name}' not found")
            input_max_channels = self.pyaudio.get_device_info_by_index(input_device_index)["maxInputChannels"]
            if input_max_channels < 1:
                raise ValueError(f"Input device '{input_device_name}' has no input channel")
            self.input_device_index = input_device_index
        else:
            self.input_device_index = self.pyaudio.get_default_input_device_info()["index"] 

        self.stream = None

        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        while True:
            data = self.queue.get()
            self._put(data)

    def current_time(self) -> float:
        if self.stream is None:
            return time.time()
        else:
            return self.stream.get_time()

    def _get_device_index(self, device_name):
        for i in range(self.pyaudio.get_device_count()):
            if self.pyaudio.get_device_info_by_index(i)["name"] == device_name:
                return i
        return None

    def start(self):
        super().start()
        self.stream = self.pyaudio.open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=16000,
                                        input=True,
                                        input_device_index=self.input_device_index,
                                        frames_per_buffer=self.chunk_size,
                                        stream_callback=self._callback)
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        result = AudioData(
            data_bytes=in_data,
            data_np=np.frombuffer(in_data, dtype=np.int16) / (2**15 - 1),
            time=time_info["input_buffer_adc_time"],
        )
        # self._put(result)
        self.queue.put(result)
        return (None, pyaudio.paContinue)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        self.queue = None
        super().stop()

    def get_device_name(self) -> str:
        return self.pyaudio.get_device_info_by_index(self.input_device_index)["name"]


if __name__ == "__main__":
    # audio = SoundfileAudio("data/audio/A01M0097_0002761_0009322.flac")

    def callback(data):
        print(data.time)

    audio = PyaudioAudio(enable_queue=True)
    audio.add_callback(callback)
    audio.start()
    # while True:
    #     data = audio.get()
    #     if data is None:
    #         break
    #     print(data["time"])
    #     # time.sleep(0.1)
    time.sleep(10)
    audio.stop()
    print("END")
