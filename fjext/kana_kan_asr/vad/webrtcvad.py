import webrtcvad
from .base_vad import BaseVAD, VADState

class WebRTCVAD(BaseVAD):
    def __init__(self, webrtcvad_mode=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vad = webrtcvad.Vad(webrtcvad_mode)

    def _run_vad(self, audio):
        is_speech = self.vad.is_speech(audio.data_bytes, 16000)
        score = 1.0 if is_speech else 0.0
        return is_speech, score

