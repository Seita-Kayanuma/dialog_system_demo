from .base_vad import BaseVAD
import torch

class SileroVAD(BaseVAD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.model = model

        self.buffer_for_vad = []
        self.prev_is_speech = False
        self.prev_score = 0.0

    def _run_vad(self, audio):
        