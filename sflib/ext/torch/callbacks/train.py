from ..trainer import TorchTrainerCallback
from torch.nn.utils import clip_grad_norm_


class ClippingGrad(TorchTrainerCallback):
    def __init__(self, max_norm, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def train_before_optimizer_step_callback(self, trainer):
        clip_grad_norm_(trainer._model.parameters(), self.max_norm,
                        self.norm_type)
        
